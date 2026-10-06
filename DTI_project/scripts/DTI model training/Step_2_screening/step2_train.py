"""
Step 2 학습: Compound x Protein 2D token embedding
             -> Compound -> Protein 단방향 Cross-Attention (Query = compound, Key/Value = protein)
             -> IC50 paffinity 회귀 (단일 task)

이 모델의 용도
    Stage 2 : Top100 단백질의 IC50 paffinity 를 예측 -> 예측값 높은 순으로 재랭킹 -> Top20
    Stage 3 : Top20 의 cross-attention weight 로 Attention Peak Score 계산
              -> 가중치가 한곳에 뚜렷하게 집중되는 단백질 5개를 최종 선택
    (추론은 step2_inference.py, 모델 정의는 step2_model.py)

입력 데이터
    1) 토큰 임베딩 캐시 H5  : compound/, protein/ (offsets, lengths, embeddings ...)
    2) Pair 인덱스 CSV      : compound_id, protein_id, task_idx(0=Ki,1=Kd,2=IC50), paffinity ...

학습 방식
    - 배치를 task 비율대로 구성(MultiTaskExactRatioBatchSampler)
    - 기본은 IC50 단일 task (ACTIVE_TASKS = ["ic50"]). 이때 GradNorm 은 자동으로 생략된다.
    - ACTIVE_TASKS 에 task 를 2개 이상 넣으면 GradNorm 으로 task별 loss weight 를 자동 조절
    - 검증 점수 기준 best epoch 자동 선택 + early stopping + LR plateau 감소

속도 관련
    화합물 토큰은 gzip H5 랜덤 접근이 매우 느려서, 압축 없는 float16 .npy(memmap)로
    1회 변환해서 사용한다 (약 85GB). `python step2_train.py build_cache` 로 미리 생성 가능.
"""

import os
import gc
import sys
import time
import random

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    ConcatDataset,
    Sampler
)

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from step2_model import CrossAttentionDTIRegressor, COMP_DIM, PROT_DIM, ARCHITECTURE


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

# ---- Paths --------------------------------------------------
BASE_DIR = "/home/team5/workspace/sj/homo_protein_seq_fasta"

TOKEN_CACHE_H5 = os.path.join(
    BASE_DIR, "step2_2D_vector_token_embedding_cache_float32.h5"
)

# 캐시 생성 스크립트의 OUTPUT_INDEX (이름을 바꿨다면 여기도 맞춰서 수정)
INTERACTION_INDEX_CSV = os.path.join(BASE_DIR, "DTI_interactions_indexed.csv")

USE_COMPOUND_MEMMAP = True
COMPOUND_MEMMAP_PATH = os.path.join(BASE_DIR, "step2_compound_tokens_float16.npy")

SAVE_PATH = "/home/team5/workspace/sj/step2_c2p_crossattn_ic50.pt"

# ---- Tasks --------------------------------------------------
# 캐시 생성 스크립트의 INPUT_FILES 순서와 동일
TASK_IDS = {"ki": 0, "kd": 1, "ic50": 2}

# 학습할 task. IC50 하나만 회귀 예측한다.
# (["ki", "kd", "ic50"] 처럼 여러 개를 넣으면 multi-head + GradNorm 으로 학습)
ACTIVE_TASKS = ["ic50"]

# ---- Token caps ---------------------------------------------
#   단백질: 서열 최대 1024 aa + <cls> + <eos> = 1026 토큰
#   화합물: MoLFormer tokenizer max length(약 202) 이하
MAX_COMPOUND_TOKENS = 256
MAX_PROTEIN_TOKENS = 1026

PROTEIN_CACHE_DTYPE = np.float16    # 단백질 토큰 RAM 캐시 (float16: 약 6.7GB)

# ---- Split --------------------------------------------------
# "random" / "compound" / "protein"
SPLIT_MODE = "random"
TRAIN_FRACTION = 0.8

# ---- Model --------------------------------------------------
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 2
FFN_DIM = 512
ATTN_DROPOUT = 0.1
BODY_DROPOUT = 0.3

# ---- Training -----------------------------------------------
MAX_EPOCHS = 50
BATCH_SIZE = 128
GRADNORM_ALPHA = 1.5

MODEL_LR = 3e-4
WEIGHT_LR = 1e-3
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

HUBER_DELTA = 1.0
INITIAL_LOSS_BATCHES = 50

USE_AMP = True               # bf16 autocast (지원되는 GPU에서만 적용)

# ---- DataLoader / 호스트 메모리 -------------------------------
# 배치 1개(단백질 토큰 128 x 1026 x 1280 fp16)가 약 330MB 라서,
#   "worker 수 x prefetch_factor" 개의 배치가 동시에 공유메모리(/dev/shm)에 쌓이고
#   pin_memory 는 배치 크기가 매번 달라 pinned 메모리를 계속 붙잡아둔다.
# 이 값들이 호스트 RAM 사용량을 좌우한다. (RAM 이 넉넉하지 않으면 더 줄이세요)
NUM_WORKERS = 4              # 학습 loader worker 수
VAL_NUM_WORKERS = 2          # 검증 loader worker 수 (검증 때만 잠깐 사용)
PREFETCH_FACTOR = 2          # worker 당 미리 만들어두는 배치 수
PIN_MEMORY = False           # 가변 크기 대형 배치에서는 pinned memory 누적 위험

MEMORY_LOG_EVERY = 500       # N 배치마다 메모리 사용량 출력 (0 이면 끔, psutil 필요)

# ---- 빠른 점검용 (smoke test: `python step2_train.py smoke`) --------
TRAIN_BATCH_LIMIT = None     # epoch 당 최대 학습 배치 수 (None = 전체)
VAL_BATCH_LIMIT = None       # 검증 최대 배치 수 (None = 전체)
VAL_SHUFFLE = False          # smoke 에서는 True (모든 task 가 섞여 나오도록)

# ---- Early stopping / LR scheduler --------------------------
EARLY_STOP_PATIENCE = 7
MIN_DELTA = 1e-4

USE_LR_SCHEDULER = True
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LR = 1e-6

# Best epoch 선택 기준 (active task 평균)
#   "rmse"     : -mean(RMSE)       (값 예측 정확도)
#   "spearman" : mean(Spearman)    (순위 정확도)
SELECTION_METRIC = "rmse"


# ============================================================
# 1. Seed
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# 2. Compound memmap 변환 (gzip H5 -> 압축 없는 float16 .npy)
# ============================================================

def build_compound_memmap(h5_path, out_path):

    print("\n" + "=" * 70)
    print("화합물 토큰 임베딩 -> float16 memmap 변환 (최초 1회)")
    print("=" * 70)

    tmp_path = out_path + ".partial"

    with h5py.File(h5_path, "r") as f:

        emb = f["compound/embeddings"]
        n_rows, dim = emb.shape

        print(f"행 수 : {n_rows:,} | dim : {dim}")
        print(f"필요 디스크 : 약 {n_rows * dim * 2 / 1024**3:.1f} GB")
        print(f"출력 : {out_path}")

        mm = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float16, shape=(n_rows, dim)
        )

        chunk = 4096 * 8   # H5 chunk(4096행)의 배수 -> 순차 읽기 효율적
        max_abs = 0.0

        for s in tqdm(range(0, n_rows, chunk), desc="변환"):
            e = min(s + chunk, n_rows)
            block = emb[s:e]
            max_abs = max(max_abs, float(np.abs(block).max()))
            mm[s:e] = block.astype(np.float16)

        mm.flush()
        del mm

    if max_abs > 6.0e4:
        print(f"[경고] 임베딩 최대 절댓값 {max_abs:.1f} -> float16 범위를 넘을 수 있습니다.")

    # 끝까지 성공했을 때만 정식 이름으로 확정
    os.replace(tmp_path, out_path)

    print(f"변환 완료 (max |value| = {max_abs:.3f})")


# ============================================================
# 3. Embedding store
# ============================================================

class EmbeddingStore:
    """
    - 단백질 토큰: 전체를 RAM에 적재 (약 6.7GB @ float16)
    - 화합물 토큰: float16 memmap(.npy)에서 필요한 구간만 읽음
    """

    def __init__(self, h5_path):

        self.h5_path = h5_path
        self._h5 = None
        self._comp_mm = None
        self._comp_fd = None
        self.comp_data_offset = 0

        self.use_memmap = USE_COMPOUND_MEMMAP

        print(f"\n[Embedding cache] {h5_path}")

        with h5py.File(h5_path, "r") as f:

            self.comp_offsets = f["compound/offsets"][:].astype(np.int64)
            self.comp_lengths = f["compound/lengths"][:].astype(np.int64)

            self.prot_offsets = f["protein/offsets"][:].astype(np.int64)
            self.prot_lengths = f["protein/lengths"][:].astype(np.int64)

            self.n_compounds = len(self.comp_offsets)
            self.n_proteins = len(self.prot_offsets)

            # 캐시 생성 스크립트 가정 확인: ids == 행 번호
            assert np.array_equal(
                f["compound/ids"][:], np.arange(self.n_compounds)
            ), "compound/ids 가 0..N-1 이 아닙니다."
            assert np.array_equal(
                f["protein/ids"][:], np.arange(self.n_proteins)
            ), "protein/ids 가 0..N-1 이 아닙니다."

            print(f"Compounds : {self.n_compounds:,} "
                  f"(token rows: {f['compound/embeddings'].shape[0]:,}, "
                  f"처리 완료: {(self.comp_lengths > 0).sum():,})")

            n_prot_rows = f["protein/embeddings"].shape[0]

            print(f"Proteins  : {self.n_proteins:,} "
                  f"(token rows: {n_prot_rows:,}, "
                  f"처리 완료: {(self.prot_lengths > 0).sum():,})")

            print(f"토큰 수: 화합물 max {self.comp_lengths.max()} / "
                  f"단백질 max {self.prot_lengths.max()}")

            if self.comp_lengths.max() > MAX_COMPOUND_TOKENS:
                print(f"[경고] 일부 화합물이 {MAX_COMPOUND_TOKENS} 토큰을 넘어 잘립니다.")
            if self.prot_lengths.max() > MAX_PROTEIN_TOKENS:
                print(f"[경고] 일부 단백질이 {MAX_PROTEIN_TOKENS} 토큰을 넘어 잘립니다.")

            print(f"단백질 토큰 임베딩을 RAM에 적재 중 "
                  f"(dtype={np.dtype(PROTEIN_CACHE_DTYPE).name})...")

            start = time.time()

            self.prot_embeddings = np.empty(
                (n_prot_rows, PROT_DIM), dtype=PROTEIN_CACHE_DTYPE
            )

            chunk = 200_000

            for s in range(0, n_prot_rows, chunk):
                e = min(s + chunk, n_prot_rows)
                self.prot_embeddings[s:e] = (
                    f["protein/embeddings"][s:e].astype(PROTEIN_CACHE_DTYPE)
                )

            print(f"적재 완료 ({time.time() - start:.1f} sec, "
                  f"{self.prot_embeddings.nbytes / 1024**3:.2f} GB)")

        if self.use_memmap:

            if not os.path.exists(COMPOUND_MEMMAP_PATH):
                build_compound_memmap(h5_path, COMPOUND_MEMMAP_PATH)

            mm = np.load(COMPOUND_MEMMAP_PATH, mmap_mode="r")

            expected_rows = int(self.comp_lengths.sum())

            if mm.shape != (expected_rows, COMP_DIM):
                raise RuntimeError(
                    f"memmap shape {mm.shape} 이 H5({expected_rows}, {COMP_DIM})와 "
                    f"다릅니다. 캐시 생성 이후 H5가 바뀌었다면 memmap 파일을 "
                    f"삭제하고 다시 실행하세요:\n  {COMPOUND_MEMMAP_PATH}"
                )

            # 압축 없는 .npy 이므로 (헤더 길이 + 행 x 차원 x 2byte) 위치를 직접 계산해 읽는다.
            # -> mmap 으로 페이지를 매핑하지 않아서 worker 의 RSS 가 계속 커지지 않고,
            #    read-only 배열 경고도 생기지 않는다.
            self.comp_data_offset = int(mm.offset)

            expected_size = self.comp_data_offset + expected_rows * COMP_DIM * 2
            actual_size = os.path.getsize(COMPOUND_MEMMAP_PATH)

            if actual_size != expected_size:
                raise RuntimeError(
                    f"memmap 파일 크기({actual_size:,}B)가 예상({expected_size:,}B)과 다릅니다. "
                    f"파일을 삭제하고 다시 생성하세요:\n  {COMPOUND_MEMMAP_PATH}"
                )

            del mm

            print(f"화합물 토큰: float16 .npy 사용 ({COMPOUND_MEMMAP_PATH}, "
                  f"{'pread' if hasattr(os, 'pread') else 'memmap'} 방식 읽기)")

        else:
            print("[경고] 화합물 토큰을 gzip H5에서 직접 읽습니다 (매우 느림).")

    # 파일 핸들/memmap은 pickle/fork 시 넘기지 않고 worker에서 새로 연다
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_comp_mm"] = None
        state["_comp_fd"] = None
        return state

    def get_compound(self, row):

        off = self.comp_offsets[row]
        length = min(int(self.comp_lengths[row]), MAX_COMPOUND_TOKENS)

        if self.use_memmap:

            if hasattr(os, "pread"):

                if self._comp_fd is None:
                    self._comp_fd = os.open(COMPOUND_MEMMAP_PATH, os.O_RDONLY)

                row_bytes = COMP_DIM * 2
                nbytes = length * row_bytes
                pos = self.comp_data_offset + int(off) * row_bytes

                buf = os.pread(self._comp_fd, nbytes, pos)

                if len(buf) != nbytes:
                    raise RuntimeError(
                        f"화합물 토큰 읽기 실패: row={row}, 요청 {nbytes}B, 실제 {len(buf)}B"
                    )

                # copy(): 쓰기 가능한 작은 배열로 만들어 torch 경고 방지 (샘플당 수십 KB)
                return np.frombuffer(buf, dtype=np.float16).reshape(length, COMP_DIM).copy()

            # pread 가 없는 OS(Windows 등): memmap 으로 읽되 복사본 반환
            if self._comp_mm is None:
                self._comp_mm = np.load(COMPOUND_MEMMAP_PATH, mmap_mode="r")

            return np.array(self._comp_mm[off:off + length])

        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path, "r",
                rdcc_nbytes=256 * 1024**2, rdcc_nslots=1_000_003
            )

        return self._h5["compound/embeddings"][off:off + length]

    def get_protein(self, row):

        off = self.prot_offsets[row]
        length = min(int(self.prot_lengths[row]), MAX_PROTEIN_TOKENS)

        return self.prot_embeddings[off:off + length]


# ============================================================
# 4. Pair table + Dataset
# ============================================================

def load_interaction_index(csv_path):

    print(f"\n[Interaction index] {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    needed = ["compound_id", "protein_id", "task_idx", "paffinity"]

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in needed if c not in header]

    if missing:
        raise RuntimeError(
            f"CSV에 필요한 컬럼이 없습니다: {missing}\n실제 컬럼: {header}"
        )

    df = pd.read_csv(csv_path, usecols=needed)

    print(f"전체 interaction : {len(df):,}")
    print(df["task_idx"].value_counts().sort_index().to_string())

    return df


def select_task_pairs(index_df, store, task_idx, name):

    sub = index_df[index_df["task_idx"] == task_idx]

    n_total = len(sub)

    sub = sub.dropna(subset=["paffinity"])

    comp_rows = sub["compound_id"].to_numpy(dtype=np.int64)
    prot_rows = sub["protein_id"].to_numpy(dtype=np.int64)
    labels = sub["paffinity"].to_numpy(dtype=np.float32)

    in_range = (
        (comp_rows >= 0) & (comp_rows < store.n_compounds)
        & (prot_rows >= 0) & (prot_rows < store.n_proteins)
    )

    valid = in_range.copy()

    # 임베딩이 아직 생성되지 않은 항목(lengths == 0)은 제외
    valid[in_range] = (
        (store.comp_lengths[comp_rows[in_range]] > 0)
        & (store.prot_lengths[prot_rows[in_range]] > 0)
    )

    comp_rows = comp_rows[valid]
    prot_rows = prot_rows[valid]
    labels = labels[valid]

    print(f"\n[{name}] task_idx={task_idx}")
    print(f"  pair 수(원본)      : {n_total:,}")
    print(f"  사용 가능 pair 수   : {len(labels):,}")

    if len(labels) == 0:
        raise RuntimeError(f"{name}: 사용 가능한 pair가 없습니다.")

    n_dup = int(pd.Series(list(zip(comp_rows, prot_rows))).duplicated().sum())

    print(f"  (compound, protein) 중복 pair : {n_dup:,}")
    print(f"  paffinity mean={labels.mean():.3f} std={labels.std():.3f} "
          f"min={labels.min():.3f} max={labels.max():.3f}")
    print(f"  고유 화합물 {len(np.unique(comp_rows)):,} / "
          f"고유 단백질 {len(np.unique(prot_rows)):,}")

    return comp_rows, prot_rows, labels


class PairDataset(Dataset):

    def __init__(self, store, comp_rows, prot_rows, labels, task_id):
        self.store = store
        self.comp_rows = comp_rows
        self.prot_rows = prot_rows
        self.labels = labels
        self.task_id = task_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        comp = torch.from_numpy(
            np.ascontiguousarray(self.store.get_compound(self.comp_rows[idx]))
        )
        prot = torch.from_numpy(
            np.ascontiguousarray(self.store.get_protein(self.prot_rows[idx]))
        )

        return (
            comp,
            prot,
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.task_id, dtype=torch.long)
        )


def collate_pairs(batch):

    comps, prots, ys, tasks = zip(*batch)

    B = len(batch)
    Lc = max(c.shape[0] for c in comps)
    Lp = max(p.shape[0] for p in prots)

    comp_pad = torch.zeros(B, Lc, COMP_DIM, dtype=comps[0].dtype)
    comp_valid = torch.zeros(B, Lc, dtype=torch.bool)

    prot_pad = torch.zeros(B, Lp, PROT_DIM, dtype=prots[0].dtype)
    prot_valid = torch.zeros(B, Lp, dtype=torch.bool)

    for i, (c, p) in enumerate(zip(comps, prots)):
        comp_pad[i, :c.shape[0]] = c
        comp_valid[i, :c.shape[0]] = True
        prot_pad[i, :p.shape[0]] = p
        prot_valid[i, :p.shape[0]] = True

    return (
        comp_pad, comp_valid,
        prot_pad, prot_valid,
        torch.stack(ys), torch.stack(tasks)
    )


def split_indices(comp_rows, prot_rows, mode, train_fraction, seed):

    rng = np.random.RandomState(seed)
    n = len(comp_rows)

    if mode == "random":
        perm = rng.permutation(n)
        n_train = int(train_fraction * n)
        return np.sort(perm[:n_train]), np.sort(perm[n_train:])

    if mode in ("compound", "protein"):

        groups = comp_rows if mode == "compound" else prot_rows
        unique_groups = rng.permutation(np.unique(groups))

        n_train_groups = int(train_fraction * len(unique_groups))
        train_groups = unique_groups[:n_train_groups]

        is_train = np.isin(groups, train_groups)

        return np.where(is_train)[0], np.where(~is_train)[0]

    raise ValueError(f"알 수 없는 SPLIT_MODE: {mode}")


# ============================================================
# 5. Multi-task exact-ratio batch sampler
# ============================================================

class MultiTaskExactRatioBatchSampler(Sampler):
    """
    매 배치가 task별 데이터 크기 비율을 그대로 유지하도록 구성
    (모든 task가 매 배치에 최소 1개 이상 포함).
    """

    def __init__(self, index_lists, batch_size=128):

        self.index_lists = [
            torch.tensor(ix, dtype=torch.long) for ix in index_lists
        ]

        self.batch_size = batch_size

        sizes = np.array([len(ix) for ix in self.index_lists], dtype=np.float64)

        counts = np.maximum(1, np.round(batch_size * sizes / sizes.sum())).astype(int)

        # 합이 batch_size가 되도록 가장 큰 task에서 보정
        diff = batch_size - int(counts.sum())
        counts[int(np.argmax(sizes))] += diff

        if counts.min() < 1:
            raise ValueError("batch_size가 task 수에 비해 너무 작습니다.")

        self.counts = counts.tolist()

        self.num_batches = min(
            len(ix) // c for ix, c in zip(self.index_lists, self.counts)
        )

    def __iter__(self):

        shuffled = [ix[torch.randperm(len(ix))] for ix in self.index_lists]

        for b in range(self.num_batches):

            parts = [
                sh[b * c:(b + 1) * c]
                for sh, c in zip(shuffled, self.counts)
            ]

            batch = torch.cat(parts).tolist()

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================
# 6. Metrics (regression)
# ============================================================

METRIC_NAMES = ["RMSE", "MAE", "Pearson", "Spearman", "R2"]


def regression_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return {name: float("nan") for name in METRIC_NAMES}

    err = y_pred - y_true

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "R2": r2
    }


def compute_selection_score(metrics_by_task):

    if SELECTION_METRIC == "rmse":
        key, sign = "RMSE", -1.0
    elif SELECTION_METRIC == "spearman":
        key, sign = "Spearman", 1.0
    else:
        raise ValueError(f"알 수 없는 SELECTION_METRIC: {SELECTION_METRIC}")

    values = [m[key] for m in metrics_by_task.values()]

    if any(np.isnan(v) for v in values):
        return float("-inf")

    return sign * float(np.mean(values))


# ============================================================
# 7. Helpers
# ============================================================

def print_gpu_info():

    print("\n" + "=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA 사용 불가 -> CPU로 실행됩니다 (매우 느림).")
        print("=" * 70)
        return

    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"CUDA : {torch.version.cuda}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)


def move_batch(batch, device):

    comp, comp_valid, prot, prot_valid, y, tasks = batch

    return (
        comp.to(device, non_blocking=True),
        comp_valid.to(device, non_blocking=True),
        prot.to(device, non_blocking=True),
        prot_valid.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        tasks.to(device, non_blocking=True)
    )


def make_autocast(device, use_amp):

    enabled = (
        use_amp
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=enabled
    ), enabled


# ============================================================
# 8. Evaluation / Initial loss
# ============================================================

def log_memory(tag=""):
    """
    호스트 메모리 진단용 출력. (psutil 이 없으면 조용히 건너뜀)
    available 이 epoch 내내 계속 줄어들면 어딘가에서 메모리가 누적되고 있다는 신호.
    """

    try:
        import psutil
    except ImportError:
        return

    try:
        proc = psutil.Process(os.getpid())
        main_rss = proc.memory_info().rss / 1024**3

        worker_rss = 0.0
        n_workers = 0

        for child in proc.children(recursive=True):
            try:
                worker_rss += child.memory_info().rss
                n_workers += 1
            except psutil.NoSuchProcess:
                pass

        vm = psutil.virtual_memory()

        shm = 0.0
        if os.path.isdir("/dev/shm"):
            shm = psutil.disk_usage("/dev/shm").used / 1024**3

        print(f"[MEM]{tag} | main {main_rss:.1f}GB | workers {n_workers}개 RSS합 "
              f"{worker_rss / 1024**3:.1f}GB(공유 중복 포함) | /dev/shm {shm:.1f}GB | "
              f"시스템 available {vm.available / 1024**3:.1f}/{vm.total / 1024**3:.1f}GB",
              flush=True)

    except Exception as e:
        print(f"[MEM] 측정 실패: {e}")


def evaluate(model, val_loader, device, criterion, use_amp, task_ids):

    model.eval()

    n_tasks = len(task_ids)

    y_true = [[] for _ in range(n_tasks)]
    y_pred = [[] for _ in range(n_tasks)]
    loss_sum = [0.0] * n_tasks
    count = [0] * n_tasks

    ctx, _ = make_autocast(device, use_amp)

    with torch.no_grad():

        for val_idx, batch in enumerate(val_loader):

            if VAL_BATCH_LIMIT is not None and val_idx >= VAL_BATCH_LIMIT:
                break

            comp, comp_valid, prot, prot_valid, y, tasks = move_batch(batch, device)

            with ctx:
                preds = model(comp, comp_valid, prot, prot_valid)

            preds = preds.float()

            for j, tid in enumerate(task_ids):

                mask = (tasks == tid)

                if not mask.any():
                    continue

                p = preds[mask, j]
                t = y[mask]

                c = int(mask.sum().item())

                loss_sum[j] += criterion(p, t).item() * c
                count[j] += c

                y_true[j].extend(t.cpu().numpy())
                y_pred[j].extend(p.cpu().numpy())

    losses = [loss_sum[j] / max(count[j], 1) for j in range(n_tasks)]
    metrics = [regression_metrics(y_true[j], y_pred[j]) for j in range(n_tasks)]

    return losses, metrics


def calculate_initial_losses(model, train_loader, device, criterion, use_amp, task_ids):

    print(f"\nInitial Loss L_i(0) 계산 중 (첫 {INITIAL_LOSS_BATCHES} batch)...")

    model.eval()

    n_tasks = len(task_ids)

    loss_sum = [0.0] * n_tasks
    count = [0] * n_tasks

    ctx, _ = make_autocast(device, use_amp)

    start = time.time()

    with torch.no_grad():

        for i, batch in enumerate(train_loader):

            if i >= INITIAL_LOSS_BATCHES:
                break

            comp, comp_valid, prot, prot_valid, y, tasks = move_batch(batch, device)

            with ctx:
                preds = model(comp, comp_valid, prot, prot_valid)

            preds = preds.float()

            for j, tid in enumerate(task_ids):

                mask = (tasks == tid)

                if not mask.any():
                    continue

                c = int(mask.sum().item())

                loss_sum[j] += criterion(preds[mask, j], y[mask]).item() * c
                count[j] += c

    initial_losses = torch.tensor(
        [loss_sum[j] / count[j] for j in range(n_tasks)],
        device=device, dtype=torch.float32
    )

    print(f"L_i(0) 계산 완료 ({time.time() - start:.1f} sec)")

    for name, value in zip(ACTIVE_TASKS, initial_losses.tolist()):
        print(f"{name.upper():<5}: {value:.6f}")

    return initial_losses


# ============================================================
# 9. Checkpoint
# ============================================================

def build_model_config():
    return {
        "task_names": list(ACTIVE_TASKS),
        "comp_dim": COMP_DIM,
        "prot_dim": PROT_DIM,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "ffn_dim": FFN_DIM,
        "attn_dropout": ATTN_DROPOUT,
        "body_dropout": BODY_DROPOUT,
        "architecture": ARCHITECTURE      # Compound -> Protein 단방향 cross-attention
    }


def save_best_checkpoint(
    path, model_state, log_task_weights, initial_losses,
    best_epoch, best_score, best_metrics, best_val_losses,
    history, target_means
):

    n_tasks = len(ACTIVE_TASKS)

    task_weights = n_tasks * torch.softmax(log_task_weights.detach().cpu(), dim=0)

    torch.save(
        {
            "model_state_dict": model_state,
            "model_config": build_model_config(),

            "task_names": list(ACTIVE_TASKS),
            "task_ids": {n: TASK_IDS[n] for n in ACTIVE_TASKS},

            "max_compound_tokens": MAX_COMPOUND_TOKENS,
            "max_protein_tokens": MAX_PROTEIN_TOKENS,

            "log_task_weights": log_task_weights.detach().cpu(),
            "task_weights": task_weights,
            "initial_losses": initial_losses.detach().cpu(),
            "target_means": target_means,

            "split_mode": SPLIT_MODE,
            "batch_size": BATCH_SIZE,
            "alpha": GRADNORM_ALPHA,
            "max_epochs": MAX_EPOCHS,
            "seed": SEED,

            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_metric": SELECTION_METRIC,
            "best_val_losses": best_val_losses,
            "best_metrics": best_metrics,

            "history": history
        },
        path
    )


# ============================================================
# 10. Main training
# ============================================================

def train():

    print("\n" + "=" * 70)
    print("Step 2: Compound -> Protein Cross-Attention Regression")
    print(f"Tasks : {ACTIVE_TASKS} | Query = compound, Key/Value = protein (단방향)")
    print("Best-Epoch Selection" + (" + GradNorm" if len(ACTIVE_TASKS) > 1 else " (단일 task: GradNorm 생략)"))
    print("=" * 70)

    for name in ACTIVE_TASKS:
        if name not in TASK_IDS:
            raise ValueError(f"알 수 없는 task: {name} (가능: {list(TASK_IDS)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_gpu_info()

    n_tasks = len(ACTIVE_TASKS)
    task_ids = [TASK_IDS[n] for n in ACTIVE_TASKS]

    # --------------------------------------------------------
    # Embedding store + pair tables + split
    # --------------------------------------------------------

    store = EmbeddingStore(TOKEN_CACHE_H5)

    index_df = load_interaction_index(INTERACTION_INDEX_CSV)

    train_subsets, val_subsets = [], []
    target_means = {}

    for name in ACTIVE_TASKS:

        c, p, y = select_task_pairs(index_df, store, TASK_IDS[name], name.upper())

        dataset = PairDataset(store, c, p, y, task_id=TASK_IDS[name])

        tr_idx, va_idx = split_indices(c, p, SPLIT_MODE, TRAIN_FRACTION, SEED)

        train_subsets.append(Subset(dataset, tr_idx.tolist()))
        val_subsets.append(Subset(dataset, va_idx.tolist()))

        target_means[name] = float(y[tr_idx].mean())

    del index_df
    gc.collect()

    print(f"\nSplit mode : {SPLIT_MODE}")

    for name, tr, va in zip(ACTIVE_TASKS, train_subsets, val_subsets):
        print(f"{name.upper():<5} train / val : {len(tr):,} / {len(va):,}")

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_dataset = ConcatDataset(train_subsets)

    index_lists = []
    offset = 0

    for tr in train_subsets:
        index_lists.append(list(range(offset, offset + len(tr))))
        offset += len(tr)

    batch_sampler = MultiTaskExactRatioBatchSampler(index_lists, batch_size=BATCH_SIZE)

    print("\nBatch configuration")
    for name, c in zip(ACTIVE_TASKS, batch_sampler.counts):
        print(f"{name.upper():<5} / batch : {c}")
    print(f"Batches / epoch : {len(batch_sampler):,}")

    def make_loader_kwargs(num_workers, persistent):
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            collate_fn=collate_pairs
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = PREFETCH_FACTOR
            kwargs["persistent_workers"] = persistent
        return kwargs

    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler,
        **make_loader_kwargs(NUM_WORKERS, persistent=True)
    )

    # 검증 loader 는 persistent 로 두지 않는다 (검증이 끝나면 worker 를 정리해서 메모리 반환)
    val_loader = DataLoader(
        ConcatDataset(val_subsets),
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        **make_loader_kwargs(VAL_NUM_WORKERS, persistent=False)
    )

    batch_mb = BATCH_SIZE * MAX_PROTEIN_TOKENS * PROT_DIM * 2 / 1024**2
    in_flight_gb = (
        max(NUM_WORKERS, VAL_NUM_WORKERS) * PREFETCH_FACTOR * batch_mb / 1024
    )

    print(f"\n메모리 예상: 배치 1개 최대 약 {batch_mb:.0f}MB | "
          f"미리 쌓아두는 배치(공유메모리) 최대 약 {in_flight_gb:.1f}GB "
          f"(학습 {NUM_WORKERS} x {PREFETCH_FACTOR} / 검증 {VAL_NUM_WORKERS} x {PREFETCH_FACTOR}) | "
          f"단백질 토큰 캐시 상주 약 {store.prot_embeddings.nbytes / 1024**3:.1f}GB")
    log_memory(" (학습 시작 전)")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CrossAttentionDTIRegressor(**build_model_config()).to(device)

    # head bias를 train 평균 paffinity로 초기화 (수렴 가속)
    with torch.no_grad():
        for j, name in enumerate(ACTIVE_TASKS):
            model.heads[j].bias.fill_(target_means[name])

    print(f"\nModel parameters : "
          f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer_model = optim.AdamW(
        model.parameters(), lr=MODEL_LR, weight_decay=WEIGHT_DECAY
    )

    # GradNorm weights (log weights -> softmax -> positive, sum = n_tasks)
    log_task_weights = nn.Parameter(torch.zeros(n_tasks, device=device))
    optimizer_weights = optim.Adam([log_task_weights], lr=WEIGHT_LR)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    scheduler = None

    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_model, mode="max",
            factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
        )

    autocast_ctx, amp_enabled = make_autocast(device, USE_AMP)

    print(f"AMP(bf16) : {'ON' if amp_enabled else 'OFF'}")

    initial_losses = calculate_initial_losses(
        model, train_loader, device, criterion, USE_AMP, task_ids
    )

    # ========================================================
    # Training
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING START")
    print(f"Max epochs : {MAX_EPOCHS} | Early stopping patience : "
          f"{EARLY_STOP_PATIENCE} | Selection : {SELECTION_METRIC}")
    print("=" * 70)

    total_start = time.time()

    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    best_log_task_weights = None
    best_metrics = None
    best_val_losses = None

    epochs_without_improvement = 0
    stopped_early = False
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        total_train_loss = 0.0
        n_batches_run = 0

        for batch_idx, batch in enumerate(train_loader):

            if TRAIN_BATCH_LIMIT is not None and batch_idx >= TRAIN_BATCH_LIMIT:
                break

            n_batches_run += 1

            comp, comp_valid, prot, prot_valid, y, tasks = move_batch(batch, device)

            # =================================================
            # Single forward (GradNorm + model update 공용)
            # =================================================

            with autocast_ctx:
                preds = model(comp, comp_valid, prot, prot_valid)

            preds = preds.float()

            losses = []

            for j, tid in enumerate(task_ids):
                mask = (tasks == tid)
                losses.append(criterion(preds[mask, j], y[mask]))

            # -------------------------------------------------
            # GradNorm: task weight 업데이트 (task가 2개 이상일 때)
            # -------------------------------------------------

            if n_tasks > 1:

                task_weights = n_tasks * torch.softmax(log_task_weights, dim=0)

                shared_layer = model.get_shared_layer()

                G = []

                for j in range(n_tasks):

                    grad_j = torch.autograd.grad(
                        task_weights[j] * losses[j], shared_layer,
                        retain_graph=True, create_graph=True
                    )[0]

                    G.append(torch.norm(grad_j, p=2))

                G = torch.stack(G)
                G_avg = G.mean()

                ratios = torch.stack([
                    losses[j].detach() / initial_losses[j]
                    for j in range(n_tasks)
                ])

                r = ratios / (ratios.mean() + 1e-8)

                targets = (G_avg * torch.pow(r, GRADNORM_ALPHA)).detach()

                loss_gradnorm = torch.abs(G - targets).sum()

                # task weight에 대한 gradient만 계산 (model 파라미터 grad는 건드리지 않음)
                grad_w = torch.autograd.grad(
                    loss_gradnorm, log_task_weights, retain_graph=True
                )[0]

                optimizer_weights.zero_grad(set_to_none=True)
                log_task_weights.grad = grad_w
                optimizer_weights.step()

                new_weights = (
                    n_tasks * torch.softmax(log_task_weights, dim=0)
                ).detach()

            else:
                new_weights = torch.ones(1, device=device)

            # -------------------------------------------------
            # Model update (같은 forward graph 재사용, 갱신된 weight 사용)
            # -------------------------------------------------

            weighted_loss = sum(
                new_weights[j] * losses[j] for j in range(n_tasks)
            )

            optimizer_model.zero_grad(set_to_none=True)

            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer_model.step()

            total_train_loss += weighted_loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"\rEpoch {epoch:02d}/{MAX_EPOCHS} "
                    f"| Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss {weighted_loss.item():.4f}",
                    end="", flush=True
                )

            if MEMORY_LOG_EVERY and (batch_idx + 1) % MEMORY_LOG_EVERY == 0:
                print()
                log_memory(f" epoch {epoch} batch {batch_idx + 1}")

        print()
        log_memory(f" epoch {epoch} 학습 종료 (검증 시작 전)")

        # ====================================================
        # Validation
        # ====================================================

        val_losses, val_metrics = evaluate(
            model, val_loader, device, criterion, USE_AMP, task_ids
        )

        log_memory(f" epoch {epoch} 검증 종료")

        metrics_by_task = dict(zip(ACTIVE_TASKS, val_metrics))
        losses_by_task = dict(zip(ACTIVE_TASKS, val_losses))

        score = compute_selection_score(metrics_by_task)

        current_lr = optimizer_model.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(score if np.isfinite(score) else -1e9)

        # ====================================================
        # Best check
        # ====================================================

        improved = score > best_score + MIN_DELTA

        if improved:

            best_score = score
            best_epoch = epoch
            best_metrics = {k: dict(v) for k, v in metrics_by_task.items()}
            best_val_losses = dict(losses_by_task)
            best_log_task_weights = log_task_weights.detach().clone()

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        current_weights = (
            n_tasks * torch.softmax(log_task_weights, dim=0)
        ).detach().cpu().numpy()

        avg_train_loss = total_train_loss / max(n_batches_run, 1)

        history.append({
            "epoch": epoch,
            "lr": float(current_lr),
            "train_loss": float(avg_train_loss),
            "val_losses": losses_by_task,
            "val_metrics": metrics_by_task,
            "score": float(score),
            "task_weights": {
                n: float(w) for n, w in zip(ACTIVE_TASKS, current_weights)
            },
            "is_best": bool(improved)
        })

        # ====================================================
        # Report
        # ====================================================

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        print("\n" + "=" * 70)
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} 완료"
              + ("   *** NEW BEST ***" if improved else ""))
        print(f"Epoch time : {epoch_time / 60:.2f} min | "
              f"Elapsed : {total_elapsed / 3600:.2f} h | LR : {current_lr:.2e}")
        print(f"Train Loss : {avg_train_loss:.6f}")

        print("\n[GradNorm Weights] "
              + " | ".join(f"{n.upper()} : {w:.4f}"
                           for n, w in zip(ACTIVE_TASKS, current_weights)))

        for name in ACTIVE_TASKS:
            print(f"\n[{name.upper()}]  val loss {losses_by_task[name]:.6f}")
            for m in METRIC_NAMES:
                print(f"  {m:<9}: {metrics_by_task[name][m]:.4f}")

        print(f"\n[Model Selection : {SELECTION_METRIC}]")
        print(f"Score : {score:.4f} | Best : {best_score:.4f} (epoch {best_epoch})")
        print(f"No-improve : {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if torch.cuda.is_available():
            print(f"GPU mem : allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                  f" / reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        print("=" * 70)

        if improved:

            save_best_checkpoint(
                SAVE_PATH, best_state, best_log_task_weights, initial_losses,
                best_epoch, best_score, best_metrics, best_val_losses,
                history, target_means
            )

            print(f"[Checkpoint] best model 저장: {SAVE_PATH}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:

            print(f"\nEarly stopping: {EARLY_STOP_PATIENCE} epoch 동안 개선 없음 "
                  f"-> epoch {epoch}에서 종료")

            stopped_early = True
            break

    # ========================================================
    # Finalize
    # ========================================================

    if best_state is None:
        raise RuntimeError(
            "유효한 검증 점수를 얻은 epoch이 없습니다 (지표가 NaN인지 확인)."
        )

    model.load_state_dict(best_state)

    save_best_checkpoint(
        SAVE_PATH, best_state, best_log_task_weights, initial_losses,
        best_epoch, best_score, best_metrics, best_val_losses,
        history, target_means
    )

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"실행 epoch : {len(history)} / {MAX_EPOCHS}"
          + (" (early stopping)" if stopped_early else " (max epoch 도달)"))
    print(f"Best epoch : {best_epoch} | Best score : {best_score:.4f}")

    for name in ACTIVE_TASKS:
        m = best_metrics[name]
        print(f"{name.upper():<5} RMSE {m['RMSE']:.4f} | MAE {m['MAE']:.4f} | "
              f"Spearman {m['Spearman']:.4f} | R2 {m['R2']:.4f}")

    print(f"Total time : {total_time / 3600:.2f} hours")
    print(f"Saved (best model) : {SAVE_PATH}")

    if not stopped_early and best_epoch >= MAX_EPOCHS - 2:
        print("\n[참고] best epoch이 max epoch 부근입니다. "
              "MAX_EPOCHS를 늘려서 재학습을 고려하세요.")

    print("=" * 70)

    del train_loader, val_loader, train_dataset, model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "build_cache":
        if os.path.exists(COMPOUND_MEMMAP_PATH):
            print(f"이미 존재합니다: {COMPOUND_MEMMAP_PATH}")
        else:
            build_compound_memmap(TOKEN_CACHE_H5, COMPOUND_MEMMAP_PATH)
    elif len(sys.argv) > 1 and sys.argv[1] == "smoke":

        # 전체 epoch 를 돌리기 전에, 학습 -> 검증 -> 체크포인트 저장까지 전 경로를
        # 몇 분 안에 점검하는 모드 (best 모델은 *_smoke.pt 로 따로 저장됨)
        MAX_EPOCHS = 2
        TRAIN_BATCH_LIMIT = 40
        VAL_BATCH_LIMIT = 10
        VAL_SHUFFLE = True
        MEMORY_LOG_EVERY = 20
        SAVE_PATH = SAVE_PATH.replace(".pt", "_smoke.pt")

        print("=" * 70)
        print(f"SMOKE TEST: {MAX_EPOCHS} epoch x 학습 {TRAIN_BATCH_LIMIT} 배치 / 검증 {VAL_BATCH_LIMIT} 배치")
        print(f"저장 경로: {SAVE_PATH}")
        print("=" * 70)

        train()

    else:
        train()
