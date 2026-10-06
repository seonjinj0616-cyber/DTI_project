"""
Step1 결과물(Ki/Kd 분류 모델)을 이용해,
입력한 SMILES 화합물에 대해 인간 단백질 전체(10만여개)를
초고속 스크리닝하여 Top100 단백질을 뽑는 파이프라인.

CPU 전용 (얕은 MLP 추론 + 단일 SMILES 인코딩뿐이라 CPU로 충분).

흐름:
    1. 입력 SMILES -> MoLFormer -> drug_vector (768d, attention_mask 기준 masked mean pooling)
    2. 전체 인간 단백질 벡터(1280d) x drug_vector -> Ki/Kd 확률 계산
    3. Ki 확률 >= 0.5 인 행만 Ki 내림차순으로 정렬해 최대 1000행을 후보 풀로 구성
       (조건을 만족하는 행이 1000개보다 적으면 있는 만큼만)
    4. 풀을 Kd 확률 내림차순으로 정렬(풀 순위)하고, Kd 확률 >= 0.5 인 행만 대상으로
       uniprot_id 별 순위가 가장 높은 대표 1개씩 채택 -> 고유 uniprot_id 최대 100개
       (조건을 만족하는 단백질이 100개보다 적으면 있는 만큼만. 개수를 채우려고 임계값 미만을 넣지 않음)
    5. 원본 CSV(esm2_ready_dataset_cropped_final.csv)의 id / type / length 를 붙여서 저장
       - top100_screening_*.csv  : 고유 uniprot_id Top100 (대표 1개씩)
       - top100_duplicates_*.csv : 대표에서 제외된 같은 uniprot_id 의 다른 변이체 목록

인간 단백질 H5 구조 (whole_human_protein_vector_..._v2.h5):
    vectors/<sequence_hash>              : (1280,) float32 단백질 벡터
    metadata/<sequence_hash>/uniprot_ids : JSON 문자열 UniProt ID 리스트 (같은 서열을 공유하는 ID들)

원본 CSV id 컬럼 매핑 (이번 수정의 핵심):
    H5에는 uniprot_id 만 들어 있고 원본 CSV의 `id`(예: P04217_H52R, WT_P04217,
    WT_P01023_win_191_1215)는 없다. 같은 uniprot_id 라도 wildtype / mutant / window 마다
    서열이 다르고 id 도 다르기 때문에, uniprot_id 만으로는 Top100 안에서 어떤 변이체인지
    구분할 수 없었다.

    -> 원본 CSV의 sequence 를 H5와 같은 방식으로 해시해서 (sequence_hash, uniprot_id) 쌍으로
       원본 행을 찾고, 그 행의 id / type / length / raw_header 를 붙인다.
       (해시 방식은 H5 키와 대조해서 자동으로 찾는다. 원본 CSV에서 (서열, uniprot_id) 조합은
        id 와 1:1 로 확인됨)
    -> 매핑 결과는 consolidated_cache/protein_meta.csv 로 캐싱되어 다음 실행부터는 바로 로딩.
       기존 벡터 캐시(protein_vectors.npy)는 그대로 재사용한다(다시 만들 필요 없음).
"""

import os
import sys
import json
import time
import hashlib
from collections import Counter

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 0. Configuration
# ============================================================

def _find_project_root(marker_name="DTI_project"):
    """
    이 스크립트 파일 위치에서 위로 올라가며 이름이 marker_name 인 폴더를 찾는다.
    DTI_project 폴더 전체를 다른 컴퓨터/다른 드라이브/다른 사용자 계정으로 옮겨도,
    폴더 이름만 같으면 경로를 자동으로 다시 찾는다.
    """

    path = os.path.dirname(os.path.abspath(__file__))

    while True:

        if os.path.basename(path) == marker_name:
            return path

        parent = os.path.dirname(path)

        if parent == path:  # 드라이브/파일시스템 루트까지 올라갔는데 못 찾음
            raise RuntimeError(
                "'%s' 폴더를 찾을 수 없습니다. 이 스크립트가 %s 폴더 안(하위 폴더 포함)에 있는지 확인하세요."
                % (marker_name, marker_name)
            )

        path = parent


PROJECT_DIR = _find_project_root()

# 체크포인트(.pt) 파일은 스크립트가 있는 model 폴더가 아니라 그 안의 model\model 서브폴더에 있음
KIKD_MODEL_DIR = os.path.join(PROJECT_DIR, "model", "model")
KIKD_MODEL_PATH = os.path.join(KIKD_MODEL_DIR, "step1_multitask_kikd_gradnorm_thr7.pt")

PROTEIN_VECTOR_DIR = os.path.join(PROJECT_DIR, "data", "esm2_human_whole_protein_vecterDB")
PROTEIN_VECTOR_H5 = os.path.join(
    PROTEIN_VECTOR_DIR,
    "whole_human_protein_vector_esm2_t33_650M_1280d_sequence_v2.h5"
)

# H5의 원본 데이터 (id, sequence, length, type, raw_header, uniprot_id 컬럼).
ORIGINAL_CSV_PATH = os.path.join(
    PROTEIN_VECTOR_DIR, "esm2_ready_dataset_cropped_final.csv"
)

# 100k+개 h5 개별 데이터셋을 매번 다시 읽으면 느리므로,
# 최초 1회 로딩 후 하나의 배열로 합쳐서 캐싱해둔다.
CONSOLIDATED_CACHE_DIR = os.path.join(PROTEIN_VECTOR_DIR, "consolidated_cache")
CACHE_VECTOR_PATH = os.path.join(CONSOLIDATED_CACHE_DIR, "protein_vectors.npy")
CACHE_ID_PATH = os.path.join(CONSOLIDATED_CACHE_DIR, "protein_uniprot_ids.npy")
# 각 캐시 행에 대응하는 원본 CSV 정보 (sequence_hash, uniprot_id, id, type, length, raw_header)
CACHE_META_PATH = os.path.join(CONSOLIDATED_CACHE_DIR, "protein_meta.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

MOL_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"

MOL_DIM = 768
PROT_DIM = 1280
HIDDEN_DIM = 512

# threshold 없이 확률 순위만으로 선별
TOP_KI = 1000   # 1차: Ki 확률 높은 순 최대 1000행을 후보 풀로 사용
TOP_KD = 100    # 2차: 풀을 Kd 확률 순으로 정렬해 '고유 uniprot_id' 기준 최대 100개 선별

# 확률 임계값: 이 값 미만이면 후보에서 제외 (개수를 채우기 위해 임계값 미만을 넣지 않음)
#   -> 조건을 만족하는 단백질이 TOP_KI / TOP_KD 보다 적으면 있는 만큼만 선별된다.
KI_PROB_THRESHOLD = 0.5
KD_PROB_THRESHOLD = 0.5

# True 로 하면: 고유 uniprot_id 가 TOP_KD 개에 못 미칠 때 Ki 풀을 (Ki >= 임계값 행 범위 안에서)
#              2배씩 늘려서라도 TOP_KD 개를 채우려고 시도한다. (풀이 TOP_KI 행을 넘을 수 있음)
# False(기본): 풀은 최대 TOP_KI 행으로 고정 - Ki Top1000 -> Kd Top100 깔때기를 엄격히 유지
AUTO_EXPAND_POOL = False

INFERENCE_BATCH_SIZE = 8192

# 결과 CSV 맨 끝에 원본 raw_header(단백질/유전자 이름 포함)를 붙일지 여부
INCLUDE_RAW_HEADER = True

# 서열 해시 자동 감지에서 시도할 알고리즘
HASH_CANDIDATES = [
    "sha256", "md5", "sha1", "sha512", "sha3_256", "blake2b", "sha224", "sha384"
]
HASH_DETECT_SAMPLE = 5000


# ============================================================
# 1. Device (CPU 전용)
# ============================================================

device = torch.device("cpu")

# CPU 멀티코어를 최대한 활용 (특히 100k+개 단백질에 대한 배치 추론에 도움)
torch.set_num_threads(os.cpu_count() or 4)

print("=" * 90)
print("Top100 Protein Screening Pipeline (CPU)")
print("=" * 90)
print(f"Device      : {device}")
print(f"CPU threads : {torch.get_num_threads()}")


# ============================================================
# 2. Model 정의 (학습 시와 동일한 구조여야 state_dict가 맞음)
# ============================================================

class MultiTask_KiKd_Classifier(nn.Module):

    def __init__(self, mol_dim=768, prot_dim=1280, hidden_dim=512):
        super().__init__()

        input_dim = mol_dim + prot_dim

        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.ki_head = nn.Linear(256, 1)
        self.kd_head = nn.Linear(256, 1)

    def forward(self, mol_vec, prot_vec):
        combined = torch.cat([mol_vec, prot_vec], dim=1)
        features = self.shared_body(combined)
        ki_logits = self.ki_head(features).squeeze(-1)
        kd_logits = self.kd_head(features).squeeze(-1)
        return ki_logits, kd_logits


# ============================================================
# 3. SMILES -> drug_vector
#
#    vectorDB 생성 스크립트의 molformer_encode와 동일한 방식:
#    attention_mask 기준 masked mean pooling.
# ============================================================

def compute_drug_vector(smiles, mol_tokenizer, mol_model):

    encoded = mol_tokenizer(
        [smiles], padding=True, truncation=True, return_tensors="pt"
    )

    # MoLFormer 내부의 선형 어텐션(MolformerFeatureMap)이 forward 마다 무작위 투영 특징을
    # 새로 뽑아서, eval() 을 걸어도 같은 입력에 매번 다른 벡터가 나온다(재현성 없음).
    # 시드를 고정해서 "무작위" 값 자체를 매번 똑같이 재현되게 만든다.
    torch.manual_seed(42)

    with torch.no_grad():
        outputs = mol_model(**encoded)
        hidden = outputs.last_hidden_state  # (1, L, 768)

    attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
    masked_hidden = hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts

    return pooled.squeeze(0)  # (768,)


# ============================================================
# 4. 원본 CSV id 매핑 (sequence_hash -> 원본 CSV 행)
# ============================================================

def load_original_csv(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"원본 CSV를 찾을 수 없습니다: {csv_path}\n"
            f"  -> 파일 상단의 ORIGINAL_CSV_PATH 를 실제 위치로 수정하세요."
        )

    # utf-8-sig: 파일 맨 앞의 BOM(\ufeff)이 'id' 컬럼명에 섞이는 것을 방지
    # dtype=str + keep_default_na=False: id/uniprot_id 가 NaN 등으로 오인 변환되는 것을 방지
    df = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )

    required = ["id", "sequence", "length", "type", "raw_header", "uniprot_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"원본 CSV에 필요한 컬럼이 없습니다: {missing}\n실제 컬럼: {list(df.columns)}"
        )

    print(f"원본 CSV 로드: {len(df):,}행 (고유 id {df['id'].nunique():,}, "
          f"고유 서열 {df['sequence'].nunique():,}, "
          f"고유 uniprot_id {df['uniprot_id'].nunique():,})")

    return df


def detect_sequence_hash(h5_keys, csv_sequences):
    """
    H5의 sequence_hash 키를 재현하는 해시 함수를 찾는다.
    (알고리즘 후보 x 키 길이 -> H5 키와 가장 많이 일치하는 조합 선택)
    반환: (해시 함수, 알고리즘 이름, 키 길이, 샘플 일치율)
    """

    keys = {str(k).lower() for k in h5_keys}

    key_len = Counter(len(k) for k in keys).most_common(1)[0][0]

    unique_seqs = pd.unique(pd.Series(csv_sequences))[:HASH_DETECT_SAMPLE]

    best = None

    for algo in HASH_CANDIDATES:

        try:
            digest_len = len(hashlib.new(algo, b"x").hexdigest())
        except ValueError:
            continue

        # H5 키가 hexdigest 의 앞부분(prefix)만 쓴 경우도 허용
        if key_len > digest_len:
            continue

        hits = sum(
            1 for s in unique_seqs
            if hashlib.new(algo, s.encode("utf-8")).hexdigest()[:key_len] in keys
        )

        frac = hits / max(len(unique_seqs), 1)

        if best is None or hits > best[0]:
            best = (hits, frac, algo)

    if best is None or best[0] == 0:
        sample_keys = list(h5_keys[:3])
        raise RuntimeError(
            "H5의 sequence_hash 를 재현하는 해시 방식을 찾지 못했습니다.\n"
            f"  H5 키 예시: {sample_keys} (길이 {key_len})\n"
            f"  시도한 알고리즘: {HASH_CANDIDATES}\n"
            "  -> H5 생성 스크립트에서 sequence_hash 를 만든 코드(해시 함수/입력 서열 전처리)를 "
            "확인해서 알려주세요."
        )

    _, frac, algo = best

    def hash_fn(seq):
        return hashlib.new(algo, seq.encode("utf-8")).hexdigest()[:key_len]

    return hash_fn, algo, key_len, frac


def attach_original_ids(seq_hashes, uniprot_ids, csv_df):
    """
    캐시의 각 행 (sequence_hash, uniprot_id) 에 대응하는 원본 CSV 행을 찾아
    id / type / length / raw_header 를 붙인 DataFrame(캐시 행과 같은 순서)을 반환.
    """

    seq_hashes = np.asarray(seq_hashes).astype(str)
    uniprot_ids = np.asarray(uniprot_ids).astype(str)

    h5_unique_hashes = pd.unique(seq_hashes)

    hash_fn, algo, key_len, frac = detect_sequence_hash(
        h5_unique_hashes, csv_df["sequence"]
    )

    print(f"해시 방식 자동 감지: {algo} (H5 키 길이 {key_len}, 샘플 일치율 {frac * 100:.1f}%)")

    csv_hash = csv_df["sequence"].map(hash_fn).to_numpy()
    csv_uniprot = csv_df["uniprot_id"].to_numpy()

    # (sequence_hash, uniprot_id) -> 원본 CSV 행 번호
    lookup = {}
    n_dup_keys = 0

    for i, (h, u) in enumerate(zip(csv_hash, csv_uniprot)):
        key = (h, u)
        if key in lookup:
            n_dup_keys += 1
            continue
        lookup[key] = i

    if n_dup_keys > 0:
        print(f"[참고] 원본 CSV에서 (서열, uniprot_id)가 같은 중복 행 {n_dup_keys:,}개 "
              f"(첫 번째 행의 id 사용)")

    # 해시가 원본에서 딱 1행에만 해당하면, uniprot_id 표기가 조금 달라도 그 행으로 매핑 (보조 규칙)
    hash_counts = Counter(csv_hash.tolist())
    unique_hash_row = {
        h: i for i, h in enumerate(csv_hash) if hash_counts[h] == 1
    }

    row_idx = np.full(len(seq_hashes), -1, dtype=np.int64)
    n_fallback = 0

    for k, (h, u) in enumerate(zip(seq_hashes, uniprot_ids)):

        hit = lookup.get((h.lower(), u))

        if hit is None:
            hit = unique_hash_row.get(h.lower())
            if hit is not None:
                n_fallback += 1

        if hit is not None:
            row_idx[k] = hit

    matched = row_idx >= 0
    n_matched = int(matched.sum())
    n_total = len(row_idx)

    print(f"원본 id 매핑: {n_matched:,} / {n_total:,} "
          f"({100.0 * n_matched / max(n_total, 1):.2f}%)"
          + (f" | 보조 규칙(해시 유일 매칭) 사용 {n_fallback:,}건" if n_fallback else ""))

    if n_matched == 0:
        raise RuntimeError("원본 CSV와 매핑된 행이 하나도 없습니다.")

    if n_matched < n_total:
        bad = np.where(~matched)[0][:5]
        print(f"[경고] 원본 CSV에서 찾지 못한 행 {n_total - n_matched:,}개 -> id 를 빈 값으로 둡니다. "
              f"예: {[(seq_hashes[b][:12] + '...', uniprot_ids[b]) for b in bad]}")

    safe_idx = np.maximum(row_idx, 0)

    meta = pd.DataFrame({
        "sequence_hash": seq_hashes,
        "uniprot_id": uniprot_ids,
    })

    for col in ["id", "type", "length", "raw_header"]:
        meta[col] = np.where(matched, csv_df[col].to_numpy()[safe_idx], "")

    return meta


# ============================================================
# 5. 인간 단백질 벡터 DB 로딩
#    (vectors/<hash> + metadata/<hash>/uniprot_ids 구조를 펼쳐서
#     UniProt ID 단위의 (N, 1280) 배열로 변환, 결과는 캐싱)
# ============================================================

def flatten_h5(h5_path, with_vectors):
    """
    H5를 (sequence_hash, uniprot_id) 단위 행으로 펼친다.
    with_vectors=False 이면 벡터는 읽지 않고 행 순서/ID 정보만 만든다
    (벡터 캐시가 이미 있을 때 원본 id 메타만 추가로 만들기 위함).
    행 순서는 두 모드에서 동일하다.
    """

    print("\nH5 펼치는 중 "
          + ("(벡터 포함 - 최초 1회)" if with_vectors else "(메타데이터만)") + "...")

    with h5py.File(h5_path, "r") as f:

        if "vectors" not in f or "metadata" not in f:
            raise RuntimeError(
                f"예상한 구조('vectors', 'metadata' 그룹)가 없습니다. "
                f"실제 최상위 키: {list(f.keys())}"
            )

        print(f"모델         : {f.attrs.get('model_name', '?')}")
        print(f"embedding_dim: {f.attrs.get('embedding_dim', '?')}")
        print(f"pooling      : {f.attrs.get('pooling', '?')}")
        print(f"version      : {f.attrs.get('version', '?')}")

        vectors_group = f["vectors"]
        metadata_group = f["metadata"]

        sequence_hashes = list(vectors_group.keys())
        print(f"고유 서열 개수: {len(sequence_hashes):,}")

        all_vectors = []
        all_uniprot_ids = []
        all_seq_hashes = []

        n_skipped_no_metadata = 0

        for seq_hash in tqdm(sequence_hashes, desc="H5 펼치는 중"):

            if with_vectors:
                vec = vectors_group[seq_hash][:].astype(np.float32)

                if vec.shape != (PROT_DIM,):
                    raise RuntimeError(
                        f"예상치 못한 벡터 shape: {seq_hash} -> {vec.shape} "
                        f"(기대: ({PROT_DIM},))"
                    )

            if seq_hash not in metadata_group:
                n_skipped_no_metadata += 1
                continue

            uniprot_ids_raw = metadata_group[seq_hash]["uniprot_ids"][()]
            if isinstance(uniprot_ids_raw, bytes):
                uniprot_ids_raw = uniprot_ids_raw.decode("utf-8")
            uniprot_ids = json.loads(uniprot_ids_raw)

            # 동일 서열을 공유하는 UniProt ID 각각에 벡터를 복제해서 펼침
            for uid in uniprot_ids:
                if with_vectors:
                    all_vectors.append(vec)
                all_uniprot_ids.append(uid)
                all_seq_hashes.append(seq_hash)

        if n_skipped_no_metadata > 0:
            print(f"[경고] metadata가 없어 건너뛴 서열: {n_skipped_no_metadata:,}개")

    protein_ids = np.array(all_uniprot_ids)
    seq_hash_arr = np.array(all_seq_hashes)

    protein_vectors = None

    if with_vectors:
        protein_vectors = np.stack(all_vectors, axis=0).astype(np.float32)

    print(f"펼친 후 UniProt 단위 행 개수: {len(protein_ids):,}")

    dup_count = len(protein_ids) - len(set(protein_ids.tolist()))
    if dup_count > 0:
        print(
            f"[참고] UniProt ID 중복 {dup_count:,}건 "
            f"(같은 uniprot_id 의 wildtype/mutant/window 서열이 각각 다른 행 - 원본 id 로 구분됨)"
        )

    return protein_vectors, protein_ids, seq_hash_arr


def _meta_is_aligned(meta_df, protein_ids):
    return (
        len(meta_df) == len(protein_ids)
        and np.array_equal(
            meta_df["uniprot_id"].to_numpy().astype(str),
            np.asarray(protein_ids).astype(str)
        )
    )


def load_protein_database(h5_path):
    """
    반환: protein_vectors (N,1280), protein_ids (N,) uniprot_id, protein_meta (N행 DataFrame)
          protein_meta 컬럼: sequence_hash, uniprot_id, id, type, length, raw_header
    """

    os.makedirs(CONSOLIDATED_CACHE_DIR, exist_ok=True)

    protein_vectors = None
    protein_ids = None
    protein_meta = None

    # ---------------- 1) 벡터 캐시 ----------------------------
    if os.path.exists(CACHE_VECTOR_PATH) and os.path.exists(CACHE_ID_PATH):

        print(f"\n캐시된 병합 배열 발견 -> 빠르게 로딩합니다.")
        print(f"  {CACHE_VECTOR_PATH}")

        protein_vectors = np.load(CACHE_VECTOR_PATH)
        protein_ids = np.load(CACHE_ID_PATH, allow_pickle=False)

        if protein_vectors.shape[1] != PROT_DIM:
            raise RuntimeError(
                f"캐시된 벡터 차원({protein_vectors.shape[1]})이 "
                f"기대값({PROT_DIM})과 다릅니다. 캐시를 삭제하고 다시 실행하세요:\n"
                f"  {CONSOLIDATED_CACHE_DIR}"
            )

        # ---------------- 2) id 메타 캐시 ----------------------
        if os.path.exists(CACHE_META_PATH):

            meta = pd.read_csv(CACHE_META_PATH, dtype=str, keep_default_na=False)

            if _meta_is_aligned(meta, protein_ids):
                protein_meta = meta
                print(f"원본 id 메타 캐시 로딩: {CACHE_META_PATH}")
            else:
                print("[경고] 메타 캐시가 벡터 캐시와 맞지 않아 다시 만듭니다.")

        if protein_meta is None:

            print("\n원본 id 메타를 새로 만듭니다 (벡터 캐시는 그대로 사용).")

            _, h5_uniprot_ids, h5_seq_hashes = flatten_h5(h5_path, with_vectors=False)

            if np.array_equal(h5_uniprot_ids.astype(str), protein_ids.astype(str)):
                csv_df = load_original_csv(ORIGINAL_CSV_PATH)
                protein_meta = attach_original_ids(h5_seq_hashes, h5_uniprot_ids, csv_df)
                protein_meta.to_csv(CACHE_META_PATH, index=False)
                print(f"메타 캐시 저장: {CACHE_META_PATH}")
            else:
                print("[경고] 벡터 캐시의 행 순서가 현재 H5와 달라서 전체를 다시 만듭니다.")
                protein_vectors = None
                protein_ids = None

    # ---------------- 3) 캐시가 없거나 어긋난 경우: 전체 생성 --------
    if protein_vectors is None:

        print("\n최초 로딩: 개별 데이터셋을 하나의 배열로 병합합니다 "
              "(다음 실행부터는 캐시를 사용해 훨씬 빨라집니다)...")

        protein_vectors, protein_ids, h5_seq_hashes = flatten_h5(h5_path, with_vectors=True)

        csv_df = load_original_csv(ORIGINAL_CSV_PATH)
        protein_meta = attach_original_ids(h5_seq_hashes, protein_ids, csv_df)

        print(f"\n캐시 저장 중... ({CONSOLIDATED_CACHE_DIR})")
        np.save(CACHE_VECTOR_PATH, protein_vectors)
        np.save(CACHE_ID_PATH, protein_ids)
        protein_meta.to_csv(CACHE_META_PATH, index=False)
        print("캐시 저장 완료. 다음 실행부터는 이 캐시를 사용합니다.")

    n_proteins = protein_vectors.shape[0]
    print(f"\n로드 완료: protein_vectors shape={protein_vectors.shape}, "
          f"protein_ids 개수={len(protein_ids)}, 메타 행 개수={len(protein_meta)}")

    assert n_proteins == len(protein_ids) == len(protein_meta), \
        "protein_vectors / protein_ids / protein_meta 개수가 일치하지 않습니다."

    return protein_vectors, protein_ids, protein_meta


# ============================================================
# 5-1. 고유 uniprot_id 기준 Top-N 선별 + 중복 목록 생성
# ============================================================

def select_top_unique_proteins(ki_probs, kd_probs, uniprot_ids, top_ki, top_kd,
                               ki_threshold=0.5, kd_threshold=0.5, auto_expand=False):
    """
    1) Ki 확률 >= ki_threshold 인 행만 Ki 내림차순으로 정렬해 상위 최대 top_ki 행을 후보 풀로 구성
       (조건을 만족하는 행이 top_ki 보다 적으면 있는 만큼만)
    2) 풀을 Kd 확률 내림차순으로 정렬 (= 풀 순위)
    3) 풀 순위 순서대로 훑으면서 Kd 확률 >= kd_threshold 인 행만, uniprot_id 별로 순위가 가장 높은
       대표 1개씩 채택 (최대 top_kd 개). Kd 가 임계값 아래로 내려가면 거기서 종료.

    auto_expand=True 이면, 고유 uniprot_id 가 top_kd 개에 못 미칠 때 풀 크기를 2배씩 늘려서
    (Ki >= ki_threshold 인 행 범위 안에서) 다시 시도한다.

    반환:
        selected    : 채택된 행의 전역 인덱스 (최종 순위 순, 길이 <= top_kd)
        ranked_pool : 사용한 풀 전체의 전역 인덱스 (Kd 내림차순 = 풀 순위 순)
        pool_size   : 최종적으로 사용한 풀 크기(행 수)
        stats       : {"n_ki_eligible": Ki>=임계값 전체 행 수,
                       "n_kd_pass_in_pool": 풀 안에서 Kd>=임계값 인 행 수}
    """

    uniprot_ids = np.asarray(uniprot_ids).astype(str)

    # stable: 확률이 같을 때 항상 같은 순서가 나오도록
    ki_order = np.argsort(-ki_probs, kind="stable")

    # 내림차순 정렬이므로 Ki >= 임계값 인 행은 정확히 앞쪽 n_ki_eligible 개
    n_ki_eligible = int((ki_probs >= ki_threshold).sum())

    pool_size = min(top_ki, n_ki_eligible)

    while True:

        pool_idx = ki_order[:pool_size]
        ranked_pool = pool_idx[np.argsort(-kd_probs[pool_idx], kind="stable")]

        selected = []
        seen = set()

        for gi in ranked_pool:

            # Kd 내림차순이므로 여기서부터는 나머지도 전부 임계값 미만
            if kd_probs[gi] < kd_threshold:
                break

            uid = uniprot_ids[gi]

            if uid in seen:
                continue

            seen.add(uid)
            selected.append(int(gi))

            if len(selected) == top_kd:
                break

        if (
            len(selected) >= top_kd
            or not auto_expand
            or pool_size >= n_ki_eligible
        ):
            break

        pool_size = min(n_ki_eligible, pool_size * 2)

    stats = {
        "n_ki_eligible": n_ki_eligible,
        "n_kd_pass_in_pool": int((kd_probs[ranked_pool] >= kd_threshold).sum()),
    }

    return np.array(selected, dtype=np.int64), ranked_pool, pool_size, stats


DUPLICATE_COLUMNS = [
    "uniprot_id",
    "kept_rank", "kept_id", "kept_type", "kept_pool_rank", "kept_kd_prob",
    "removed_id", "removed_type", "removed_length",
    "removed_pool_rank", "removed_ki_prob", "removed_kd_prob",
]


def build_duplicate_table(selected, ranked_pool, uniprot_ids, ki_probs, kd_probs,
                          protein_meta, include_raw_header=True, kd_threshold=0.0):
    """
    최종 선별된 uniprot_id 마다, 후보 풀 안에 있던 "대표가 아닌" 같은 uniprot_id 행들을 정리.
    (대표 = 풀 순위가 가장 높은 행)
    Kd 확률이 kd_threshold 미만인 변이체는 중복 제거가 아니라 임계값 탈락이므로 목록에서 제외한다.
    (풀 안의 행은 이미 Ki >= 임계값)
    """

    uniprot_ids = np.asarray(uniprot_ids).astype(str)

    columns = list(DUPLICATE_COLUMNS)
    if include_raw_header:
        columns.append("removed_raw_header")

    kept_gi = {uniprot_ids[gi]: int(gi) for gi in selected}
    kept_rank = {uniprot_ids[gi]: r for r, gi in enumerate(selected, start=1)}
    pool_rank = {int(gi): r for r, gi in enumerate(ranked_pool, start=1)}

    dup_idx = np.array(
        [
            int(gi) for gi in ranked_pool
            if uniprot_ids[gi] in kept_gi
            and int(gi) != kept_gi[uniprot_ids[gi]]
            and kd_probs[gi] >= kd_threshold
        ],
        dtype=np.int64
    )

    if len(dup_idx) == 0:
        return pd.DataFrame(columns=columns)

    dup_uids = uniprot_ids[dup_idx]
    kept_idx = np.array([kept_gi[u] for u in dup_uids], dtype=np.int64)

    dup_meta = protein_meta.iloc[dup_idx]
    kept_meta = protein_meta.iloc[kept_idx]

    data = {
        "uniprot_id": dup_uids,
        "kept_rank": [kept_rank[u] for u in dup_uids],
        "kept_id": kept_meta["id"].to_numpy(),
        "kept_type": kept_meta["type"].to_numpy(),
        "kept_pool_rank": [pool_rank[int(g)] for g in kept_idx],
        "kept_kd_prob": kd_probs[kept_idx],
        "removed_id": dup_meta["id"].to_numpy(),
        "removed_type": dup_meta["type"].to_numpy(),
        "removed_length": dup_meta["length"].to_numpy(),
        "removed_pool_rank": [pool_rank[int(g)] for g in dup_idx],
        "removed_ki_prob": ki_probs[dup_idx],
        "removed_kd_prob": kd_probs[dup_idx],
    }

    if include_raw_header:
        data["removed_raw_header"] = dup_meta["raw_header"].to_numpy()

    df = pd.DataFrame(data)

    return df.sort_values(
        ["kept_rank", "removed_pool_rank"], kind="stable"
    ).reset_index(drop=True)


# ============================================================
# 6. Main pipeline
# ============================================================

def run_screening(target_smiles):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_start = time.time()

    # --------------------------------------------------------
    # 6-1. MoLFormer 로드 + SMILES -> drug_vector
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("1. MoLFormer 로딩 + SMILES 인코딩")
    print("=" * 90)
    print(f"Target SMILES: {target_smiles}")

    mol_tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
    mol_model.eval()
    for p in mol_model.parameters():
        p.requires_grad = False

    drug_vector = compute_drug_vector(target_smiles, mol_tokenizer, mol_model)
    print(f"drug_vector shape: {tuple(drug_vector.shape)}")

    del mol_model  # 더 안 쓰므로 메모리 반환

    # --------------------------------------------------------
    # 6-2. 인간 단백질 벡터 DB 로딩 (+ 원본 id 메타)
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("2. 인간 단백질 벡터 DB 로딩")
    print("=" * 90)

    protein_vectors_np, protein_ids, protein_meta = load_protein_database(PROTEIN_VECTOR_H5)
    n_proteins = protein_vectors_np.shape[0]

    protein_vectors = torch.from_numpy(protein_vectors_np)

    # --------------------------------------------------------
    # 6-3. Ki/Kd 모델 로드
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("3. Ki/Kd 분류 모델 로딩")
    print("=" * 90)
    print(f"경로: {KIKD_MODEL_PATH}")

    kikd_model = MultiTask_KiKd_Classifier(
        mol_dim=MOL_DIM, prot_dim=PROT_DIM, hidden_dim=HIDDEN_DIM
    )

    # weights_only=False: PyTorch 2.6부터 기본값이 True로 바뀌면서
    # 체크포인트 안의 numpy 스칼라 등 순수 텐서가 아닌 객체를 거부함.
    # 본인이 직접 학습시킨 신뢰 가능한 파일이므로 명시적으로 허용.
    kikd_ckpt = torch.load(KIKD_MODEL_PATH, map_location="cpu", weights_only=False)
    kikd_model.load_state_dict(kikd_ckpt["model_state_dict"])
    kikd_model.eval()

    # --------------------------------------------------------
    # 6-4. 전체 인간 단백질에 대해 Ki/Kd 확률 계산 (배치 처리)
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(f"4. 전체 {n_proteins:,}개 단백질에 대해 Ki/Kd 확률 계산")
    print("=" * 90)

    ki_probs = np.zeros(n_proteins, dtype=np.float32)
    kd_probs = np.zeros(n_proteins, dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(
            range(0, n_proteins, INFERENCE_BATCH_SIZE), desc="Ki/Kd 추론"
        ):
            end = min(start + INFERENCE_BATCH_SIZE, n_proteins)

            prot_batch = protein_vectors[start:end]
            mol_batch = drug_vector.unsqueeze(0).expand(end - start, -1)

            ki_logits, kd_logits = kikd_model(mol_batch, prot_batch)

            ki_probs[start:end] = torch.sigmoid(ki_logits).numpy()
            kd_probs[start:end] = torch.sigmoid(kd_logits).numpy()

    print("Ki/Kd 확률 계산 완료.")

    del kikd_model

    # --------------------------------------------------------
    # 6-5. Ki >= 임계값 -> 최대 Top 1000 풀 -> Kd >= 임계값 -> 고유 uniprot_id 최대 100개
    #      (같은 uniprot_id 는 순위가 가장 높은 대표 1개만 채택.
    #       임계값을 못 넘는 후보는 제외하며, 개수를 채우기 위해 넣지 않는다)
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(f"5. Ki >= {KI_PROB_THRESHOLD} 최대 {TOP_KI}행 풀 -> "
          f"Kd >= {KD_PROB_THRESHOLD} 고유 uniprot_id 최대 {TOP_KD}개 선별")
    print("=" * 90)

    final_indices, ranked_pool, pool_size, stats = select_top_unique_proteins(
        ki_probs, kd_probs, protein_ids, TOP_KI, TOP_KD,
        ki_threshold=KI_PROB_THRESHOLD, kd_threshold=KD_PROB_THRESHOLD,
        auto_expand=AUTO_EXPAND_POOL
    )

    print(f"Ki >= {KI_PROB_THRESHOLD} 인 행 : {stats['n_ki_eligible']:,} / {n_proteins:,}")

    print(f"Ki 후보 풀 크기 : {pool_size:,}행"
          + (f" (Ki 임계값 통과 행이 {TOP_KI:,}개보다 적어 전부 사용)"
             if stats["n_ki_eligible"] < TOP_KI else "")
          + (f" (고유 uniprot_id 부족으로 {TOP_KI:,}행에서 확장됨)" if pool_size > TOP_KI else ""))

    if pool_size > 0:

        pass_mask = kd_probs[ranked_pool] >= KD_PROB_THRESHOLD
        pool_unique = len(np.unique(protein_ids[ranked_pool]))
        pass_unique = len(np.unique(protein_ids[ranked_pool[pass_mask]]))

        print(f"풀 내 고유 uniprot_id : {pool_unique:,}개")
        print(f"풀 Ki 확률 범위 : {ki_probs[ranked_pool].min():.4f} ~ {ki_probs[ranked_pool].max():.4f}")
        print(f"풀 안에서 Kd >= {KD_PROB_THRESHOLD} 인 행 : {stats['n_kd_pass_in_pool']:,}개 "
              f"(고유 uniprot_id {pass_unique:,}개)")

    if len(final_indices) > 0:

        scan_depth = int(np.where(ranked_pool == final_indices[-1])[0][0]) + 1

        print(f"Kd 순위를 {scan_depth:,}번째 행까지 내려가며 고유 {len(final_indices)}개 확보 "
              f"(그 사이 중복 행 {scan_depth - len(final_indices)}개 건너뜀)")
        print(f"최종 고유 단백질 {len(final_indices):,}개 선별 완료. "
              f"(Ki 확률 범위: {ki_probs[final_indices].min():.4f} ~ {ki_probs[final_indices].max():.4f} | "
              f"Kd 확률 범위: {kd_probs[final_indices].min():.4f} ~ {kd_probs[final_indices].max():.4f})")

    else:
        print(f"[경고] Ki >= {KI_PROB_THRESHOLD} 이고 Kd >= {KD_PROB_THRESHOLD} 인 단백질이 없습니다. "
              f"빈 결과가 저장됩니다.")

    if 0 < len(final_indices) < TOP_KD:
        print(f"[참고] 임계값을 통과한 고유 단백질이 {TOP_KD}개보다 적어 "
              f"{len(final_indices)}개만 선별했습니다 (정상 동작).")

    # --------------------------------------------------------
    # 6-6. 결과 정리 + 저장 (Top100 CSV + 중복 목록 CSV)
    #      (final_indices는 이미 Kd 확률 내림차순 = 최종 순위 순)
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("6. 결과 정리")
    print("=" * 90)

    dup_df = build_duplicate_table(
        final_indices, ranked_pool, protein_ids, ki_probs, kd_probs,
        protein_meta, include_raw_header=INCLUDE_RAW_HEADER,
        kd_threshold=KD_PROB_THRESHOLD
    )

    removed_count = dup_df.groupby("uniprot_id").size().to_dict() if len(dup_df) else {}

    final_meta = protein_meta.iloc[final_indices].reset_index(drop=True)
    final_uniprot = protein_ids[final_indices]

    columns = {
        "id": final_meta["id"].to_numpy(),
        "uniprot_id": final_uniprot,
        "type": final_meta["type"].to_numpy(),
        "length": final_meta["length"].to_numpy(),
        "ki_prob": ki_probs[final_indices],
        "kd_prob": kd_probs[final_indices],
        "n_variants_removed": [int(removed_count.get(str(u), 0)) for u in final_uniprot],
    }

    if INCLUDE_RAW_HEADER:
        columns["raw_header"] = final_meta["raw_header"].to_numpy()

    result_df = pd.DataFrame(columns)

    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))

    # 안전 확인: 고유 uniprot_id 여야 함
    assert result_df["uniprot_id"].is_unique, "최종 결과에 중복 uniprot_id 가 남아 있습니다."

    n_empty_id = int((result_df["id"] == "").sum())

    print(f"Top {len(result_df)} 고유 uniprot_id 수: {result_df['uniprot_id'].nunique()}")
    print(f"제거된 중복 행(같은 uniprot_id 의 다른 변이체): {len(dup_df):,}개 "
          f"(해당 uniprot_id {dup_df['uniprot_id'].nunique() if len(dup_df) else 0}개)")

    if n_empty_id > 0:
        print(f"[경고] 원본 id 를 찾지 못한 행 {n_empty_id}개")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_smiles_tag = "".join(c if c.isalnum() else "_" for c in target_smiles)[:40]

    output_path = os.path.join(
        OUTPUT_DIR, f"top{TOP_KD}_screening_{safe_smiles_tag}_{timestamp}.csv"
    )
    duplicates_path = os.path.join(
        OUTPUT_DIR, f"top{TOP_KD}_duplicates_{safe_smiles_tag}_{timestamp}.csv"
    )

    # utf-8-sig: 엑셀에서 바로 열어도 한글/특수문자가 깨지지 않도록
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    dup_df.to_csv(duplicates_path, index=False, encoding="utf-8-sig")

    print(f"\nTop{TOP_KD} 결과 저장 완료 : {output_path}")
    print(f"중복 목록 저장 완료      : {duplicates_path}"
          + ("  (중복 없음 - 헤더만 저장)" if len(dup_df) == 0 else ""))

    print(f"\n[미리보기 - Kd 확률 내림차순 상위 10개]")
    preview_cols = [c for c in result_df.columns if c != "raw_header"]
    print(result_df[preview_cols].head(10).to_string(index=False))

    total_elapsed = time.time() - total_start
    print(f"\n총 소요 시간: {total_elapsed:.1f}초")

    return result_df


# ============================================================
# 7. Entry point
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        target_smiles = sys.argv[1]
    else:
        target_smiles = input("검색할 화합물의 SMILES를 입력하세요: ").strip()

    if not target_smiles:
        raise ValueError("SMILES가 입력되지 않았습니다.")

    run_screening(target_smiles)