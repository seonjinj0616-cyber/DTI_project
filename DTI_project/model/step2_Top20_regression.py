"""
Step 1 + Step 2 통합 실행 스크립트 (CPU / Windows)

    python step2_inference.py --smiles "<SMILES>"

    이 파일 하나만 실행하면 Step 1(protein_Top100_classification.py)이 먼저 자동 실행되어
    인간 단백질 전체에서 Top100 을 뽑고, 그 결과를 이어서 아래 Step 2 로 처리한다.
    (이미 뽑아둔 Top100 CSV 를 재사용하려면 --top100-csv 로 지정 -> Step 1 은 건너뜀)

    필요한 파일 (모두 MODEL_DIR = DTI_project/model 폴더에 함께 둔다):
        step2_Top20_regression.py 는 DTI_project/model 에,
        step2_model.py 는 DTI_project/model/script 에,
        protein_Top100_classification.py 는 DTI_project/model 에,
        step1_multitask_kikd_gradnorm_thr7.pt / step2_c2p_crossattn_ic50.pt 는 DTI_project/model/model 에 둔다

Step 2 추론 파이프라인 (IC50 재랭킹 -> Attention Peak 기반 최종 선택)

    Step 1 결과(Top100: id, uniprot_id) + 입력 SMILES
        |
        v  Top100 의 id 로 원본 CSV(esm2_ready_dataset_cropped_final.csv)에서 sequence 매칭
        |     (id 는 wildtype / mutant / window 를 구분하는 고유 키이므로 uniprot_id 가 아니라
        |      id 로 서열을 찾는다. uniprot_id 는 교차 검증에만 사용)
        v  MoLFormer / ESM-2 토큰 임베딩 (캐시 생성 때와 동일한 방식)
        |
        v  [Stage 2 ] Compound -> Protein 단방향 cross-attention 모델로 IC50 paffinity 예측
        |             예측값 높은 순 -> Top100 재랭킹 -> Top N(20)
        v  [Stage 3a] Top N 각각의 cross-attention weight(화합물 토큰 -> 단백질 residue)로
        |             Attention Peak Score 계산
        v  [Stage 3b] Peak Score 가 가장 뚜렷한(가중치가 한곳에 집중되는) 단백질 FINAL_N(5)개를 최종 선택
        v  [Stage 3c] 최종 단백질에 대해 attention + gradient + ablation 으로 상세 site 분석

Attention Peak Score
    a_j : 화합물 토큰 전체 / 모든 층에 걸쳐 평균한 attention 을, <cls>/<eos> 를 뺀 residue 축에서
          합이 1 이 되도록 정규화한 값 (residue j 가 받은 가중치 비율)

    window_enrichment (기본) = 연속 W residue 창 하나에 몰린 attention 질량의 최댓값 / (W / L)
        -> attention 이 균등하면 1.0, 한 구간에 몰릴수록 커진다. 균등 기대값(W/L)으로 나누므로
           서열 길이가 다른 단백질끼리도 비교할 수 있다.
    그 밖에 함께 저장하는 지표
        peak_mass             : 같은 창에 몰린 attention 질량 자체 (0~1)
        peak_zscore           : 단일 residue 최댓값이 평균에서 표준편차 몇 배 위인지
        entropy_concentration : 1 - 정규화 엔트로피 (전체 분포가 얼마나 뾰족한지)
    선택 기준은 --peak-method 로 바꿀 수 있다.

출력 (out_dir/stage2_<timestamp>/)
    rerank_top100.csv                 : 후보 전체의 예측값 / rank_score / 순위 / Peak Score / 최종 선택 여부
    sequence_match_report.csv         : Top100 각 행이 원본 CSV 의 어느 행에 어떻게 매칭됐는지
    top20_attention_peak_scores.csv   : Top N 의 Peak Score 지표와 순위(peak_rank), peak 구간/서열
    top20_attention_profile.csv       : Top N 의 residue 별 attention
    top20_site_comparison.csv         : Top N 의 peak 원본 좌표, 알려진 site 와의 겹침/거리/우연 확률, 판정 상태
    final5_proteins.csv               : 최종 선택된 단백질 (site 비교 요약 컬럼 포함)
    detail_residue_scores.csv         : 상세 분석 대상의 residue 별 attention/gradient/ablation 점수
    detail_sites.csv                  : 상세 분석 대상의 high-attention site 구간 + 방법 간 일치도
    detail_site_embeddings.npz        : site residue 들의 ESM-2 임베딩 평균(1280d) -> 이후 유사도 계산용
    run_config.json                   : 실행 설정

주의
    - attention 은 "모델이 어디를 봤는가"이지 "예측의 원인"과 동일하지 않을 수 있다.
      그래서 gradient / ablation 을 함께 계산하고, 방법 간 Spearman 일치도를 같이 저장한다.
    - 서열은 학습 때와 동일하게 앞 1024 aa 까지만 사용한다. 원본 CSV 는 이미 1024 aa 이하로
      crop 되어 있어(length 최대 1024) 실제로 잘리는 서열은 없다.
"""

import os
import re
import sys
import gc
import glob
import json
import time
import argparse
import importlib

import numpy as np
import pandas as pd

import torch

from scipy.stats import rankdata, spearmanr


# ============================================================
# 0. 프로젝트 경로 (DTI_project 폴더를 다른 위치로 옮겨도 자동으로 다시 찾음)
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

# step2_model.py 는 이 스크립트와 같은 폴더가 아니라 model\script 서브폴더에 있음
# -> import 하기 전에 그 폴더를 sys.path 에 추가해야 한다
STEP2_SCRIPT_DIR = os.path.join(PROJECT_DIR, "model", "script")

if STEP2_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, STEP2_SCRIPT_DIR)

from step2_model import CrossAttentionDTIRegressor, ARCHITECTURE


# ============================================================
# 0-2. Configuration
# ============================================================

# 스크립트(.py)가 있는 폴더
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

# 예측 모델(.pt 체크포인트)이 있는 폴더 -- model 폴더가 아니라 그 안의 model 서브폴더
# (step1 체크포인트와 같은 패턴으로 가정. 실제 위치가 다르면 알려주세요.)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "model")

# 원본 CSV(esm2_ready_dataset_cropped_final.csv)와 인간 단백질 벡터 DB 가 있는 폴더
BASE_DIR = os.path.join(PROJECT_DIR, "data", "esm2_human_whole_protein_vecterDB")

# Step 2 모델 체크포인트 (Compound->Protein 단방향, IC50 회귀. 서버에서 학습한 파일을 이 경로로 복사)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "step2_c2p_crossattn_ic50.pt")

# Step 1 스크립트 모듈 이름 (MODEL_DIR 안의 protein_Top100_classification.py)
STEP1_MODULE = "protein_Top100_classification"

# Step1 H5 의 원본 데이터 (id, sequence, length, type, raw_header, uniprot_id 컬럼).
# Top100 의 id 로 여기서 sequence 를 가져온다. 서버상의 실제 위치에 맞게 수정하세요.
ORIGINAL_CSV_PATH = os.path.join(BASE_DIR, "esm2_ready_dataset_cropped_final.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

# 토큰 임베딩 캐시 생성 때와 같은 모델 (Step1 스크립트와 같은 이름을 써서 HuggingFace 캐시 중복 다운로드 방지)
MOL_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# CPU 에서는 배치를 작게 (ESM-2 650M, 1024 aa 서열 기준 메모리 절약)
ESM_BATCH_SIZE = 4

TOP_N = 20                    # IC50 예측값 기준 재랭킹 후 남길 개수
FINAL_N = 5                   # Attention Peak Score 로 최종 선택할 개수

# ---- Attention Peak Score ---------------------------------------
PEAK_WINDOW = 15              # 가중치가 몰렸는지 볼 연속 residue 창 크기
# 최종 선택 기준: "window_enrichment"(기본) / "window_mass" / "zscore" / "entropy"
PEAK_METHOD = "window_enrichment"

# 상세 분석(attention + gradient + ablation, CPU 에서 오래 걸림) 대상
#   "final": 최종 선택된 FINAL_N 개만 (기본)   "top": Top N 전부
DETAIL_SCOPE = "final"

# ---- Ablation ------------------------------------------------
ABLATION_WINDOW = 11          # 한 번에 가리는 residue 수
ABLATION_STRIDE = 5
ABLATION_CHUNK = 32           # 한 번에 forward 하는 masked 변형 수

# ---- Site 추출 ------------------------------------------------
# 각 방법 점수를 percentile(0~1)로 바꾼 뒤 가중 평균 (0 이면 해당 방법 제외)
SITE_WEIGHTS = {
    "attention": 1.0,
    "gradient": 1.0,
    "ablation": 1.0
}

SMOOTH_WINDOW = 5             # 이동평균 창 (residue 단위)
SITE_TOP_FRACTION = 0.10      # 상위 10% residue 를 site 후보로
SITE_MERGE_GAP = 2            # 이 간격 이하로 떨어진 구간은 하나로 합침
SITE_MIN_LENGTH = 3
MAX_SITES_PER_PROTEIN = 5


# ============================================================
# 1. Input helpers
# ============================================================

def load_original_sequences(csv_path):
    """원본 CSV 에서 서열 매칭에 필요한 컬럼만 읽는다 (id 는 고유 키)."""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"원본 CSV를 찾을 수 없습니다: {csv_path}\n"
            f"  -> --original-csv 로 경로를 지정하거나 파일 상단의 ORIGINAL_CSV_PATH 를 수정하세요."
        )

    # utf-8-sig: 맨 앞 BOM 이 'id' 컬럼명에 섞이는 것을 방지
    # dtype=str + keep_default_na=False: id/uniprot_id 가 NaN 등으로 오인 변환되는 것을 방지
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    required = ["id", "sequence", "uniprot_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"원본 CSV에 필요한 컬럼이 없습니다: {missing}\n실제 컬럼: {list(df.columns)}"
        )

    if not df["id"].is_unique:
        raise RuntimeError("원본 CSV의 id 가 고유하지 않습니다. id 로 서열을 특정할 수 없습니다.")

    df["sequence"] = df["sequence"].str.strip().str.upper()

    print(f"원본 CSV 로드: {len(df):,}행 ({csv_path})")

    return df


def match_sequences(top_df, orig_df):
    """
    Top100 각 행에 원본 CSV의 sequence 를 붙인다.

    1순위: id 로 정확히 매칭 (Step1 결과 CSV 에 id 가 있을 때) + uniprot_id / length 교차 검증
    2순위: id 컬럼이 없는 예전 결과 CSV 면 uniprot_id 로 매칭하되 wildtype 행을 우선 사용 (경고 출력)

    반환: (서열이 확보된 top_df, 매칭 리포트 DataFrame)
    """

    top_df = top_df.copy()
    orig_by_id = orig_df.set_index("id")

    report = []
    seqs, matched_ids = [], []

    has_id = "id" in top_df.columns

    if not has_id:
        print("[경고] Top100 CSV 에 id 컬럼이 없어 uniprot_id 로 매칭합니다. "
              "같은 uniprot_id 의 변이체/window 가 여러 개면 wildtype 행을 우선 사용하므로, "
              "정확한 매칭을 위해 id 가 포함된 Step1 결과 CSV 를 사용하세요.")

    for i, row in top_df.iterrows():

        uid = str(row["uniprot_id"])
        given_id = str(row["id"]) if has_id else ""

        status = ""
        orig_row = None

        if has_id and given_id in orig_by_id.index:

            orig_row = orig_by_id.loc[given_id]
            matched_id = given_id
            status = "id_exact"

            if str(orig_row["uniprot_id"]) != uid:
                status = "id_exact_UNIPROT_MISMATCH"

            elif "length" in top_df.columns and str(row["length"]).strip() not in ("", "nan"):
                try:
                    if int(float(row["length"])) != len(orig_row["sequence"]):
                        status = "id_exact_LENGTH_MISMATCH"
                except ValueError:
                    pass

        elif has_id:
            status = "id_NOT_FOUND"
            matched_id = ""

        else:
            cand = orig_df[orig_df["uniprot_id"] == uid]

            if len(cand) == 0:
                status = "uniprot_NOT_FOUND"
                matched_id = ""
            else:
                wt = cand[cand["id"].str.startswith("WT_")]
                pick = (wt if len(wt) > 0 else cand).iloc[0]
                orig_row = pick
                matched_id = pick["id"]
                status = f"uniprot_fallback(candidates={len(cand)})"

        seq = str(orig_row["sequence"]) if orig_row is not None else ""

        seqs.append(seq)
        matched_ids.append(matched_id)

        report.append({
            "row": i,
            "given_id": given_id,
            "uniprot_id": uid,
            "matched_id": matched_id,
            "status": status,
            "sequence_length": len(seq),
        })

    top_df["sequence"] = seqs

    if not has_id:
        top_df.insert(0, "id", matched_ids)

    report_df = pd.DataFrame(report)

    # 서열을 못 찾은 행은 제외. 교차 검증(MISMATCH) 실패 행도 원본이 달라졌을 수 있으므로 제외한다.
    keep = ~report_df["status"].str.contains("MISMATCH|NOT_FOUND").to_numpy()

    print(f"서열 매칭 성공(검증 통과): {int(keep.sum())} / {len(top_df)}")
    print(report_df["status"].str.replace(r"\(candidates=\d+\)", "", regex=True)
          .value_counts().to_string())

    bad = report_df[report_df["status"].str.contains("MISMATCH|NOT_FOUND")]

    if len(bad) > 0:
        print("[경고] 확인이 필요한 행:")
        print(bad[["given_id", "uniprot_id", "matched_id", "status"]].head(10).to_string(index=False))

    top_df = top_df[keep].reset_index(drop=True)

    return top_df, report_df


def safe_key(text):
    """npz 저장용 키 (경로 구분자 등 특수문자 제거)."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", str(text))


def run_step1_screening(smiles, module_dir=None):
    """
    Step 1(protein_Top100_classification.py)을 이 프로세스 안에서 직접 실행하고
    (Top100 결과 DataFrame, 저장된 CSV 경로)를 반환한다.
    Step 1 은 자기 설정(모델/벡터DB/출력 경로)을 그대로 사용한다.
    """

    module_dir = module_dir or MODEL_DIR

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        step1 = importlib.import_module(STEP1_MODULE)
    except ModuleNotFoundError as e:
        # Step 1 파일 자체가 없는 경우만 안내하고, 그 안의 다른 패키지 누락(h5py 등)은 원래 오류 그대로 전달
        if e.name == STEP1_MODULE:
            raise FileNotFoundError(
                f"Step 1 스크립트를 찾을 수 없습니다: {STEP1_MODULE}.py\n"
                f"  -> 이 파일과 같은 폴더({module_dir})에 두거나, "
                f"이미 뽑은 Top100 CSV 를 --top100-csv 로 지정하세요."
            ) from e
        raise

    print("\n" + "=" * 80)
    print("Step 1 : 인간 단백질 전체 스크리닝 -> Top100 (protein_Top100_classification.py)")
    print("=" * 80)

    started = time.time()

    step1_df = step1.run_screening(smiles)

    # Step 1 이 저장한 CSV 경로 (이번 실행에서 새로 만들어진 가장 최근 파일)
    saved_path = None
    step1_out_dir = getattr(step1, "OUTPUT_DIR", None)

    if step1_out_dir and os.path.isdir(step1_out_dir):
        fresh = [
            f for f in glob.glob(os.path.join(step1_out_dir, "top*_screening_*.csv"))
            if os.path.getmtime(f) >= started - 1
        ]
        if fresh:
            saved_path = max(fresh, key=os.path.getmtime)

    gc.collect()

    return step1_df, saved_path


def load_model(ckpt_path, device):

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt["model_config"].get("architecture") != ARCHITECTURE:
        raise RuntimeError(
            f"체크포인트의 모델 구조({ckpt['model_config'].get('architecture', '양방향(예전 구조)')})가 "
            f"현재 코드({ARCHITECTURE}: Compound->Protein 단방향)와 다릅니다. "
            f"새 step2_train.py 로 다시 학습한 체크포인트를 사용하세요."
        )

    model = CrossAttentionDTIRegressor(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 입력에 대한 gradient만 필요하므로 모델 파라미터 grad는 끈다
    for p in model.parameters():
        p.requires_grad_(False)

    return model, ckpt


# ============================================================
# 2. Token embedding 생성 (캐시 생성 스크립트와 동일한 방식)
# ============================================================

def encode_compound_tokens(smiles, device, max_tokens):

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
    mol_model.to(device).eval()

    enc = tokenizer([smiles], padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    # MoLFormer 내부의 선형 어텐션(MolformerFeatureMap)이 forward 마다 무작위 투영 특징을
    # 새로 뽑아서, eval() 을 걸어도 같은 입력에 매번 다른 벡터가 나온다(재현성 없음).
    # 시드를 고정해서 "무작위" 값 자체를 매번 똑같이 재현되게 만든다 (Step1 과 같은 시드).
    torch.manual_seed(42)

    with torch.no_grad():
        hidden = mol_model(**enc).last_hidden_state.cpu()

    valid = enc["attention_mask"].cpu().bool()

    tokens = hidden[0][valid[0]].float()[:max_tokens]

    del mol_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return tokens   # (Lc, 768)


def encode_protein_tokens(seq_dict, device, max_aa, max_tokens):

    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
    esm_model.to(device).eval()

    ids = sorted(seq_dict, key=lambda k: len(seq_dict[k]))

    tokens_by_id = {}

    for s in range(0, len(ids), ESM_BATCH_SIZE):

        batch_ids = ids[s:s + ESM_BATCH_SIZE]
        batch_seqs = [seq_dict[i][:max_aa] for i in batch_ids]

        enc = tokenizer(batch_seqs, padding=True, truncation=False, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            hidden = esm_model(**enc).last_hidden_state.cpu()

        mask = enc["attention_mask"].cpu().bool()

        for i, uid in enumerate(batch_ids):
            tokens_by_id[uid] = hidden[i][mask[i]].float()[:max_tokens]   # (L, 1280)

        print(f"  ESM-2: {min(s + ESM_BATCH_SIZE, len(ids))}/{len(ids)}", end="\r", flush=True)

    print()

    del esm_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return tokens_by_id


# ============================================================
# 3. Prediction
# ============================================================

def pad_tokens(token_list, device):

    B = len(token_list)
    Lp = max(t.shape[0] for t in token_list)
    dim = token_list[0].shape[1]

    padded = torch.zeros(B, Lp, dim, dtype=torch.float32)
    valid = torch.zeros(B, Lp, dtype=torch.bool)

    for i, t in enumerate(token_list):
        padded[i, :t.shape[0]] = t
        valid[i, :t.shape[0]] = True

    return padded.to(device), valid.to(device)


@torch.no_grad()
def predict_all(model, comp_tokens, prot_token_list, device, batch_size=16):

    comp = comp_tokens.to(device).unsqueeze(0)          # (1, Lc, 768)

    outputs = []

    for s in range(0, len(prot_token_list), batch_size):

        prot, prot_valid = pad_tokens(prot_token_list[s:s + batch_size], device)

        B = prot.shape[0]

        comp_b = comp.expand(B, -1, -1)
        comp_valid = torch.ones(B, comp.shape[1], dtype=torch.bool, device=device)

        preds = model(comp_b, comp_valid, prot, prot_valid)

        outputs.append(preds.float().cpu().numpy())

    return np.concatenate(outputs, axis=0)              # (N, n_tasks)


# ============================================================
# 4. Residue importance (Stage 3)
#    토큰 구성: [<cls>, residue_1 ... residue_n, <eos>]  ->  residue i 는 토큰 index i (1-based)
# ============================================================

def gradient_saliency(model, comp_d, comp_valid, prot_d, prot_valid, cols, n_res):
    """|d score / d protein_embedding * embedding| 의 L2 norm (residue별)."""

    prot_in = prot_d.detach().clone().float().requires_grad_(True)

    with torch.enable_grad():
        preds = model(comp_d, comp_valid, prot_in, prot_valid)
        score = preds[0, cols].mean()
        grad = torch.autograd.grad(score, prot_in)[0]

    saliency = (grad * prot_in.detach()).norm(dim=-1)[0]          # (Lp,)

    return saliency[1:1 + n_res].cpu().numpy()


@torch.no_grad()
def ablation_scores(model, comp_d, prot_d, cols, n_res, base_score, device):
    """
    sliding window로 residue를 가렸을 때(attention key + pooling에서 제외)
    예측 점수가 얼마나 떨어지는지. 값이 클수록 그 구간이 예측에 중요.
    """

    Lp = prot_d.shape[1]
    Lc = comp_d.shape[1]

    window = min(ABLATION_WINDOW, n_res)

    starts = list(range(0, n_res - window + 1, ABLATION_STRIDE))

    if starts[-1] + window < n_res:
        starts.append(n_res - window)

    delta_sum = np.zeros(n_res, dtype=np.float64)
    cover = np.zeros(n_res, dtype=np.float64)

    for s in range(0, len(starts), ABLATION_CHUNK):

        chunk_starts = starts[s:s + ABLATION_CHUNK]
        K = len(chunk_starts)

        valid = torch.ones(K, Lp, dtype=torch.bool, device=device)

        for k, st in enumerate(chunk_starts):
            valid[k, 1 + st:1 + st + window] = False

        comp_valid = torch.ones(K, Lc, dtype=torch.bool, device=device)

        preds = model(
            comp_d.expand(K, -1, -1), comp_valid,
            prot_d.expand(K, -1, -1), valid
        )

        scores = preds[:, cols].mean(dim=1).float().cpu().numpy()

        for k, st in enumerate(chunk_starts):
            delta_sum[st:st + window] += base_score - scores[k]
            cover[st:st + window] += 1.0

    return delta_sum / np.maximum(cover, 1.0)


def percentile_norm(x):
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x) / len(x)


def smooth(x, window):

    n = len(x)
    window = min(window, n)

    if window <= 1:
        return np.asarray(x, dtype=np.float64)

    kernel = np.ones(window)

    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones(n), kernel, mode="same")

    return num / den


def extract_sites(score, top_fraction, merge_gap, min_length, max_sites):
    """상위 top_fraction residue를 연속 구간으로 묶어 site 후보로 반환 (0-based, inclusive)."""

    n = len(score)

    threshold = np.quantile(score, 1.0 - top_fraction)
    idx = np.where(score >= threshold)[0]

    segments = []

    if len(idx) > 0:

        start = prev = int(idx[0])

        for i in idx[1:]:
            i = int(i)
            if i - prev - 1 <= merge_gap:
                prev = i
            else:
                segments.append((start, prev))
                start = prev = i

        segments.append((start, prev))

    segments = [(s, e) for s, e in segments if e - s + 1 >= min_length]

    segments.sort(key=lambda se: -float(score[se[0]:se[1] + 1].mean()))

    segments = segments[:max_sites]

    in_site = np.zeros(n, dtype=bool)

    for s, e in segments:
        in_site[s:e + 1] = True

    return segments, in_site


def safe_spearman(a, b):

    if a is None or b is None:
        return float("nan")

    a = np.asarray(a)
    b = np.asarray(b)

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(spearmanr(a, b)[0])


PEAK_METHODS = {
    "window_enrichment": "peak_enrichment",
    "window_mass": "peak_mass",
    "zscore": "peak_zscore",
    "entropy": "entropy_concentration",
}


@torch.no_grad()
def attention_profile(model, comp_tokens, prot_tokens, cols, device):
    """
    화합물 토큰 -> 단백질 residue cross-attention 을 residue 축 분포(합=1)로 만든다.
    (층 평균 -> 화합물 토큰 평균 -> <cls>/<eos> 제외 -> 재정규화)
    반환: (예측 점수, residue별 attention (n_res,))
    """

    n_res = prot_tokens.shape[0] - 2

    comp_d = comp_tokens.to(device).unsqueeze(0)
    prot_d = prot_tokens.to(device).unsqueeze(0)

    comp_valid = torch.ones(1, comp_d.shape[1], dtype=torch.bool, device=device)
    prot_valid = torch.ones(1, prot_d.shape[1], dtype=torch.bool, device=device)

    preds, info = model(comp_d, comp_valid, prot_d, prot_valid, return_attn=True)

    score = float(preds[0, cols].mean().item())

    attn = torch.stack(info["attn_c2p"], dim=0).mean(dim=0)[0]          # (Lc, Lp)
    per_residue = attn.mean(dim=0)[1:1 + n_res]                           # (n_res,)
    per_residue = per_residue / per_residue.sum().clamp(min=1e-12)

    return score, per_residue.float().cpu().numpy()


def attention_peak_metrics(attention, window):
    """
    residue별 attention 분포가 한곳에 얼마나 뚜렷하게 몰려 있는지 여러 지표로 계산.
    (window 는 서열보다 길면 서열 길이로 줄어들고, 그때 enrichment 는 1.0)
    """

    a = np.asarray(attention, dtype=np.float64)
    n = len(a)

    total = a.sum()
    a = a / total if total > 0 else np.full(n, 1.0 / n)

    w = max(1, min(int(window), n))

    csum = np.concatenate([[0.0], np.cumsum(a)])
    window_mass = csum[w:] - csum[:-w]                     # 각 시작 위치의 창 안 attention 질량

    best = int(np.argmax(window_mass))
    peak_mass = float(window_mass[best])
    enrichment = peak_mass / (w / n)                       # 균등 분포 기대값 대비 몇 배

    std = float(a.std())
    zscore = float((a.max() - a.mean()) / std) if std > 1e-12 else 0.0     # 완전 균등이면 0

    nonzero = a[a > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    entropy_concentration = 1.0 - entropy / np.log(n) if n > 1 else 0.0

    return {
        "peak_start": best + 1,                            # 1-based
        "peak_end": best + w,
        "peak_window": w,
        "peak_mass": peak_mass,
        "peak_enrichment": float(enrichment),
        "peak_zscore": zscore,
        "entropy_concentration": float(entropy_concentration),
        "max_residue": int(np.argmax(a)) + 1,
        "max_weight": float(a.max()),
    }


def select_final_proteins(peak_df, final_n, method):
    """
    Peak Score(선택 기준 지표)가 큰 순으로 정렬해 상위 final_n 개를 최종 선택.
    점수가 같으면 IC50 재랭킹 순위(stage2_rank)가 높은 쪽을 먼저.
    반환: (peak_rank 가 붙은 전체 표, 최종 선택된 표)
    """

    if method not in PEAK_METHODS:
        raise ValueError(f"알 수 없는 peak method: {method} (가능: {list(PEAK_METHODS)})")

    column = PEAK_METHODS[method]

    ranked = peak_df.copy()
    ranked["attention_peak_score"] = ranked[column]

    ranked = ranked.sort_values(
        ["attention_peak_score", "stage2_rank"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    ranked.insert(0, "peak_rank", np.arange(1, len(ranked) + 1))

    return ranked, ranked.head(final_n).copy()


def analyze_protein(model, comp_tokens, prot_tokens, seq_used, cols, device):

    n_res = prot_tokens.shape[0] - 2

    if n_res != len(seq_used):
        raise RuntimeError(
            f"토큰 수({prot_tokens.shape[0]})와 서열 길이({len(seq_used)})가 맞지 않습니다."
        )

    comp_d = comp_tokens.to(device).unsqueeze(0)
    prot_d = prot_tokens.to(device).unsqueeze(0)

    comp_valid = torch.ones(1, comp_d.shape[1], dtype=torch.bool, device=device)
    prot_valid = torch.ones(1, prot_d.shape[1], dtype=torch.bool, device=device)

    # ---- base prediction + attention -------------------------
    with torch.no_grad():
        preds, info = model(comp_d, comp_valid, prot_d, prot_valid, return_attn=True)

    base_score = float(preds[0, cols].mean().item())

    # compound->protein attention: 층 평균 -> 화합물 토큰 평균 -> residue 구간만 사용
    attn_layers = torch.stack(info["attn_c2p"], dim=0).mean(dim=0)[0]   # (Lc, Lp)
    attention = attn_layers.mean(dim=0)[1:1 + n_res]
    attention = (attention / attention.sum().clamp(min=1e-12)).cpu().numpy()   # attention_profile 과 동일

    # ---- gradient, ablation ----------------------------------
    gradient = gradient_saliency(
        model, comp_d, comp_valid, prot_d, prot_valid, cols, n_res
    )

    ablation = ablation_scores(model, comp_d, prot_d, cols, n_res, base_score, device)

    # ---- 결합 ---------------------------------------------------
    raw = {
        "attention": attention,
        "gradient": gradient,
        "ablation": np.clip(ablation, 0.0, None)     # 음수(가려도 오히려 상승)는 중요도 0으로
    }

    weighted = []

    for name, values in raw.items():
        w = SITE_WEIGHTS.get(name, 0.0)
        if values is not None and w > 0:
            weighted.append((w, percentile_norm(values)))

    combined = sum(w * v for w, v in weighted) / sum(w for w, _ in weighted)

    smoothed = smooth(combined, SMOOTH_WINDOW)

    segments, in_site = extract_sites(
        smoothed, SITE_TOP_FRACTION, SITE_MERGE_GAP,
        SITE_MIN_LENGTH, MAX_SITES_PER_PROTEIN
    )

    # ---- 방법 간 일치도 (attention이 실제 예측 기여와 맞는지 진단) -------
    agreement = {
        "spearman_attention_vs_ablation": safe_spearman(attention, ablation),
        "spearman_gradient_vs_ablation": safe_spearman(gradient, ablation),
        "spearman_attention_vs_gradient": safe_spearman(attention, gradient)
    }

    # ---- site embedding (Stage 4 유사도용) ---------------------
    if in_site.any():
        residue_tokens = prot_tokens[1:1 + n_res].numpy()
        site_embedding = residue_tokens[in_site].mean(axis=0)
    else:
        site_embedding = None

    return {
        "base_score": base_score,
        "attention": attention,
        "gradient": gradient,
        "ablation": ablation,
        "combined": combined,
        "smoothed": smoothed,
        "in_site": in_site,
        "segments": segments,
        "agreement": agreement,
        "site_embedding": site_embedding
    }


# ============================================================
# 5. Main
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Step1(Top100 스크리닝) 자동 실행 -> 서열 매칭 -> IC50 재랭킹(Top N) -> Attention Peak 최종 선택"
    )

    p.add_argument("--smiles", type=str, default=None)
    p.add_argument("--top100-csv", type=str, default=None,
                   help="이미 뽑아둔 Step1 결과 CSV (top100_screening_*.csv). "
                        "지정하면 Step 1 을 건너뛰고 이 파일을 사용. 생략하면 Step 1 을 자동 실행")
    p.add_argument("--original-csv", type=str, default=ORIGINAL_CSV_PATH,
                   help="서열을 가져올 원본 CSV (esm2_ready_dataset_cropped_final.csv)")
    p.add_argument("--ckpt", type=str, default=CHECKPOINT_PATH)
    p.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--top-n", type=int, default=TOP_N,
                   help="IC50 예측값 기준 재랭킹 후 남길 개수 (기본 20)")
    p.add_argument("--final-n", type=int, default=FINAL_N,
                   help="Attention Peak Score 로 최종 선택할 개수 (기본 5)")
    p.add_argument("--peak-window", type=int, default=PEAK_WINDOW,
                   help="Peak Score 에서 가중치가 몰렸는지 볼 연속 residue 창 크기")
    p.add_argument("--peak-method", type=str, default=PEAK_METHOD, choices=list(PEAK_METHODS),
                   help="최종 선택 기준 지표")
    p.add_argument("--detail-scope", type=str, default=DETAIL_SCOPE, choices=["final", "top"],
                   help="상세 분석(gradient/ablation) 대상: final=최종 선택만, top=Top N 전부")
    p.add_argument("--rank-tasks", nargs="+", default=None,
                   help="rank_score 에 쓸 task (기본: 체크포인트의 모든 task). 예: ki kd")

    return p.parse_args()


def main():

    args = parse_args()

    smiles = args.smiles or input("SMILES를 입력하세요: ").strip()

    if not smiles:
        raise ValueError("SMILES가 입력되지 않았습니다.")

    total_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CPU 멀티코어 활용 (Step 1 을 import 하면 Step 1 쪽에서도 같은 값으로 설정됨)
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 80)
    print("Step 1 + Step 2 pipeline (screening -> sequence match -> rerank -> site analysis)")
    print("=" * 80)
    print(f"Device : {device} (threads {torch.get_num_threads()})")
    print(f"SMILES : {smiles}")

    # ---------------- Step 1: Top100 (자동 실행 또는 기존 CSV 재사용) ----
    if args.top100_csv:

        print(f"\nStep 1 건너뜀 - 기존 결과 사용: {args.top100_csv}")

        step1_csv = args.top100_csv

        top_df = pd.read_csv(step1_csv, dtype={"id": str, "uniprot_id": str},
                             keep_default_na=False, encoding="utf-8-sig")

    else:

        top_df, step1_csv = run_step1_screening(smiles)

        top_df = top_df.copy()

        for col in ("id", "uniprot_id"):
            if col in top_df.columns:
                top_df[col] = top_df[col].astype(str)

        print(f"\nStep 1 완료 (누적 {time.time() - total_start:.0f}초)"
              + (f" | 결과 파일: {step1_csv}" if step1_csv else ""))

    # ---------------- Top100 목록 + 원본 CSV 서열 매칭 ----------
    if "uniprot_id" not in top_df.columns:
        raise RuntimeError(f"Top100 결과에 uniprot_id 컬럼이 없습니다. 컬럼: {list(top_df.columns)}")

    if len(top_df) == 0:
        raise RuntimeError(
            "Step 1 결과가 비어 있습니다 (Ki/Kd 확률 임계값을 통과한 단백질이 없었을 수 있음). "
            "Step 2 를 진행할 후보가 없습니다."
        )

    # Step1 의 rank 는 stage2 순위와 헷갈리지 않게 이름을 바꿔둔다
    if "rank" in top_df.columns:
        top_df = top_df.rename(columns={"rank": "stage1_rank"})

    print(f"\n후보 단백질(Step1 결과) : {len(top_df)}개")

    orig_df = load_original_sequences(args.original_csv)

    top_df, match_report = match_sequences(top_df, orig_df)

    del orig_df

    if len(top_df) == 0:
        raise RuntimeError("서열을 확보한 단백질이 없습니다. --original-csv 경로와 id 를 확인하세요.")

    # id 는 고유 키여야 함 (같은 id 가 두 번 들어오면 토큰/결과가 섞임)
    top_df = top_df.drop_duplicates("id").reset_index(drop=True)

    # ---------------- 모델 ----------------------------------
    model, ckpt = load_model(args.ckpt, device)

    task_names = ckpt["task_names"]

    rank_tasks = args.rank_tasks or task_names

    for t in rank_tasks:
        if t not in task_names:
            raise ValueError(f"rank task '{t}' 는 체크포인트 task {task_names} 에 없습니다.")

    cols = [task_names.index(t) for t in rank_tasks]

    max_comp_tokens = ckpt["max_compound_tokens"]
    max_prot_tokens = ckpt["max_protein_tokens"]
    max_aa = max_prot_tokens - 2

    print(f"모델 task : {task_names} | rank_score = mean({rank_tasks})")
    print(f"체크포인트 best epoch : {ckpt.get('best_epoch')}")

    n_long = int((top_df["sequence"].str.len() > max_aa).sum())

    if n_long > 0:
        print(f"[경고] {max_aa} aa 보다 긴 서열 {n_long}개 -> 앞 {max_aa} aa 만 사용합니다.")

    # ---------------- 토큰 임베딩 ---------------------------
    print("\n[1/5] MoLFormer 토큰 임베딩")

    comp_tokens = encode_compound_tokens(smiles, device, max_comp_tokens)

    print(f"  compound tokens : {tuple(comp_tokens.shape)}")

    print("\n[2/5] ESM-2 토큰 임베딩")

    # 키는 uniprot_id 가 아니라 id (wildtype / mutant / window 가 서로 다른 서열이므로)
    seq_dict = dict(zip(top_df["id"], top_df["sequence"]))

    esm_start = time.time()

    prot_tokens = encode_protein_tokens(seq_dict, device, max_aa, max_prot_tokens)

    print(f"  ESM-2 완료 ({time.time() - esm_start:.0f}초)")

    # ---------------- Stage 2: IC50 예측 + Top N 재랭킹 ----------
    print(f"\n[3/5] {'/'.join(t.upper() for t in rank_tasks)} 예측 + Top {args.top_n} 재랭킹")

    ids = list(top_df["id"])

    preds = predict_all(model, comp_tokens, [prot_tokens[i] for i in ids], device)

    result = top_df.drop(columns=["sequence"]).copy()
    result["sequence_length"] = [min(len(seq_dict[i]), max_aa) for i in ids]

    for j, t in enumerate(task_names):
        result[f"pred_{t}"] = preds[:, j]

    # 예측 paffinity 가 클수록 결합이 강함 -> 높은 순으로 재랭킹
    result["rank_score"] = preds[:, cols].mean(axis=1)

    result = result.sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)
    result.insert(0, "stage2_rank", np.arange(1, len(result) + 1))

    top_n = min(args.top_n, len(result))
    final_n = min(args.final_n, top_n)

    result["in_top_n"] = result["stage2_rank"] <= top_n

    print(result.head(top_n)[
        ["stage2_rank", "id", "uniprot_id"] + [f"pred_{t}" for t in task_names]
    ].to_string(index=False))

    # ---------------- Stage 3a: Top N Attention Peak Score --------
    print(f"\n[4/5] Top {top_n} Attention Peak Score 계산 "
          f"(창 {args.peak_window} residue, 선택 기준 {args.peak_method})")

    peak_rows = []
    profile_frames = []
    att_by_id = {}

    for rank_idx in range(top_n):

        pid = result.loc[rank_idx, "id"]
        uid = result.loc[rank_idx, "uniprot_id"]

        seq_used = seq_dict[pid][:max_aa]

        _, att = attention_profile(model, comp_tokens, prot_tokens[pid], cols, device)

        att_by_id[pid] = att

        metrics = attention_peak_metrics(att, args.peak_window)

        peak_rows.append({
            "id": pid,
            "uniprot_id": uid,
            "stage2_rank": rank_idx + 1,
            "rank_score": float(result.loc[rank_idx, "rank_score"]),
            "sequence_length": len(seq_used),
            **metrics,
            "peak_sequence": seq_used[metrics["peak_start"] - 1:metrics["peak_end"]],
        })

        profile_frames.append(pd.DataFrame({
            "id": pid,
            "uniprot_id": uid,
            "stage2_rank": rank_idx + 1,
            "residue_index": np.arange(1, len(seq_used) + 1),
            "aa": list(seq_used),
            "attention": att,
        }))

    peak_df = pd.DataFrame(peak_rows)

    # ---------------- Stage 3b: attention peak score 만으로 최종 5개 선택 -------
    ranked_peak, final_df = select_final_proteins(peak_df, final_n, args.peak_method)
    final_ids = list(final_df["id"])

    print(f"\n[Attention Peak Score 순위 - 선택 기준: {args.peak_method}]")
    show_cols = ["peak_rank", "id", "uniprot_id", "stage2_rank", "rank_score",
                 "attention_peak_score", "peak_mass", "peak_start", "peak_end"]
    table = ranked_peak[show_cols].copy()
    table.insert(0, "final", ["*" if i in final_ids else "" for i in table["id"]])

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n최종 선택 {len(final_df)}개: " + ", ".join(final_ids))

    # 후보 전체 표에 Peak Score / 최종 선택 여부를 붙인다
    result = result.merge(
        ranked_peak[["id", "peak_rank", "attention_peak_score", "peak_mass", "peak_zscore",
                     "entropy_concentration", "peak_start", "peak_end", "peak_sequence"]],
        on="id", how="left"
    )

    result["in_final"] = result["id"].isin(final_ids)

    # ---------------- Stage 3c: 상세 residue 분석 ------------------
    detail_ids = final_ids if args.detail_scope == "final" else list(ranked_peak["id"])

    print(f"\n[5/5] 상세 residue 분석 ({args.detail_scope}: {len(detail_ids)}개 - "
          f"attention + gradient + ablation)")

    by_id = result.set_index("id")

    residue_frames = []
    site_rows = []
    site_embeddings = {}

    for pid in detail_ids:

        uid = by_id.loc[pid, "uniprot_id"]
        stage2_rank = int(by_id.loc[pid, "stage2_rank"])
        peak_rank = int(ranked_peak.loc[ranked_peak["id"] == pid, "peak_rank"].iloc[0])

        seq_used = seq_dict[pid][:max_aa]

        out = analyze_protein(model, comp_tokens, prot_tokens[pid], seq_used, cols, device)

        n_res = len(seq_used)

        residue_frames.append(pd.DataFrame({
            "id": pid,
            "uniprot_id": uid,
            "stage2_rank": stage2_rank,
            "peak_rank": peak_rank,
            "residue_index": np.arange(1, n_res + 1),
            "aa": list(seq_used),
            "attention": out["attention"],
            "gradient": out["gradient"],
            "ablation_delta": out["ablation"],
            "combined": out["combined"],
            "smoothed": out["smoothed"],
            "in_site": out["in_site"]
        }))

        for site_rank, (s0, e0) in enumerate(out["segments"], start=1):
            site_rows.append({
                "id": pid,
                "uniprot_id": uid,
                "stage2_rank": stage2_rank,
                "peak_rank": peak_rank,
                "site_rank": site_rank,
                "start": s0 + 1,
                "end": e0 + 1,
                "length": e0 - s0 + 1,
                "mean_score": float(out["smoothed"][s0:e0 + 1].mean()),
                "sequence": seq_used[s0:e0 + 1],
                **out["agreement"]
            })

        if out["site_embedding"] is not None:
            site_embeddings[safe_key(pid)] = out["site_embedding"].astype(np.float32)

        sites_str = ", ".join(f"{s0 + 1}-{e0 + 1}" for s0, e0 in out["segments"]) or "-"

        print(f"  peak#{peak_rank:>2} {pid:<26} ({uid}) 예측 {out['base_score']:.3f} | "
              f"sites: {sites_str} | "
              f"attn~ablation rho={out['agreement']['spearman_attention_vs_ablation']:.2f}")

    # ---------------- 저장 -----------------------------------
    run_dir = os.path.join(args.out_dir, "stage2_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    result.to_csv(os.path.join(run_dir, "rerank_top100.csv"), index=False, encoding="utf-8-sig")
    match_report.to_csv(os.path.join(run_dir, "sequence_match_report.csv"),
                        index=False, encoding="utf-8-sig")

    ranked_peak.to_csv(os.path.join(run_dir, f"top{top_n}_attention_peak_scores.csv"),
                       index=False, encoding="utf-8-sig")

    pd.concat(profile_frames, ignore_index=True).to_csv(
        os.path.join(run_dir, f"top{top_n}_attention_profile.csv"), index=False
    )

    result[result["in_final"]].sort_values("peak_rank").to_csv(
        os.path.join(run_dir, f"final{final_n}_proteins.csv"), index=False, encoding="utf-8-sig"
    )

    pd.concat(residue_frames, ignore_index=True).to_csv(
        os.path.join(run_dir, "detail_residue_scores.csv"), index=False
    )
    pd.DataFrame(site_rows).to_csv(
        os.path.join(run_dir, "detail_sites.csv"), index=False, encoding="utf-8-sig"
    )

    if site_embeddings:
        # 키 = 특수문자를 정리한 id (detail_sites.csv 의 id 와 safe_key 로 대응)
        np.savez(os.path.join(run_dir, "detail_site_embeddings.npz"), **site_embeddings)

    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump({
            "smiles": smiles,
            "step1_result_csv": step1_csv,
            "step1_skipped": bool(args.top100_csv),
            "original_csv": args.original_csv,
            "checkpoint": args.ckpt,
            "architecture": ARCHITECTURE,
            "task_names": task_names,
            "rank_tasks": rank_tasks,
            "top_n": top_n,
            "final_n": final_n,
            "peak_window": args.peak_window,
            "peak_method": args.peak_method,
            "detail_scope": args.detail_scope,
            "final_ids": final_ids,
            "ablation_window": ABLATION_WINDOW,
            "ablation_stride": ABLATION_STRIDE,
            "site_weights": SITE_WEIGHTS,
            "smooth_window": SMOOTH_WINDOW,
            "site_top_fraction": SITE_TOP_FRACTION,
            "site_merge_gap": SITE_MERGE_GAP,
            "site_min_length": SITE_MIN_LENGTH
        }, f, indent=2, ensure_ascii=False)

    print(f"\n저장 완료: {run_dir}")
    print(f"총 소요 시간: {(time.time() - total_start) / 60:.1f}분")


if __name__ == "__main__":
    main()