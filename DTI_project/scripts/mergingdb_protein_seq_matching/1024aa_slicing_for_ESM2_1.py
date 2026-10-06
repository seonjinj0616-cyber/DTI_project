import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

# ============================================================
# 1. 경로 설정 및 파라미터 정의
# ============================================================
target_dir = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"
ref_dir = r"C:\workspace\python\project_personal\DTI_project\data\esm2_human_whole_protein_vecterDB\homo_protein_seq_fasta"

files_to_process = {
    "IC50": "IC50_esm2_final_qc.csv",
    "Ki": "Ki_esm2_final_qc.csv",
    "Kd": "Kd_esm2_final_qc.csv"
}

TARGET_LENGTH = 1024

# ============================================================
# 2. Reference 데이터 로드 (첫 번째 UniProt ID 기준 중복 제거)
# ============================================================
ref_path = os.path.join(ref_dir, "esm2_ready_dataset_cropped_final.csv")
print(f"Loading reference dataset: {ref_path}")
ref_df = pd.read_csv(ref_path, dtype=str, low_memory=False)

ref_df_first = ref_df.drop_duplicates(subset=['uniprot_id'], keep='first')
ref_dict = dict(zip(ref_df_first['uniprot_id'], ref_df_first['sequence'].fillna('')))
print(f"[*] Reference Unique UniProt IDs (First Occurrences): {len(ref_dict):,}개")

# API 요청 결과 캐싱 딕셔너리
active_site_cache = {}


def clean_sequence(seq):
    """비표준 아미노산 및 공백 정제"""
    if pd.isna(seq):
        return ""
    seq = str(seq).strip().upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)


def find_reference_start_position(orig_seq, ref_seq):
    """
    레퍼런스 서열을 통해 원본 서열 내 시작 인덱스(0-based)를 추정.
    ★ 절대로 레퍼런스 서열을 복사하지 않고 위치(idx)만 반환함.
    """
    orig_seq = clean_sequence(orig_seq)
    ref_seq = clean_sequence(ref_seq)

    if not orig_seq or not ref_seq:
        return None

    # 1단계: 시드(Seed) 기반 검색 (다양한 길이 및 오프셋 시도)
    for ref_offset in [0, 20, 50]:
        if len(ref_seq) < ref_offset + 20:
            continue
        for seed_len in [100, 80, 60, 50, 40, 30, 20]:
            if len(ref_seq) >= ref_offset + seed_len:
                seed = ref_seq[ref_offset: ref_offset + seed_len]
                pos = orig_seq.find(seed)
                if pos != -1:
                    return max(0, pos - ref_offset)

    # 2단계: 슬라이딩 윈도우 유사도 기반 탐색 (유사도 65% 이상일 때 위치 인정)
    window_size = min(60, len(ref_seq), len(orig_seq))
    if window_size < 20:
        return 0

    ref_window = ref_seq[:window_size]
    best_pos = None
    best_score = 0.0

    step = 5 if len(orig_seq) > 2000 else 2
    for start in range(0, max(1, len(orig_seq) - window_size + 1), step):
        candidate = orig_seq[start: start + window_size]
        score = SequenceMatcher(None, ref_window, candidate).ratio()
        if score > best_score:
            best_score = score
            best_pos = start

    if best_score >= 0.65:
        return best_pos

    return None


def get_active_site_position(uniprot_id):
    """UniProt REST API 호출하여 Active/Binding Site 위치(1-based) 반환"""
    if uniprot_id in active_site_cache:
        return active_site_cache[uniprot_id]

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for feat in data.get('features', []):
                if feat.get('type') in ['Active site', 'Binding site']:
                    start_pos = feat.get('location', {}).get('start', {}).get('value')
                    if start_pos:
                        active_site_cache[uniprot_id] = start_pos
                        return start_pos
    except Exception:
        pass

    active_site_cache[uniprot_id] = None
    return None


def process_custom_slicing(uniprot_id, raw_seq):
    """
    3단계 조건 기반 슬라이싱 함수
    반환값: (절단된 서열, 적용 규칙, 시작 인덱스, 종료 인덱스, 원본 서열 길이)
    """
    orig_seq = clean_sequence(raw_seq)
    seq_len = len(orig_seq)

    # [기본 조건] 1024aa 이하인 경우 그대로 유지
    if seq_len <= TARGET_LENGTH:
        return orig_seq, "ORIGINAL_LE_1024", 0, seq_len, seq_len

    # [조건 1] Reference 기반 위치 탐색 후 '원본 서열' 슬라이싱
    if uniprot_id in ref_dict:
        ref_seq = ref_dict[uniprot_id]
        start_idx = find_reference_start_position(orig_seq, ref_seq)
        if start_idx is not None:
            end_idx = start_idx + TARGET_LENGTH
            if end_idx > seq_len:
                end_idx = seq_len
                start_idx = max(0, seq_len - TARGET_LENGTH)
            return orig_seq[start_idx:end_idx], "RULE1_REF_POSITION_CROP", start_idx, end_idx, seq_len

    # [조건 2] Active Site 위치 중심 '원본 서열' 슬라이싱
    active_pos = get_active_site_position(uniprot_id)
    if active_pos is not None:
        center_idx = active_pos - 1
        start_idx = center_idx - (TARGET_LENGTH // 2)
        end_idx = center_idx + (TARGET_LENGTH // 2)

        if start_idx < 0:
            start_idx = 0
            end_idx = min(TARGET_LENGTH, seq_len)
        elif end_idx > seq_len:
            end_idx = seq_len
            start_idx = max(0, seq_len - TARGET_LENGTH)

        return orig_seq[start_idx:end_idx], "RULE2_ACTIVE_SITE_CROP", start_idx, end_idx, seq_len

    # [조건 3] N-터미널 기준 '원본 서열' 앞 1024aa 슬라이싱
    return orig_seq[:TARGET_LENGTH], "RULE3_HEAD_TRUNCATION", 0, min(TARGET_LENGTH, seq_len), seq_len


# ============================================================
# 3. 데이터셋 처리 및 상세 규칙 리스트 CSV 생성
# ============================================================
print("\n" + "=" * 60)
print("STARTING STRICT ORIGINAL SEQUENCE SLICING PIPELINE")
print("=" * 60)

for name, fname in files_to_process.items():
    fpath = os.path.join(target_dir, fname)
    if not os.path.exists(fpath):
        print(f"[Skip] File not found: {fpath}")
        continue

    df = pd.read_csv(fpath, low_memory=False)
    print(f"\nProcessing [{name}] ({len(df):,} rows)...")

    id_col = 'uniprot_id' if 'uniprot_id' in df.columns else 'Target_ID'
    seq_col = 'protein_sequence' if 'protein_sequence' in df.columns else 'sequence'

    sliced_sequences = []
    applied_rules = []
    start_indices = []
    end_indices = []
    orig_lengths = []

    # 전체 데이터 슬라이싱 실행
    for u_id, s_val in tqdm(zip(df[id_col], df[seq_col]), total=len(df), desc=f"Slicing {name}"):
        final_seq, rule, start_i, end_i, orig_len = process_custom_slicing(str(u_id).strip(), s_val)
        sliced_sequences.append(final_seq)
        applied_rules.append(rule)
        start_indices.append(start_i)
        end_indices.append(end_i)
        orig_lengths.append(orig_len)

    # 결과 업데이트
    df[seq_col] = sliced_sequences
    df['slicing_rule'] = applied_rules

    # QC 통계 출력
    lengths = df[seq_col].astype(str).apply(len)
    print(f"\n  [QC Statistics for {name}]")
    print(f"  - Max Sequence Length : {lengths.max()} aa")
    print(f"  - Sequences >1024 aa  : {(lengths > 1024).sum()}개")
    print(f"  - Rule Distribution:")
    for rule_name, count in df['slicing_rule'].value_counts().items():
        print(f"      * {rule_name}: {count:,}개")

    # 메인 정제 파일 저장
    df.to_csv(fpath, index=False, encoding='utf-8-sig')
    print(f"  --> Saved main cleaned dataset to: {fname}")

    # ------------------------------------------------------------
    # 4. 규칙별 작업 리스트 상세 CSV 생성 (요청 사항)
    # ------------------------------------------------------------
    qc_details_df = pd.DataFrame({
        'uniprot_id': df[id_col],
        'original_length': orig_lengths,
        'sliced_length': lengths,
        'slicing_rule': applied_rules,
        'crop_start_idx': start_indices,
        'crop_end_idx': end_indices,
        'sliced_sequence': sliced_sequences
    })

    # 1) 전체 규칙 통합 상세 로그 파일 저장
    detail_all_path = os.path.join(target_dir, f"{name}_slicing_details_all.csv")
    qc_details_df.to_csv(detail_all_path, index=False, encoding='utf-8-sig')

    # 2) 각 규칙별(RULE1, RULE2, RULE3) 개별 CSV 파일 각각 생성
    for rule_type in ["RULE1_REF_POSITION_CROP", "RULE2_ACTIVE_SITE_CROP", "RULE3_HEAD_TRUNCATION"]:
        rule_subset = qc_details_df[qc_details_df['slicing_rule'] == rule_type]
        rule_out_path = os.path.join(target_dir, f"{name}_list_{rule_type}.csv")
        rule_subset.to_csv(rule_out_path, index=False, encoding='utf-8-sig')
        print(f"  [Rule List Exported] {rule_type}: {len(rule_subset):,}건 -> {os.path.basename(rule_out_path)}")

print("\n" + "=" * 60)
print("ALL PIPELINES AND RULE LIST CSV GENERATIONS COMPLETED!")
print("=" * 60)