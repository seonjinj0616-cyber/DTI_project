import os
import pandas as pd

# ============================================================
# 1. 경로 및 파일 설정
# ============================================================
DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\final_final"

FILES = [
    "IC50_ems2_fi.csv",
    "Ki_ems2_fi.csv",
    "Kd_ems2_fi.csv"
]

EXPECTED_COLUMNS = [
    "smiles",
    "uniprot_id",
    "target_name",
    "affinity_type",
    "affinity_value_mean",
    "paffinity",
    "protein_name",
    "protein_sequence",
    "sequence_length"
]

print("=" * 80)
print("STARTING FINAL DATASET QUALITY CONTROL (QC) PIPELINE")
print("=" * 80)

qc_summary_list = []

# ============================================================
# 2. 파일별 QC 검사 수행
# ============================================================
for filename in FILES:
    file_path = os.path.join(DATA_DIR, filename)

    print("\n" + "-" * 80)
    print(f"[QC Target File: {filename}]")
    print("-" * 80)

    if not os.path.exists(file_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
        continue

    # 데이터 로드
    df = pd.read_csv(file_path, low_memory=False)
    total_rows = len(df)
    print(f"1. 전체 데이터 행 수 : {total_rows:,}개")

    # --------------------------------------------------------
    # QC 1: 컬럼 구성을 확인
    # --------------------------------------------------------
    col_list = df.columns.tolist()
    is_col_exact = (col_list == EXPECTED_COLUMNS)
    print(f"2. 컬럼 구조 및 순서 일치 : {'PASS' if is_col_exact else 'FAIL'}")
    if not is_col_exact:
        print(f"   - 실제 컬럼: {col_list}")

    # --------------------------------------------------------
    # QC 2: 전체 결측치(Null / NaN / 공백 / 문자열 'nan') 검사
    # --------------------------------------------------------
    print("\n[컬럼별 결측치 현황]")
    null_counts = {}
    has_any_null = False

    for col in df.columns:
        # 실제 pandas null + 공백/문자열 nan 포함 검사
        str_col = df[col].fillna("").astype(str).str.strip().str.lower()
        missing_mask = (df[col].isna()) | (str_col == "") | (str_col.isin(["nan", "none", "null"]))
        cnt = missing_mask.sum()
        null_counts[col] = cnt
        status = "OK" if cnt == 0 else f"WARNING ({cnt:,}개 결측)"
        print(f"  - {col:20s}: {cnt:>8,}개 | {status}")
        if cnt > 0:
            has_any_null = True

    # --------------------------------------------------------
    # QC 3: 단백질 서열(protein_sequence) 길이 검사
    # --------------------------------------------------------
    seq_series = df["protein_sequence"].fillna("").astype(str).str.strip()
    actual_lens = seq_series.str.len()

    max_len = actual_lens.max()
    min_len = actual_lens.min()
    over_1024_cnt = (actual_lens > 1024).sum()

    # sequence_length 컬럼과의 일치 여부 검사
    length_mismatch_cnt = (df["sequence_length"] != actual_lens).sum()

    print("\n[단백질 서열(protein_sequence) 검증]")
    print(f"  - 최소 서열 길이     : {min_len} aa")
    print(f"  - 최대 서열 길이     : {max_len} aa (목표: <= 1024 aa)")
    print(f"  - 1024aa 초과 개수    : {over_1024_cnt:,}개")
    print(f"  - 길이 불일치(Sync)   : {length_mismatch_cnt:,}개")

    # --------------------------------------------------------
    # QC 4: 종합 판정 (Verdict)
    # --------------------------------------------------------
    seq_pass = (max_len <= 1024) and (over_1024_cnt == 0) and (length_mismatch_cnt == 0)
    null_pass = not has_any_null

    if seq_pass and null_pass and is_col_exact:
        overall_status = "ALL PASS"
    else:
        overall_status = "FAIL / CHECK NEEDED"

    print(f"\n>>> [{filename}] QC 결과: {overall_status}")

    qc_summary_list.append({
        "Filename": filename,
        "Total Rows": f"{total_rows:,}",
        "Max Seq Len": max_len,
        ">1024aa Count": over_1024_cnt,
        "Total Nulls": sum(null_counts.values()),
        "Column Match": is_col_exact,
        "QC Verdict": overall_status
    })

# ============================================================
# 3. 최종 QC 요약 리포트
# ============================================================
print("\n\n" + "=" * 80)
print("FINAL QUALITY CONTROL SUMMARY REPORT")
print("=" * 80)

summary_df = pd.DataFrame(qc_summary_list)
print(summary_df.to_string(index=False))
print("=" * 80)