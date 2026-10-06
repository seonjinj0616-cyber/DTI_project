import os
import pandas as pd

# ============================================================
# 1. 경로 및 파일명 매핑 설정
# ============================================================
# 원본 업데이트 파일이 저장된 실제 경로
INPUT_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"
OUTPUT_DIR = os.path.join(INPUT_DIR, "final_final")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# [입력 파일명 : 출력 파일명] 개별 매핑
FILE_MAPPING = {
    "IC50_esm2_final_updated.csv": "IC50_ems2_fi.csv",
    "Ki_esm2_final_updated.csv": "Ki_ems2_fi.csv",
    "Kd_esm2_final_updated.csv": "Kd_ems2_fi.csv"
}

# ============================================================
# 2. 삭제 대상 컬럼 목록
# ============================================================
DROP_COLUMNS = [
    "esm2_sequence",
    "esm2_sequence_source",
    "esm2_sequence_length",
    "slicing_rule",
    "crop_start_idx",
    "crop_end_idx",
    "original_length",
    "sliced_length"
]

# ============================================================
# 3. 최종 요구 컬럼 규격 및 순서
# ============================================================
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

summary_reports = []

# ============================================================
# 4. 파일별 파이프라인 처리
# ============================================================
for in_filename, out_filename in FILE_MAPPING.items():

    input_path = os.path.join(INPUT_DIR, in_filename)
    output_path = os.path.join(OUTPUT_DIR, out_filename)

    print("\n" + "=" * 80)
    print(f"INPUT FILE  : {in_filename}")
    print(f"OUTPUT FILE : {out_filename}")
    print("=" * 80)

    # --------------------------------------------------------
    # 파일 존재 검사 및 CSV 읽기
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        print(f"[경고] 파일을 찾을 수 없습니다: {input_path}")
        continue

    df = pd.read_csv(input_path, low_memory=False)
    original_rows = len(df)

    print("\n[BEFORE]")
    print(f"Rows    : {original_rows:,}")
    print(f"Columns : {len(df.columns)}")

    print("\n기존 컬럼:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    # ========================================================
    # 5. protein_sequence 결측/유효성 검사
    # ========================================================
    protein_seq = (
        df["protein_sequence"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    missing_mask = (
            protein_seq.eq("")
            | protein_seq.str.lower().isin(["nan", "none", "null", "unresolved"])
    )

    missing_count = missing_mask.sum()

    print("\n[protein_sequence 검사]")
    print(f"전체 행                 : {len(df):,}")
    print(f"protein_sequence 존재   : {(~missing_mask).sum():,}")
    print(f"protein_sequence 없음   : {missing_count:,}")

    # ========================================================
    # 6. protein_sequence 결측 행 제거
    # ========================================================
    if missing_count > 0:
        print("\n삭제 대상 행 샘플:")
        sample_cols = [c for c in ["uniprot_id", "target_name", "protein_sequence"] if c in df.columns]
        missing_rows = df.loc[missing_mask, sample_cols]
        print(missing_rows.head(10).to_string(index=False))

        df = df.loc[~missing_mask].copy()
        print(f"\nprotein_sequence 없는 {missing_count:,}개 행을 삭제했습니다.")
    else:
        print("\n삭제할 protein_sequence missing 행이 없습니다.")

    # ========================================================
    # 7. 불필요 컬럼 삭제
    # ========================================================
    actual_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=actual_drop_columns)

    # ========================================================
    # 8. sequence_length 동기화 (최신 슬라이싱 서열 길이 적용)
    # ========================================================
    df["protein_sequence"] = df["protein_sequence"].astype(str).str.strip()
    df["sequence_length"] = df["protein_sequence"].str.len()
    print("\n[sequence_length 동기화] 최신 슬라이싱 서열 길이로 재계산 완료")

    # ========================================================
    # 9. 최종 컬럼 규격 정렬 및 검증
    # ========================================================
    missing_expected = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if not missing_expected:
        df = df[EXPECTED_COLUMNS]
        col_status = "PASS"
        print("\nColumn structure : PASS")
        print("→ 모든 기대 컬럼이 존재하며, 지정된 순서로 정렬되었습니다.")
    else:
        col_status = "WARNING"
        print("\nColumn structure : WARNING")
        print(f"→ 누락된 기대 컬럼: {missing_expected}")

    final_columns = df.columns.tolist()

    print("\n[AFTER]")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(final_columns)}")

    print("\n최종 컬럼:")
    for i, col in enumerate(final_columns, 1):
        print(f"{i:2d}. {col}")

    # ========================================================
    # 10. protein_sequence 최종 재검증
    # ========================================================
    final_seq_check = df["protein_sequence"].fillna("").astype(str).str.strip()
    final_missing_count = (final_seq_check.eq("") | final_seq_check.str.lower().isin(["nan", "none", "null"])).sum()

    print("\n[protein_sequence 최종 전체 검사]")
    print(f"전체 행                : {len(df):,}")
    print(f"protein_sequence 존재  : {len(df) - final_missing_count:,}")
    print(f"protein_sequence 없음  : {final_missing_count:,}")

    if final_missing_count == 0:
        print("RESULT : PASS (모든 행에 유효한 서열 존재)")
    else:
        print("RESULT : FAIL (서열 누락 행 존재)")

    # ========================================================
    # 11. 최종 파일 저장
    # ========================================================
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")

    # 요약 정보 기록
    summary_reports.append({
        "Input File": in_filename,
        "Output File": out_filename,
        "Original Rows": original_rows,
        "Final Rows": len(df),
        "Dropped Rows": missing_count,
        "Column Pass": col_status,
        "Max Seq Len": df["sequence_length"].max()
    })

# ============================================================
# 12. 최종 종합 요약 리포트
# ============================================================
print("\n\n" + "=" * 80)
print("전체 데이터 정제 작업 최종 리포트")
print("=" * 80)

summary_df = pd.DataFrame(summary_reports)
print(summary_df.to_string(index=False))

print(f"\n출력 저장 폴더: {OUTPUT_DIR}")