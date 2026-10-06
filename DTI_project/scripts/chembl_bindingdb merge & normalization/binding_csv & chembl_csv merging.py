"""
merge_dti_datasets.py

목적:
    ChEMBL CSV와 BindingDB CSV를 동일한 18개 컬럼 schema로 맞춘 뒤
    원본 measurement를 삭제/변경하지 않고 그대로 이어붙인다.

현재 단계에서 하지 않는 것:
    - 중복 제거
    - cross-source 중복 제거
    - 이상치 제거
    - affinity 값 필터링
    - relation 기반 필터링
    - pair aggregation
    - median / mean 계산

ChEMBL 특수 매핑:
    affinity_value_nM
        -> affinity_value

    affinity_relation_normalized
        -> affinity_relation

    affinity_unit
        -> 최종적으로 "nM"으로 통일
        (affinity_value에는 이미 nM 환산값인 affinity_value_nM 사용)

BindingDB:
    18개 컬럼을 그대로 사용
"""

import os
import pandas as pd


# =============================================================================
# 1. 경로 설정
# =============================================================================

CHEMBL_PATH = "../../data/output/chembl_dti.csv"
BINDINGDB_PATH = "../../data/output/bindingdb_dti.csv"
OUTPUT_PATH = "../../data/output/merged_dti_raw.csv"

CHUNK_SIZE = 200_000


# =============================================================================
# 2. 최종 통합 schema
# =============================================================================

FINAL_COLUMNS = [
    "source",
    "source_record_id",
    "chembl_id",
    "bindingdb_id",
    "smiles",
    "target_chembl_id",
    "uniprot_id",
    "target_name",
    "target_type",
    "organism",
    "assay_id",
    "assay_type",
    "affinity_type",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
    "pchembl_value",
    "publication_id",
]


# =============================================================================
# 3. 파일 존재 확인
# =============================================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n파일을 찾을 수 없습니다:\n{os.path.abspath(path)}"
        )


# =============================================================================
# 4. CSV header 확인
# =============================================================================

def get_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)


# =============================================================================
# 5. ChEMBL schema 변환
# =============================================================================

def normalize_chembl_chunk(df):
    """
    ChEMBL 데이터를 최종 18개 컬럼 schema로 변환한다.

    주의:
        원본 row 삭제 없음.
        원본 measurement 값 자체도 변경하지 않음.

    단,
        affinity_value_nM -> affinity_value
        affinity_relation_normalized -> affinity_relation
        affinity_unit -> "nM"
    """

    result = pd.DataFrame(index=df.index)

    # -------------------------------------------------------------------------
    # 동일한 이름의 컬럼
    # -------------------------------------------------------------------------

    same_columns = [
        "source",
        "source_record_id",
        "chembl_id",
        "bindingdb_id",
        "smiles",
        "target_chembl_id",
        "uniprot_id",
        "target_name",
        "target_type",
        "organism",
        "assay_id",
        "assay_type",
        "affinity_type",
        "pchembl_value",
        "publication_id",
    ]

    for col in same_columns:
        if col not in df.columns:
            raise ValueError(
                f"ChEMBL 파일에 필요한 컬럼이 없습니다: {col}"
            )

        result[col] = df[col]

    # -------------------------------------------------------------------------
    # ChEMBL affinity_value
    #
    # affinity_value_nM는 이미 nM로 환산된 값이므로
    # 최종 affinity_value에 그대로 사용
    # -------------------------------------------------------------------------

    if "affinity_value_nM" not in df.columns:
        raise ValueError(
            "ChEMBL 파일에 'affinity_value_nM' 컬럼이 없습니다."
        )

    result["affinity_value"] = df["affinity_value_nM"]

    # -------------------------------------------------------------------------
    # ChEMBL affinity_relation
    # -------------------------------------------------------------------------

    if "affinity_relation_normalized" not in df.columns:
        raise ValueError(
            "ChEMBL 파일에 'affinity_relation_normalized' 컬럼이 없습니다."
        )

    result["affinity_relation"] = df["affinity_relation_normalized"]

    # -------------------------------------------------------------------------
    # affinity_unit
    #
    # affinity_value가 이미 affinity_value_nM이므로
    # 최종 unit은 무조건 nM
    # -------------------------------------------------------------------------

    result["affinity_unit"] = "nM"

    # -------------------------------------------------------------------------
    # 최종 컬럼 순서
    # -------------------------------------------------------------------------

    return result[FINAL_COLUMNS]


# =============================================================================
# 6. BindingDB schema 변환
# =============================================================================

def normalize_bindingdb_chunk(df):
    """
    BindingDB는 이미 최종 schema와 동일하므로
    필요한 18개 컬럼만 동일한 순서로 정렬한다.

    값 자체는 변경하지 않는다.
    """

    missing_columns = [
        col for col in FINAL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "BindingDB 파일에 필요한 컬럼이 없습니다:\n"
            + "\n".join(missing_columns)
        )

    return df[FINAL_COLUMNS]


# =============================================================================
# 7. main
# =============================================================================

def main():

    print("=" * 90)
    print("ChEMBL + BindingDB RAW MERGE")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # 파일 확인
    # -------------------------------------------------------------------------

    check_file(CHEMBL_PATH)
    check_file(BINDINGDB_PATH)

    print(f"\nChEMBL   : {os.path.abspath(CHEMBL_PATH)}")
    print(f"BindingDB: {os.path.abspath(BINDINGDB_PATH)}")
    print(f"Output   : {os.path.abspath(OUTPUT_PATH)}")

    # -------------------------------------------------------------------------
    # Header 확인
    # -------------------------------------------------------------------------

    chembl_columns = get_columns(CHEMBL_PATH)
    bindingdb_columns = get_columns(BINDINGDB_PATH)

    print("\n" + "-" * 90)
    print("SCHEMA CHECK")
    print("-" * 90)

    print(f"ChEMBL columns    : {len(chembl_columns)}")
    print(f"BindingDB columns : {len(bindingdb_columns)}")
    print(f"Final columns     : {len(FINAL_COLUMNS)}")

    print("\n최종 컬럼:")
    for i, col in enumerate(FINAL_COLUMNS, 1):
        print(f"{i:2d}. {col}")

    # -------------------------------------------------------------------------
    # ChEMBL 필수 컬럼 확인
    # -------------------------------------------------------------------------

    required_chembl = [
        "source",
        "source_record_id",
        "chembl_id",
        "bindingdb_id",
        "smiles",
        "target_chembl_id",
        "uniprot_id",
        "target_name",
        "target_type",
        "organism",
        "assay_id",
        "assay_type",
        "affinity_type",
        "affinity_value_nM",
        "affinity_relation_normalized",
        "pchembl_value",
        "publication_id",
    ]

    missing_chembl = [
        col for col in required_chembl
        if col not in chembl_columns
    ]

    if missing_chembl:
        raise ValueError(
            "\nChEMBL 필수 컬럼 누락:\n"
            + "\n".join(missing_chembl)
        )

    # -------------------------------------------------------------------------
    # BindingDB 필수 컬럼 확인
    # -------------------------------------------------------------------------

    missing_bindingdb = [
        col for col in FINAL_COLUMNS
        if col not in bindingdb_columns
    ]

    if missing_bindingdb:
        raise ValueError(
            "\nBindingDB 필수 컬럼 누락:\n"
            + "\n".join(missing_bindingdb)
        )

    print("\nSchema check: PASS")

    # -------------------------------------------------------------------------
    # 기존 output 삭제
    #
    # 이전 실행 결과가 있으면 append할 때 섞이는 것을 방지
    # -------------------------------------------------------------------------

    if os.path.exists(OUTPUT_PATH):
        print("\n기존 output 파일 삭제:")
        print(f"  {OUTPUT_PATH}")
        os.remove(OUTPUT_PATH)

    # =========================================================================
    # 1. ChEMBL
    # =========================================================================

    print("\n" + "=" * 90)
    print("[1/2] ChEMBL 처리")
    print("=" * 90)

    chembl_total = 0
    first_write = True

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            CHEMBL_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        normalized = normalize_chembl_chunk(chunk)

        normalized.to_csv(
            OUTPUT_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False
        chembl_total += len(normalized)

        print(
            f"  Chunk {chunk_no}: "
            f"{len(normalized):,} rows "
            f"| 누적 {chembl_total:,}"
        )

    print(f"\nChEMBL 완료: {chembl_total:,} rows")

    # =========================================================================
    # 2. BindingDB
    # =========================================================================

    print("\n" + "=" * 90)
    print("[2/2] BindingDB 처리")
    print("=" * 90)

    bindingdb_total = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            BINDINGDB_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        normalized = normalize_bindingdb_chunk(chunk)

        normalized.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=False,
            index=False,
        )

        bindingdb_total += len(normalized)

        print(
            f"  Chunk {chunk_no}: "
            f"{len(normalized):,} rows "
            f"| 누적 {bindingdb_total:,}"
        )

    print(f"\nBindingDB 완료: {bindingdb_total:,} rows")

    # =========================================================================
    # 3. 최종 결과
    # =========================================================================

    expected_total = chembl_total + bindingdb_total

    print("\n" + "=" * 90)
    print("MERGE COMPLETE")
    print("=" * 90)

    print(f"ChEMBL rows       : {chembl_total:,}")
    print(f"BindingDB rows    : {bindingdb_total:,}")
    print(f"Expected total    : {expected_total:,}")
    print(f"Output            : {OUTPUT_PATH}")

    print("\n처리 내용:")
    print("  [O] 원본 measurement 유지")
    print("  [O] ChEMBL affinity_value_nM -> affinity_value")
    print("  [O] ChEMBL affinity_relation_normalized -> affinity_relation")
    print("  [O] ChEMBL affinity_unit -> nM")
    print("  [O] BindingDB affinity 값 그대로 유지")
    print("  [O] 중복 제거 없음")
    print("  [O] Filtering 없음")
    print("  [O] Aggregation 없음")

    print("\n최종 schema:")
    print(FINAL_COLUMNS)

    print("\n다음 단계에서 QC 및 중복 분석을 진행하면 됩니다.")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    main()