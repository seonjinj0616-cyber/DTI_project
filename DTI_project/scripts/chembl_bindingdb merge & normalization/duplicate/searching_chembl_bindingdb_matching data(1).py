import os
import pandas as pd


# ============================================================
# 0. 경로 설정
# ============================================================

INPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\bindingdb_chembl_merging\merged_dti_DEDUP5_RELATION_CLEAN.csv"
)

OUTPUT_DIR = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\bindingdb_chembl_merging\chembl_bindingdb_overlap_qc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. QC 매칭 기준
#
# 현재 사용 중인 QC 조건
#
# chembl_id
# + uniprot_id
# + affinity_type
# + affinity_value (소수점 둘째 자리 반올림)
# + affinity_unit
#
# publication_id는 현재 제외
# target_name도 현재 제외
# affinity_relation도 현재 제외
# ============================================================

GROUP_KEY_COLS = [
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "affinity_value_rounded",
    "affinity_unit",
]


# ============================================================
# 2. CSV 읽기
# ============================================================

print("=" * 100)
print("ChEMBL ↔ BindingDB 중복 그룹 실제 데이터 추출")
print("=" * 100)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"전체 행 수: {len(df):,}")
print()


# ============================================================
# 3. affinity_value 숫자 변환 및 반올림
#
# 원본 affinity_value는 그대로 유지
# QC용 컬럼만 추가
# ============================================================

df["affinity_value_numeric"] = pd.to_numeric(
    df["affinity_value"],
    errors="coerce"
)

df["affinity_value_rounded"] = (
    df["affinity_value_numeric"].round(2)
)

print("affinity_value QC 반올림: 소수점 둘째 자리")
print()


# ============================================================
# 4. source 분리
# ============================================================

chembl = df[df["source"] == "ChEMBL"].copy()
bindingdb = df[df["source"] == "BindingDB"].copy()

print(f"ChEMBL rows   : {len(chembl):,}")
print(f"BindingDB rows: {len(bindingdb):,}")
print()


# ============================================================
# 5. 각 source의 그룹별 행 개수 계산
# ============================================================

chembl_group_count = (
    chembl
    .groupby(GROUP_KEY_COLS, dropna=False)
    .size()
    .reset_index(name="chembl_count")
)

bindingdb_group_count = (
    bindingdb
    .groupby(GROUP_KEY_COLS, dropna=False)
    .size()
    .reset_index(name="bindingdb_count")
)


# ============================================================
# 6. 공통 그룹 찾기
# ============================================================

overlap_groups = pd.merge(
    chembl_group_count,
    bindingdb_group_count,
    on=GROUP_KEY_COLS,
    how="inner"
)

print(f"공통 그룹 수: {len(overlap_groups):,}")
print()


# ============================================================
# 7. 그룹별 관계 유형 생성
# ============================================================

overlap_groups["overlap_type"] = (
    overlap_groups["chembl_count"].astype(str)
    + ":"
    + overlap_groups["bindingdb_count"].astype(str)
)


# ============================================================
# 8. 각 그룹에 고유 ID 부여
#
# 예:
# OVERLAP_00000001
# OVERLAP_00000002
# ...
# ============================================================

overlap_groups = overlap_groups.reset_index(drop=True)

overlap_groups["matched_group_id"] = (
    "OVERLAP_"
    + (overlap_groups.index + 1)
    .astype(str)
    .str.zfill(8)
)


# ============================================================
# 9. 실제 ChEMBL 행 추출
# ============================================================

chembl_overlap = pd.merge(
    chembl,
    overlap_groups[
        GROUP_KEY_COLS
        + [
            "chembl_count",
            "bindingdb_count",
            "overlap_type",
            "matched_group_id"
        ]
    ],
    on=GROUP_KEY_COLS,
    how="inner"
)

chembl_overlap["matched_source"] = "ChEMBL"


# ============================================================
# 10. 실제 BindingDB 행 추출
# ============================================================

bindingdb_overlap = pd.merge(
    bindingdb,
    overlap_groups[
        GROUP_KEY_COLS
        + [
            "chembl_count",
            "bindingdb_count",
            "overlap_type",
            "matched_group_id"
        ]
    ],
    on=GROUP_KEY_COLS,
    how="inner"
)

bindingdb_overlap["matched_source"] = "BindingDB"


# ============================================================
# 11. ChEMBL + BindingDB 실제 중복 데이터 합치기
# ============================================================

duplicate_rows = pd.concat(
    [
        chembl_overlap,
        bindingdb_overlap
    ],
    ignore_index=True
)


# ============================================================
# 12. 정렬
#
# 같은 matched_group_id끼리 모아서
# ChEMBL → BindingDB 순으로 확인하기 쉽게 정렬
# ============================================================

duplicate_rows["source_order"] = (
    duplicate_rows["matched_source"]
    .map({
        "ChEMBL": 0,
        "BindingDB": 1
    })
)

duplicate_rows = (
    duplicate_rows
    .sort_values(
        [
            "matched_group_id",
            "source_order"
        ]
    )
    .drop(columns=["source_order"])
    .reset_index(drop=True)
)


# ============================================================
# 13. 결과 저장
# ============================================================

ALL_DUPLICATE_FILE = os.path.join(
    OUTPUT_DIR,
    "chembl_bindingdb_ALL_OVERLAPPING_ROWS.csv"
)

duplicate_rows.to_csv(
    ALL_DUPLICATE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 그룹 정보 + 실제 행 개수 확인
# ============================================================

group_summary = (
    duplicate_rows
    .groupby(
        [
            "matched_group_id",
            "overlap_type"
        ],
        as_index=False
    )
    .agg(
        total_rows=("source", "size"),
        chembl_rows=(
            "matched_source",
            lambda x: (x == "ChEMBL").sum()
        ),
        bindingdb_rows=(
            "matched_source",
            lambda x: (x == "BindingDB").sum()
        )
    )
)


GROUP_SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "chembl_bindingdb_OVERLAPPING_GROUP_SUMMARY.csv"
)

group_summary.to_csv(
    GROUP_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 1:1만 별도로 추출
# ============================================================

one_to_one_groups = overlap_groups[
    (overlap_groups["chembl_count"] == 1) &
    (overlap_groups["bindingdb_count"] == 1)
].copy()

one_to_one_rows = duplicate_rows[
    duplicate_rows["overlap_type"] == "1:1"
].copy()


ONE_TO_ONE_FILE = os.path.join(
    OUTPUT_DIR,
    "chembl_bindingdb_OVERLAPPING_1to1_ROWS.csv"
)

one_to_one_rows.to_csv(
    ONE_TO_ONE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 1:N / N:1 / N:N만 별도 추출
# ============================================================

non_one_to_one_rows = duplicate_rows[
    duplicate_rows["overlap_type"] != "1:1"
].copy()


NON_ONE_TO_ONE_FILE = os.path.join(
    OUTPUT_DIR,
    "chembl_bindingdb_OVERLAPPING_NON_1to1_ROWS.csv"
)

non_one_to_one_rows.to_csv(
    NON_ONE_TO_ONE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 최종 QC 출력
# ============================================================

print("=" * 100)
print("중복 데이터 추출 완료")
print("=" * 100)

print(f"공통 그룹 수                 : {len(overlap_groups):,}")
print(f"1:1 그룹 수                  : {len(one_to_one_groups):,}")
print(
    f"1:N / N:1 / N:N 그룹 수     : "
    f"{len(overlap_groups) - len(one_to_one_groups):,}"
)

print()
print(f"실제 중복 ChEMBL 행 수       : {len(chembl_overlap):,}")
print(f"실제 중복 BindingDB 행 수    : {len(bindingdb_overlap):,}")
print(f"실제 중복 전체 행 수         : {len(duplicate_rows):,}")

print()
print("=" * 100)
print("저장 파일")
print("=" * 100)

print(f"전체 중복 실제 행 :")
print(ALL_DUPLICATE_FILE)

print()
print(f"중복 그룹 요약 :")
print(GROUP_SUMMARY_FILE)

print()
print(f"1:1 중복 데이터 :")
print(ONE_TO_ONE_FILE)

print()
print(f"1:N / N:1 / N:N 데이터 :")
print(NON_ONE_TO_ONE_FILE)

print()
print("=" * 100)
print("분석 종료")
print("=" * 100)