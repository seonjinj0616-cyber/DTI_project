import os
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\output"

INPUT_FILE = "DEDUP8_affinity_relation_mean.csv"

INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILE)

OUTPUT_FILE = "DEDUP9_affinity_relation_filtered.csv"

OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_FILE)


# ============================================================
# 2. 동일 그룹 판단 기준
#
# 아래 4개 컬럼의 내용이 모두 동일하면 같은 그룹
# ============================================================

GROUP_COLS = [
    "smiles",
    "affinity_type",
    "uniprot_id",
    "target_name"
]

RELATION_COL = "affinity_relation"
VALUE_COL = "affinity_value_mean"


# ============================================================
# 3. CSV LOAD
# ============================================================

print("=" * 80)
print("CSV LOAD")
print("=" * 80)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(f"파일 : {INPUT_PATH}")
print(f"전체 행 수 : {len(df):,}")
print()


# ============================================================
# 4. 필수 컬럼 확인
# ============================================================

required_cols = GROUP_COLS + [
    RELATION_COL,
    VALUE_COL
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_cols}"
    )


# ============================================================
# 5. 기본 전처리
# ============================================================

df[RELATION_COL] = (
    df[RELATION_COL]
    .astype(str)
    .str.strip()
)

# affinity_value_mean 숫자형 변환
df[VALUE_COL] = pd.to_numeric(
    df[VALUE_COL],
    errors="coerce"
)

print("affinity_relation 분포:")
print(df[RELATION_COL].value_counts(dropna=False))

print()

print("affinity_value_mean 결측:")
print(df[VALUE_COL].isna().sum())

print()


# ============================================================
# 6. 그룹별 affinity_relation 종류 생성
# ============================================================

grouped = (
    df.groupby(
        GROUP_COLS,
        dropna=False
    )[RELATION_COL]
    .agg(lambda x: frozenset(x))
    .reset_index()
)

grouped["relation_count"] = grouped[
    RELATION_COL
].apply(len)

grouped["relation_set_str"] = grouped[
    RELATION_COL
].apply(
    lambda x: ",".join(sorted(x))
)


# ============================================================
# 7. 그룹 유형 확인
# ============================================================

# ------------------------------------------------------------
# A. '=' + ('<' 또는 '>') 그룹
# ------------------------------------------------------------

eq_mixed_mask = (
    grouped[RELATION_COL].apply(
        lambda x: "=" in x
    )
    &
    grouped[RELATION_COL].apply(
        lambda x: "<" in x or ">" in x
    )
)

eq_mixed_groups = grouped[
    eq_mixed_mask
].copy()


# ------------------------------------------------------------
# B. '<'와 '>'만 존재하는 그룹
# ------------------------------------------------------------

lt_gt_mask = grouped[RELATION_COL].apply(
    lambda x: (
        "<" in x
        and ">" in x
        and "=" not in x
    )
)

lt_gt_groups = grouped[
    lt_gt_mask
].copy()


# ------------------------------------------------------------
# C. relation이 정확히 1개만 존재하는 그룹
# ------------------------------------------------------------

single_relation_mask = (
    grouped["relation_count"] == 1
)

single_relation_groups = grouped[
    single_relation_mask
].copy()


print("=" * 80)
print("GROUP SUMMARY")
print("=" * 80)

print(
    f"전체 고유 그룹 수                         : "
    f"{len(grouped):,}"
)

print(
    f"'=' + ('<' 또는 '>') 그룹                 : "
    f"{len(eq_mixed_groups):,}"
)

print(
    f"'<,>'만 존재하는 그룹                     : "
    f"{len(lt_gt_groups):,}"
)

print(
    f"relation 1종류만 존재하는 그룹            : "
    f"{len(single_relation_groups):,}"
)

print()


# ============================================================
# 8. 삭제/유지용 그룹 Key 생성
# ============================================================

# ------------------------------------------------------------
# 8-1. '=' + '<' 또는 '>' 그룹
#
# 해당 그룹에서는 '=' 데이터만 유지
# ------------------------------------------------------------

eq_mixed_keys = eq_mixed_groups[
    GROUP_COLS
].copy()


# ------------------------------------------------------------
# 8-2. '<' + '>' 그룹
#
# 해당 그룹에서
#
# affinity_value_mean >= 1000 → '>' 유지
# affinity_value_mean < 1000  → '<' 유지
# ------------------------------------------------------------

lt_gt_keys = lt_gt_groups[
    GROUP_COLS
].copy()


# ============================================================
# 9. 각 행이 어떤 그룹에 속하는지 표시
# ============================================================

df["_group_type"] = "OTHER"


# '=' + 비교기호 그룹
df.loc[
    df.set_index(GROUP_COLS).index.isin(
        eq_mixed_keys.set_index(GROUP_COLS).index
    ),
    "_group_type"
] = "EQ_MIXED"


# '<' + '>' 그룹
df.loc[
    df.set_index(GROUP_COLS).index.isin(
        lt_gt_keys.set_index(GROUP_COLS).index
    ),
    "_group_type"
] = "LT_GT"


# relation 하나만 존재하는 그룹
single_keys = single_relation_groups[
    GROUP_COLS
].copy()

df.loc[
    df.set_index(GROUP_COLS).index.isin(
        single_keys.set_index(GROUP_COLS).index
    ),
    "_group_type"
] = "SINGLE"


# ============================================================
# 10. 최종 유지 조건
# ============================================================

# 기본적으로 모든 행 유지
keep_mask = pd.Series(
    True,
    index=df.index
)


# ============================================================
# CASE 1
#
# '=' + '<' 또는 '>' 그룹
#
# '=' 데이터만 유지
# '<', '>' 삭제
# ============================================================

case1_mask = (
    df["_group_type"] == "EQ_MIXED"
)

case1_keep = (
    case1_mask
    &
    (df[RELATION_COL] == "=")
)

case1_delete = (
    case1_mask
    &
    (df[RELATION_COL] != "=")
)

keep_mask.loc[case1_delete] = False


# ============================================================
# CASE 2
#
# '<' + '>'만 존재하는 그룹
#
# affinity_value_mean >= 1000
#     → '>' 데이터만 유지
#
# affinity_value_mean < 1000
#     → '<' 데이터만 유지
# ============================================================

case2_mask = (
    df["_group_type"] == "LT_GT"
)


# >= 1000 → >
case2_keep_gt = (
    case2_mask
    &
    (df[VALUE_COL] >= 1000)
    &
    (df[RELATION_COL] == ">")
)


# < 1000 → <
case2_keep_lt = (
    case2_mask
    &
    (df[VALUE_COL] < 1000)
    &
    (df[RELATION_COL] == "<")
)


case2_keep = (
    case2_keep_gt
    |
    case2_keep_lt
)


# CASE 2에서 위 조건에 맞지 않는 것은 삭제
case2_delete = (
    case2_mask
    &
    ~case2_keep
)

keep_mask.loc[case2_delete] = False


# ============================================================
# 11. 최종 데이터 생성
# ============================================================

df_filtered = df[
    keep_mask
].copy()


# 임시 컬럼 제거
df_filtered.drop(
    columns=["_group_type"],
    inplace=True
)


# ============================================================
# 12. 결과 저장
# ============================================================

df_filtered.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 삭제/유지 통계
# ============================================================

original_count = len(df)

filtered_count = len(df_filtered)

deleted_count = (
    original_count - filtered_count
)


print("=" * 80)
print("FILTER RESULT")
print("=" * 80)

print(
    f"원본 행 수                              : "
    f"{original_count:,}"
)

print(
    f"최종 행 수                              : "
    f"{filtered_count:,}"
)

print(
    f"삭제 행 수                              : "
    f"{deleted_count:,}"
)

print(
    f"삭제 비율                               : "
    f"{deleted_count / original_count * 100:.4f}%"
)

print()


# ============================================================
# 14. CASE 1 통계
# ============================================================

case1_original = case1_mask.sum()
case1_deleted = case1_delete.sum()
case1_kept = case1_keep.sum()

print("=" * 80)
print("CASE 1 : '=' + ('<' 또는 '>')")
print("=" * 80)

print(
    f"해당 그룹 수                            : "
    f"{len(eq_mixed_groups):,}"
)

print(
    f"원본 행 수                              : "
    f"{case1_original:,}"
)

print(
    f"'=' 유지 행 수                          : "
    f"{case1_kept:,}"
)

print(
    f"'< 또는 >' 삭제 행 수                   : "
    f"{case1_deleted:,}"
)

print()


# ============================================================
# 15. CASE 2 통계
# ============================================================

case2_original = case2_mask.sum()
case2_deleted = case2_delete.sum()

print("=" * 80)
print("CASE 2 : '<' + '>'만 존재")
print("=" * 80)

print(
    f"해당 그룹 수                            : "
    f"{len(lt_gt_groups):,}"
)

print(
    f"원본 행 수                              : "
    f"{case2_original:,}"
)

print(
    f"affinity_value_mean >= 1000 → '>' 유지 : "
    f"{case2_keep_gt.sum():,}"
)

print(
    f"affinity_value_mean < 1000 → '<' 유지  : "
    f"{case2_keep_lt.sum():,}"
)

print(
    f"삭제 행 수                              : "
    f"{case2_deleted:,}"
)

print()


# ============================================================
# 16. CASE 3
#
# relation이 정확히 1종류만 존재하는 그룹
#
# affinity_type별 그룹 분포
# ============================================================

print("=" * 80)
print("CASE 3 : affinity_relation이 정확히 1종류만 존재")
print("=" * 80)

print(
    f"전체 그룹 수                            : "
    f"{len(single_relation_groups):,}"
)

print()


# relation별 + affinity_type별 그룹 수
single_relation_type_distribution = (
    single_relation_groups
    .groupby(
        [
            RELATION_COL,
            "affinity_type"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="group_count"
    )
    .sort_values(
        [
            RELATION_COL,
            "group_count"
        ],
        ascending=[True, False]
    )
)


print("relation + affinity_type별 그룹 수")
print(
    single_relation_type_distribution.to_string(
        index=False
    )
)

print()


# ============================================================
# 17. affinity_type 전체 분포
# ============================================================

affinity_type_distribution = (
    single_relation_groups
    .groupby(
        "affinity_type",
        dropna=False
    )
    .size()
    .sort_values(
        ascending=False
    )
)


print("=" * 80)
print("CASE 3 : affinity_type 전체 분포")
print("=" * 80)

print(
    affinity_type_distribution
)

print()


# ============================================================
# 18. 실제 행 기준 affinity_type 분포
#
# 그룹 수가 아니라 실제 데이터 행 수
# ============================================================

single_keys_index = single_relation_groups[
    GROUP_COLS
].set_index(GROUP_COLS).index

single_rows = df[
    df.set_index(GROUP_COLS).index.isin(
        single_keys_index
    )
].copy()


single_row_type_distribution = (
    single_rows
    .groupby(
        "affinity_type",
        dropna=False
    )
    .size()
    .sort_values(
        ascending=False
    )
)


print("=" * 80)
print("CASE 3 : 실제 행 기준 affinity_type 분포")
print("=" * 80)

print(
    single_row_type_distribution
)

print()


# ============================================================
# 19. 결과 파일 정보
# ============================================================

print("=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"저장 파일 : {OUTPUT_PATH}"
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
