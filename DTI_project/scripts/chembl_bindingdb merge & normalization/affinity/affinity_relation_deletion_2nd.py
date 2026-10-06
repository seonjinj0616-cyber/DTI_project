
import os
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\output"

INPUT_FILE = "DEDUP9_affinity_relation_filtered.csv"

INPUT_PATH = os.path.join(
    DATA_DIR,
    INPUT_FILE
)

OUTPUT_FILE = "DEDUP10_affinity_filtered.csv"

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    OUTPUT_FILE
)


# ============================================================
# 2. 동일 그룹 판단 기준
#
# 아래 4개 컬럼의 값이 모두 동일하면 동일 그룹
# ============================================================

GROUP_COLS = [
    "smiles",
    "affinity_type",
    "uniprot_id",
    "target_name"
]

RELATION_COL = "affinity_relation"

MEAN_COL = "affinity_value_mean"

MIN_COL = "affinity_value_min"

MAX_COL = "affinity_value_max"


# ============================================================
# 3. CSV LOAD
# ============================================================

print("=" * 100)
print("CSV LOAD")
print("=" * 100)

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
    MEAN_COL,
    MIN_COL,
    MAX_COL
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

df["affinity_type"] = (
    df["affinity_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

for col in [
    MEAN_COL,
    MIN_COL,
    MAX_COL
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 6. 원본 통계
# ============================================================

original_count = len(df)

print("=" * 100)
print("ORIGINAL DATA")
print("=" * 100)

print(
    f"전체 행 수 : {original_count:,}"
)

print("\naffinity_type 분포:")
print(
    df["affinity_type"]
    .value_counts(dropna=False)
)

print("\naffinity_relation 분포:")
print(
    df[RELATION_COL]
    .value_counts(dropna=False)
)

print()


# ============================================================
# 7. CASE 1
#
# IC50 / Ki
# affinity_relation이 '='가 아닌 데이터 삭제
# ============================================================

print("=" * 100)
print("CASE 1 : IC50 / Ki에서 '=' 이외 relation 삭제")
print("=" * 100)


case1_mask = (
    df["affinity_type"].isin(["ic50", "ki"])
)

case1_delete_mask = (
    case1_mask
    &
    (df[RELATION_COL] != "=")
)

case1_delete_count = case1_delete_mask.sum()

print(
    f"IC50 / Ki 전체 행 수 : "
    f"{case1_mask.sum():,}"
)

print(
    f"삭제 대상 행 수      : "
    f"{case1_delete_count:,}"
)

# 삭제
df = df[
    ~case1_delete_mask
].copy()

print(
    f"CASE 1 처리 후 행 수 : "
    f"{len(df):,}"
)

print()


# ============================================================
# 8. CASE 2
#
# Kd + '<'
#
# affinity_value_mean > 100
#     → 삭제
#
# affinity_value_mean <= 100
#     → affinity_relation '=' 변경
# ============================================================

print("=" * 100)
print("CASE 2 : KD + '<'")
print("=" * 100)


case2_mask = (
    (df["affinity_type"] == "kd")
    &
    (df[RELATION_COL] == "<")
)


# ------------------------------------------------------------
# 100 초과 → 삭제
# ------------------------------------------------------------

case2_delete_mask = (
    case2_mask
    &
    (df[MEAN_COL] > 100)
)

case2_delete_count = case2_delete_mask.sum()


# ------------------------------------------------------------
# 100 이하 → '=' 변경
# ------------------------------------------------------------

case2_change_mask = (
    case2_mask
    &
    (df[MEAN_COL] <= 100)
)

case2_change_count = case2_change_mask.sum()


print(
    f"KD + '<' 전체 행 수        : "
    f"{case2_mask.sum():,}"
)

print(
    f"100 초과 → 삭제             : "
    f"{case2_delete_count:,}"
)

print(
    f"100 이하 → '=' 변경         : "
    f"{case2_change_count:,}"
)


# 먼저 삭제
df = df[
    ~case2_delete_mask
].copy()


# 남은 <=100 데이터를 '='로 변경
df.loc[
    case2_change_mask.loc[df.index],
    RELATION_COL
] = "="


print(
    f"CASE 2 처리 후 행 수        : "
    f"{len(df):,}"
)

print()


# ============================================================
# 9. CASE 3
#
# Kd + '>'
#
# affinity_value_mean < 1000
#     → 삭제
#
# >= 1000
#     → 유지
# ============================================================

print("=" * 100)
print("CASE 3 : KD + '>'")
print("=" * 100)


case3_mask = (
    (df["affinity_type"] == "kd")
    &
    (df[RELATION_COL] == ">")
)


case3_delete_mask = (
    case3_mask
    &
    (df[MEAN_COL] < 1000)
)

case3_delete_count = case3_delete_mask.sum()


case3_keep_count = (
    case3_mask.sum()
    - case3_delete_count
)


print(
    f"KD + '>' 전체 행 수        : "
    f"{case3_mask.sum():,}"
)

print(
    f"1000 미만 → 삭제            : "
    f"{case3_delete_count:,}"
)

print(
    f"1000 이상 → 유지            : "
    f"{case3_keep_count:,}"
)


df = df[
    ~case3_delete_mask
].copy()


print(
    f"CASE 3 처리 후 행 수        : "
    f"{len(df):,}"
)

print()


# ============================================================
# 10. CASE 4
#
# 최종 KD + '>' affinity_value_mean 분포
#
# 1000 이상 ~ 10000 이하
# 10000 초과 ~ 50000 이하
# 50000 초과
# ============================================================

print("=" * 100)
print("CASE 4 : 최종 KD + '>' affinity_value_mean 분포")
print("=" * 100)


kd_gt = df[
    (df["affinity_type"] == "kd")
    &
    (df[RELATION_COL] == ">")
].copy()


# ------------------------------------------------------------
# 구간 분류
# ------------------------------------------------------------

kd_gt["value_range"] = pd.cut(
    kd_gt[MEAN_COL],
    bins=[
        float("-inf"),
        999.999999999,
        10000,
        50000,
        float("inf")
    ],
    labels=[
        "< 1000",
        "1000 ~ 10000",
        "10000 초과 ~ 50000",
        "> 50000"
    ],
    right=True
)


# 요청한 최종 범위만 출력
range_distribution = (
    kd_gt["value_range"]
    .value_counts()
    .reindex(
        [
            "1000 ~ 10000",
            "10000 초과 ~ 50000",
            "> 50000"
        ],
        fill_value=0
    )
)


print(
    f"최종 KD + '>' 전체 행 수 : "
    f"{len(kd_gt):,}"
)

print()

print("affinity_value_mean 분포:")

for category, count in range_distribution.items():

    percentage = (
        count / len(kd_gt) * 100
        if len(kd_gt) > 0
        else 0
    )

    print(
        f"{category:25s} : "
        f"{count:>10,} "
        f"({percentage:.4f}%)"
    )

print()


# ============================================================
# 11. CASE 5
#
# affinity_value_mean과
# affinity_value_min / max 비교
#
# max / mean >= 10
#     OR
#
# mean / min >= 10
#
# → mean과 min/max가 10배 이상 차이
# ============================================================

print("=" * 100)
print("CASE 5 : affinity_value_mean vs min/max 10배 이상 차이")
print("=" * 100)


# ------------------------------------------------------------
# 0 이하 값 제거
#
# ratio 계산을 위해 양수 데이터만 사용
# ------------------------------------------------------------

valid_ratio_mask = (
    (df[MEAN_COL] > 0)
    &
    (df[MIN_COL] > 0)
    &
    (df[MAX_COL] > 0)
)


ratio_df = df[
    valid_ratio_mask
].copy()


# ------------------------------------------------------------
# ratio 계산
# ------------------------------------------------------------

ratio_df["max_mean_ratio"] = (
    ratio_df[MAX_COL]
    /
    ratio_df[MEAN_COL]
)


ratio_df["mean_min_ratio"] = (
    ratio_df[MEAN_COL]
    /
    ratio_df[MIN_COL]
)


# ------------------------------------------------------------
# 10배 이상 조건
# ------------------------------------------------------------

ratio_df["max_10x"] = (
    ratio_df["max_mean_ratio"] >= 10
)

ratio_df["min_10x"] = (
    ratio_df["mean_min_ratio"] >= 10
)


ratio_df["10x_difference"] = (
    ratio_df["max_10x"]
    |
    ratio_df["min_10x"]
)


# ------------------------------------------------------------
# affinity_type별 통계
# ------------------------------------------------------------

type_distribution = (
    df["affinity_type"]
    .value_counts()
)


ratio_distribution = (
    ratio_df
    .groupby("affinity_type")
    .agg(
        total_valid_rows=("affinity_type", "size"),
        max_mean_10x=("max_10x", "sum"),
        mean_min_10x=("min_10x", "sum"),
        either_10x=("10x_difference", "sum")
    )
    .reindex(
        ["ic50", "ki", "kd"]
    )
)


# 전체 데이터 기준
ratio_distribution["total_rows"] = (
    type_distribution
    .reindex(
        ["ic50", "ki", "kd"]
    )
    .fillna(0)
    .astype(int)
)


# 순서 정리
ratio_distribution = ratio_distribution[
    [
        "total_rows",
        "total_valid_rows",
        "max_mean_10x",
        "mean_min_10x",
        "either_10x"
    ]
]


print(
    ratio_distribution
)

print()


# ------------------------------------------------------------
# 비율도 출력
# ------------------------------------------------------------

print("10배 이상 차이 비율:")

for affinity_type in [
    "ic50",
    "ki",
    "kd"
]:

    if affinity_type not in ratio_distribution.index:
        continue

    total_valid = ratio_distribution.loc[
        affinity_type,
        "total_valid_rows"
    ]

    either = ratio_distribution.loc[
        affinity_type,
        "either_10x"
    ]

    if total_valid > 0:
        percentage = (
            either / total_valid * 100
        )
    else:
        percentage = 0

    print(
        f"{affinity_type.upper():5s} : "
        f"{either:,} / {total_valid:,} "
        f"({percentage:.4f}%)"
    )

print()


# ============================================================
# 12. CASE 5 상세 분포
#
# max/mean >= 10
# mean/min >= 10
# 둘 다 해당
# ============================================================

print("=" * 100)
print("CASE 5 상세")
print("=" * 100)


for affinity_type in [
    "ic50",
    "ki",
    "kd"
]:

    subset = ratio_df[
        ratio_df["affinity_type"] == affinity_type
    ]

    print(f"\n[{affinity_type.upper()}]")

    print(
        f"전체 valid : {len(subset):,}"
    )

    print(
        f"MAX / MEAN >= 10 : "
        f"{subset['max_10x'].sum():,}"
    )

    print(
        f"MEAN / MIN >= 10 : "
        f"{subset['min_10x'].sum():,}"
    )

    print(
        f"둘 중 하나라도 10배 이상 : "
        f"{subset['10x_difference'].sum():,}"
    )

print()


# ============================================================
# 13. 10배 이상 차이나는 실제 데이터 저장
# ============================================================

outlier_10x = ratio_df[
    ratio_df["10x_difference"]
].copy()


OUTLIER_FILE = "DEDUP9_affinity_value_10x_difference.csv"

OUTLIER_PATH = os.path.join(
    DATA_DIR,
    OUTLIER_FILE
)


outlier_10x.to_csv(
    OUTLIER_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("=" * 100)
print("10X DIFFERENCE DATA")
print("=" * 100)

print(
    f"10배 이상 차이 데이터 : "
    f"{len(outlier_10x):,}"
)

print(
    f"저장 : {OUTLIER_PATH}"
)

print()


# ============================================================
# 14. 최종 필터링 데이터 저장
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 최종 결과
# ============================================================

final_count = len(df)

deleted_total = (
    original_count
    - final_count
)


print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(
    f"원본 행 수       : "
    f"{original_count:,}"
)

print(
    f"최종 행 수       : "
    f"{final_count:,}"
)

print(
    f"총 삭제 행 수    : "
    f"{deleted_total:,}"
)

print(
    f"삭제 비율        : "
    f"{deleted_total / original_count * 100:.4f}%"
)

print()

print(
    f"최종 파일        : "
    f"{OUTPUT_PATH}"
)

print()

print("=" * 100)
print("DONE")
print("=" * 100)
