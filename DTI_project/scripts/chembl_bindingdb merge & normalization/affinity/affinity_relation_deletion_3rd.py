
import os
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\output"

INPUT_FILE = "DEDUP10_affinity_filtered.csv"

INPUT_PATH = os.path.join(
    DATA_DIR,
    INPUT_FILE
)

OUTPUT_FILE = "merged_chembl_binding_final.csv"

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    OUTPUT_FILE
)


# ============================================================
# 2. 동일 그룹 판단 기준
#
# 모든 그룹 판단은 아래 4개 컬럼이 동일한 것을 기준으로 함
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

df["affinity_type"] = (
    df["affinity_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df[RELATION_COL] = (
    df[RELATION_COL]
    .astype(str)
    .str.strip()
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


original_count = len(df)


# ============================================================
# 6. CASE 1
#
# affinity_value_mean vs min/max
#
# 다음 중 하나라도 10배 이상이면 전체 삭제
#
# MAX / MEAN >= 10
# 또는
# MEAN / MIN >= 10
#
# affinity_type 관계없이 적용
# ============================================================

print("=" * 100)
print("CASE 1 : affinity_value_mean vs min/max 10배 이상 차이")
print("=" * 100)


# ------------------------------------------------------------
# 정상적인 ratio 계산이 가능한 데이터
#
# mean, min, max가 모두 양수여야 함
# ------------------------------------------------------------

valid_ratio_mask = (
    (df[MEAN_COL] > 0)
    &
    (df[MIN_COL] > 0)
    &
    (df[MAX_COL] > 0)
)


# ratio 계산용 임시 컬럼
df["_max_mean_ratio"] = float("nan")
df["_mean_min_ratio"] = float("nan")


df.loc[
    valid_ratio_mask,
    "_max_mean_ratio"
] = (
    df.loc[
        valid_ratio_mask,
        MAX_COL
    ]
    /
    df.loc[
        valid_ratio_mask,
        MEAN_COL
    ]
)


df.loc[
    valid_ratio_mask,
    "_mean_min_ratio"
] = (
    df.loc[
        valid_ratio_mask,
        MEAN_COL
    ]
    /
    df.loc[
        valid_ratio_mask,
        MIN_COL
    ]
)


# ------------------------------------------------------------
# 10배 이상 여부
# ------------------------------------------------------------

max_10x_mask = (
    df["_max_mean_ratio"] >= 10
)

min_10x_mask = (
    df["_mean_min_ratio"] >= 10
)


delete_10x_mask = (
    max_10x_mask
    |
    min_10x_mask
)


delete_10x_count = delete_10x_mask.sum()


print(
    f"전체 데이터                         : "
    f"{len(df):,}"
)

print(
    f"MAX / MEAN >= 10                    : "
    f"{max_10x_mask.sum():,}"
)

print(
    f"MEAN / MIN >= 10                    : "
    f"{min_10x_mask.sum():,}"
)

print(
    f"둘 중 하나라도 10배 이상             : "
    f"{delete_10x_count:,}"
)


# ------------------------------------------------------------
# affinity_type별 삭제 수
# ------------------------------------------------------------

print("\n10배 이상 차이 삭제 데이터의 affinity_type 분포:")

print(
    df.loc[
        delete_10x_mask,
        "affinity_type"
    ]
    .value_counts()
)


# ------------------------------------------------------------
# 실제 삭제
# ------------------------------------------------------------

df = df[
    ~delete_10x_mask
].copy()


print(
    f"\nCASE 1 처리 후 행 수                : "
    f"{len(df):,}"
)

print()


# ============================================================
# 7. CASE 2
#
# affinity_type = Kd
# affinity_relation = ">"
#
# 해당 데이터 전체 삭제
# ============================================================

print("=" * 100)
print("CASE 2 : Kd + affinity_relation '>' 삭제")
print("=" * 100)


kd_gt_mask = (
    (df["affinity_type"] == "kd")
    &
    (df[RELATION_COL] == ">")
)


kd_gt_delete_count = kd_gt_mask.sum()


print(
    f"KD + '>' 삭제 대상                    : "
    f"{kd_gt_delete_count:,}"
)


df = df[
    ~kd_gt_mask
].copy()


print(
    f"CASE 2 처리 후 행 수                : "
    f"{len(df):,}"
)

print()


# ============================================================
# 8. 임시 컬럼 제거
# ============================================================

df.drop(
    columns=[
        "_max_mean_ratio",
        "_mean_min_ratio"
    ],
    inplace=True
)


# ============================================================
# 9. CASE 3
#
# affinity_type별 affinity_value_mean 분포
#
# < 10
# 10 이상 ~ 100 미만
# 100 이상 ~ 1000 미만
# 1000 초과
#
# 추가적으로 1000 정확히 해당하는 데이터도 확인
# ============================================================

print("=" * 100)
print("CASE 3 : affinity_type별 affinity_value_mean 분포")
print("=" * 100)


# ------------------------------------------------------------
# 분포 구간 함수
# ------------------------------------------------------------

def classify_value(value):

    if pd.isna(value):
        return "결측"

    if value < 10:
        return "< 10"

    elif value < 100:
        return "10 이상 ~ 100 미만"

    elif value < 1000:
        return "100 이상 ~ 1000 미만"

    elif value == 1000:
        return "= 1000"

    else:
        return "> 1000"


df["_value_range"] = (
    df[MEAN_COL]
    .apply(classify_value)
)


# ------------------------------------------------------------
# affinity_type × value_range
# ------------------------------------------------------------

distribution = (
    df
    .groupby(
        [
            "affinity_type",
            "_value_range"
        ],
        dropna=False
    )
    .size()
    .unstack(
        fill_value=0
    )
)


# 원하는 컬럼 순서
range_order = [
    "< 10",
    "10 이상 ~ 100 미만",
    "100 이상 ~ 1000 미만",
    "= 1000",
    "> 1000",
    "결측"
]


for col in range_order:

    if col not in distribution.columns:
        distribution[col] = 0


distribution = distribution[
    range_order
]


# TOTAL
distribution["TOTAL"] = (
    distribution[
        range_order
    ].sum(axis=1)
)


print("\n[affinity_type별 개수]")

print(
    distribution
)


# ============================================================
# 10. affinity_type별 비율
# ============================================================

print("\n[affinity_type별 비율 %]")


percentage = (
    distribution[
        range_order
    ]
    .div(
        distribution["TOTAL"],
        axis=0
    )
    * 100
)


print(
    percentage.round(4)
)

print()


# ============================================================
# 11. 실제 affinity_type별 전체 통계
# ============================================================

print("=" * 100)
print("AFFINITY TYPE SUMMARY")
print("=" * 100)


for affinity_type in [
    "ic50",
    "ki",
    "kd"
]:

    subset = df[
        df["affinity_type"] == affinity_type
    ]

    print(
        f"\n[{affinity_type.upper()}]"
    )

    print(
        f"전체 데이터 : {len(subset):,}"
    )

    print(
        f"< 10                    : "
        f"{(subset[MEAN_COL] < 10).sum():,}"
    )

    mask_10_100 = (
            (subset[MEAN_COL] >= 10)
            & (subset[MEAN_COL] < 100)
    )

    print(
        f"10 이상 ~ 100 미만      : "
        f"{mask_10_100.sum():,}"
    )

    mask_100_1000 = (
            (subset[MEAN_COL] >= 100)
            & (subset[MEAN_COL] < 1000)
    )

    print(
        f"100 이상 ~ 1000 미만    : "
        f"{mask_100_1000.sum():,}"
    )

    print(
        f"1000                    : "
        f"{(subset[MEAN_COL] == 1000).sum():,}"
    )

    print(
        f"> 1000                  : "
        f"{(subset[MEAN_COL] > 1000).sum():,}"
    )

    print(
        f"결측                    : "
        f"{subset[MEAN_COL].isna().sum():,}"
    )


# ============================================================
# 12. 분포 결과 저장
# ============================================================

DISTRIBUTION_FILE = (
    "DEDUP10_affinity_value_distribution.csv"
)

DISTRIBUTION_PATH = os.path.join(
    DATA_DIR,
    DISTRIBUTION_FILE
)


distribution.to_csv(
    DISTRIBUTION_PATH,
    encoding="utf-8-sig"
)


# ============================================================
# 13. 최종 데이터 저장
# ============================================================

df.drop(
    columns=["_value_range"],
    inplace=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 최종 통계
# ============================================================

final_count = len(df)

total_deleted = (
    original_count
    - final_count
)


print()
print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(
    f"원본 행 수                    : "
    f"{original_count:,}"
)

print(
    f"최종 행 수                    : "
    f"{final_count:,}"
)

print(
    f"총 삭제 행 수                 : "
    f"{total_deleted:,}"
)

print(
    f"총 삭제 비율                  : "
    f"{total_deleted / original_count * 100:.4f}%"
)

print()

print(
    f"최종 데이터 저장              : "
    f"{OUTPUT_PATH}"
)

print(
    f"분포 결과 저장                : "
    f"{DISTRIBUTION_PATH}"
)

print()

print("=" * 100)
print("DONE")
print("=" * 100)

