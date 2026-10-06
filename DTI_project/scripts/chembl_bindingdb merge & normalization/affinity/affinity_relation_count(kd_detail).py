
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


# ============================================================
# 2. 동일 그룹 판단 기준
#
# 아래 4개 컬럼의 값이 모두 동일하면 같은 그룹
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
    .str.lower()
)

# IC50 / Ki / Kd 표기 통일
df["affinity_type"] = (
    df["affinity_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# 숫자 변환
df[VALUE_COL] = pd.to_numeric(
    df[VALUE_COL],
    errors="coerce"
)


print("=" * 100)
print("BASIC QC")
print("=" * 100)

print("affinity_type 분포")
print(
    df["affinity_type"].value_counts(
        dropna=False
    )
)

print()

print("affinity_relation 분포")
print(
    df[RELATION_COL].value_counts(
        dropna=False
    )
)

print()

print(
    f"affinity_value_mean 결측 : "
    f"{df[VALUE_COL].isna().sum():,}"
)

print()


# ============================================================
# 6. 고유 그룹 생성
#
# SMILES + affinity_type + uniprot_id + target_name
# 기준으로 동일 그룹을 하나의 그룹으로 취급
#
# DEDUP8 데이터에서 혹시 동일 그룹이 여러 행에 존재하더라도
# 그룹 단위로 중복을 제거하여 분석
# ============================================================

group_df = (
    df[
        GROUP_COLS
        + [RELATION_COL, VALUE_COL]
    ]
    .drop_duplicates(
        subset=GROUP_COLS
    )
    .copy()
)


print("=" * 100)
print("GROUP QC")
print("=" * 100)

print(
    f"원본 행 수        : {len(df):,}"
)

print(
    f"고유 그룹 수       : {len(group_df):,}"
)

print()


# ============================================================
# 7. 1번
#
# affinity_type별
# affinity_relation (= / < / >) 분포
#
# 고유 그룹 기준
# ============================================================

print("=" * 100)
print("1. affinity_type별 affinity_relation 분포")
print("=" * 100)


relation_distribution = (
    group_df
    .groupby(
        [
            "affinity_type",
            RELATION_COL
        ],
        dropna=False
    )
    .size()
    .unstack(
        fill_value=0
    )
)


# 컬럼 순서 정리
for relation in ["=", "<", ">"]:
    if relation not in relation_distribution.columns:
        relation_distribution[relation] = 0

relation_distribution = relation_distribution[
    ["=", "<", ">"]
]


# 전체 합계
relation_distribution["TOTAL"] = (
    relation_distribution["="]
    + relation_distribution["<"]
    + relation_distribution[">"]
)


# 비율
relation_percentage = (
    relation_distribution[
        ["=", "<", ">"]
    ]
    .div(
        relation_distribution["TOTAL"],
        axis=0
    )
    * 100
)


print("\n[그룹 개수]")
print(
    relation_distribution
)

print("\n[그룹 비율 %]")
print(
    relation_percentage.round(4)
)

print()


# ============================================================
# 8. KD 데이터만 추출
# ============================================================

kd_df = group_df[
    group_df["affinity_type"] == "kd"
].copy()


print("=" * 100)
print("KD DATA")
print("=" * 100)

print(
    f"KD 전체 그룹 수 : {len(kd_df):,}"
)

print()

print("KD affinity_relation 분포")

kd_relation_distribution = (
    kd_df[RELATION_COL]
    .value_counts()
    .reindex(
        ["=", "<", ">"],
        fill_value=0
    )
)

print(
    kd_relation_distribution
)

print()


# ============================================================
# 9. affinity_value 분포를 계산하는 함수
#
# 반환:
# - 전체
# - 1000 이상
# - 1000 미만
# - 500 미만
# - 100 미만
# - 50 미만
# - 10 미만
#
# 모두 cumulative count
# ============================================================

def print_value_distribution(
    data,
    relation
):

    print("=" * 100)
    print(
        f"KD / affinity_relation = '{relation}' "
        f"affinity_value_mean 분포"
    )
    print("=" * 100)

    subset = data[
        data[RELATION_COL] == relation
    ].copy()

    # affinity_value_mean이 없는 데이터
    missing_value_count = (
        subset[VALUE_COL]
        .isna()
        .sum()
    )

    # 실제 숫자가 있는 데이터
    valid = subset[
        subset[VALUE_COL].notna()
    ].copy()

    total = len(valid)

    # --------------------------------------------------------
    # 요청한 기준
    # --------------------------------------------------------

    count_1000_or_more = (
        valid[VALUE_COL] >= 1000
    ).sum()

    count_under_1000 = (
        valid[VALUE_COL] < 1000
    ).sum()

    count_under_500 = (
        valid[VALUE_COL] < 500
    ).sum()

    count_under_100 = (
        valid[VALUE_COL] < 100
    ).sum()

    count_under_50 = (
        valid[VALUE_COL] < 50
    ).sum()

    count_under_10 = (
        valid[VALUE_COL] < 10
    ).sum()


    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------

    print(
        f"전체 그룹 수                         : "
        f"{total:,}"
    )

    print(
        f"affinity_value_mean 결측             : "
        f"{missing_value_count:,}"
    )

    print()

    print(
        f"1000 이상                            : "
        f"{count_1000_or_more:,}"
    )

    print(
        f"1000 미만                            : "
        f"{count_under_1000:,}"
    )

    print(
        f"  └─ 500 미만                        : "
        f"{count_under_500:,}"
    )

    print(
        f"      └─ 100 미만                    : "
        f"{count_under_100:,}"
    )

    print(
        f"          └─ 50 미만                 : "
        f"{count_under_50:,}"
    )

    print(
        f"              └─ 10 미만             : "
        f"{count_under_10:,}"
    )

    print()

    # --------------------------------------------------------
    # 비율
    # --------------------------------------------------------

    if total > 0:

        print("[비율]")

        print(
            f"1000 이상 : "
            f"{count_1000_or_more:,} "
            f"({count_1000_or_more / total * 100:.4f}%)"
        )

        print(
            f"1000 미만 : "
            f"{count_under_1000:,} "
            f"({count_under_1000 / total * 100:.4f}%)"
        )

        print(
            f"500 미만  : "
            f"{count_under_500:,} "
            f"({count_under_500 / total * 100:.4f}%)"
        )

        print(
            f"100 미만  : "
            f"{count_under_100:,} "
            f"({count_under_100 / total * 100:.4f}%)"
        )

        print(
            f"50 미만   : "
            f"{count_under_50:,} "
            f"({count_under_50 / total * 100:.4f}%)"
        )

        print(
            f"10 미만   : "
            f"{count_under_10:,} "
            f"({count_under_10 / total * 100:.4f}%)"
        )

    print()

    return {
        "relation": relation,
        "total": total,
        "missing": missing_value_count,
        ">=1000": count_1000_or_more,
        "<1000": count_under_1000,
        "<500": count_under_500,
        "<100": count_under_100,
        "<50": count_under_50,
        "<10": count_under_10
    }


# ============================================================
# 10. 2번
#
# KD + '='
# ============================================================

result_equal = print_value_distribution(
    kd_df,
    "="
)


# ============================================================
# 11. 3번
#
# KD + '>'
# ============================================================

result_greater = print_value_distribution(
    kd_df,
    ">"
)


# ============================================================
# 12. 4번
#
# KD + '<'
# ============================================================

result_less = print_value_distribution(
    kd_df,
    "<"
)


# ============================================================
# 13. 결과를 DataFrame으로 정리
# ============================================================

result_df = pd.DataFrame(
    [
        result_equal,
        result_greater,
        result_less
    ]
)

print("=" * 100)
print("KD VALUE DISTRIBUTION SUMMARY")
print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)

print()


# ============================================================
# 14. 결과 CSV 저장
# ============================================================

OUTPUT_FILE = "DEDUP8_KD_affinity_value_distribution.csv"

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    OUTPUT_FILE
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"결과 저장 : {OUTPUT_PATH}"
)

print()


# ============================================================
# 15. 완료
# ============================================================

print("=" * 100)
print("DONE")
print("=" * 100)

