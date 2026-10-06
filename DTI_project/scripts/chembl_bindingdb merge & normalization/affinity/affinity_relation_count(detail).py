
import os
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\bindingdb_chembl_merging"
FILE_NAME = "merged_chembl_binding_final.csv"

INPUT_PATH = os.path.join(DATA_DIR, FILE_NAME)

OUTPUT_EQ_MIXED = os.path.join(
    DATA_DIR,
    "merged_chembl_binding_final.csv"
)


# ============================================================
# 2. 분석 대상 컬럼
# ============================================================

GROUP_COLS = [
    "smiles",
    "affinity_type",
    "uniprot_id",
    "target_name"
]

RELATION_COL = "affinity_relation"


# ============================================================
# 3. CSV 읽기
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
# 4. 기본 QC
# ============================================================

required_cols = GROUP_COLS + [RELATION_COL]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_cols}"
    )

# affinity_relation 결측 제거
df = df.dropna(subset=[RELATION_COL]).copy()

# relation 문자열 정리
df[RELATION_COL] = (
    df[RELATION_COL]
    .astype(str)
    .str.strip()
)

print(f"affinity_relation 결측 제거 후 : {len(df):,}")
print()


# ============================================================
# 5. 동일 DTI 그룹별 affinity_relation 종류 확인
#
# 동일 기준:
# SMILES + affinity_type + uniprot_id + target_name
# ============================================================

grouped = (
    df.groupby(GROUP_COLS, dropna=False)[RELATION_COL]
    .agg(
        relation_set=lambda x: frozenset(x)
    )
    .reset_index()
)

# relation 개수
grouped["relation_count"] = grouped["relation_set"].apply(len)

# 보기 편하게 문자열화
grouped["relation_set_str"] = grouped["relation_set"].apply(
    lambda x: ",".join(sorted(x))
)


# ============================================================
# 6. affinity_relation이 =를 가지면서
#    < 또는 >도 가지는 그룹
#
#    예:
#    =,<
#    =,>
#    =,<,>
# ============================================================

eq_mixed_mask = (
    grouped["relation_set"].apply(lambda x: "=" in x)
    &
    grouped["relation_set"].apply(
        lambda x: ("<" in x) or (">" in x)
    )
)

eq_mixed_groups = grouped[eq_mixed_mask].copy()


print("=" * 80)
print("1. '=' + ('<' 또는 '>')가 함께 존재하는 그룹")
print("=" * 80)

print(
    f"그룹 수 : {len(eq_mixed_groups):,}"
)

print("\nrelation 조합별 그룹 수:")
print(
    eq_mixed_groups["relation_set_str"]
    .value_counts()
    .sort_index()
)

print()


# ============================================================
# 7. 실제 원본 행 조회
#
# 위 그룹에 해당하는 모든 원본 데이터 출력
# ============================================================

if len(eq_mixed_groups) > 0:

    eq_mixed_keys = eq_mixed_groups[GROUP_COLS].copy()

    eq_mixed_rows = df.merge(
        eq_mixed_keys,
        on=GROUP_COLS,
        how="inner"
    )

    print("해당 그룹의 실제 행 수:")
    print(f"{len(eq_mixed_rows):,}")

    # 결과 저장
    eq_mixed_rows.to_csv(
        OUTPUT_EQ_MIXED,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n조회 결과 저장:")
    print(OUTPUT_EQ_MIXED)

else:
    eq_mixed_rows = pd.DataFrame()
    print("해당 그룹이 없습니다.")

print()


# ============================================================
# 8. affinity_relation이 정확히 1종류만 존재하는 그룹
#
# 예:
# =
# <
# >
# ============================================================

single_relation_groups = grouped[
    grouped["relation_count"] == 1
].copy()


print("=" * 80)
print("2. affinity_relation이 정확히 1종류만 존재하는 그룹")
print("=" * 80)

print(
    f"전체 그룹 수 : {len(single_relation_groups):,}"
)

print("\nrelation별 그룹 수:")

single_relation_counts = (
    single_relation_groups["relation_set_str"]
    .value_counts()
    .reindex(["=", "<", ">"])
    .fillna(0)
    .astype(int)
)

print(single_relation_counts)

print()


# ============================================================
# 9. affinity_relation이 < 또는 >만 가지는 그룹
#
# 즉 '='가 전혀 없는 그룹
#
# 가능한 경우:
# <
# >
# <,>
# ============================================================

only_comparison_mask = grouped["relation_set"].apply(
    lambda x: (
        len(x) > 0
        and "=" not in x
        and x.issubset({"<", ">"})
    )
)

only_comparison_groups = grouped[
    only_comparison_mask
].copy()


print("=" * 80)
print("3. affinity_relation이 '<' 또는 '>'만 존재하는 그룹")
print("=" * 80)

print(
    f"전체 그룹 수 : {len(only_comparison_groups):,}"
)

print("\nrelation 조합별 그룹 수:")

comparison_counts = (
    only_comparison_groups["relation_set_str"]
    .value_counts()
    .reindex(["<", ">", "<,>"])
    .fillna(0)
    .astype(int)
)

print(comparison_counts)

print()


# ============================================================
# 10. 최종 요약
# ============================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"전체 원본 행 수                  : {len(df):,}")
print(f"전체 고유 그룹 수                : {len(grouped):,}")
print()

print(
    f"[= + < 또는 >] 그룹 수           : "
    f"{len(eq_mixed_groups):,}"
)

print(
    f"[relation 1종류만] 그룹 수        : "
    f"{len(single_relation_groups):,}"
)

print(
    f"[< 또는 >만 존재] 그룹 수         : "
    f"{len(only_comparison_groups):,}"
)

print()

print("relation 1종류만:")
for relation in ["=", "<", ">"]:
    count = single_relation_counts.get(relation, 0)
    print(f"  {relation:>2} only : {count:,} 그룹")

print()

print("< 또는 >만 존재:")
for relation in ["<", ">", "<,>"]:
    count = comparison_counts.get(relation, 0)
    print(f"  {relation:>3} : {count:,} 그룹")

print()

print("=" * 80)
print("DONE")
print("=" * 80)

