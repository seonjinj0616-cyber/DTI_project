import os
import pandas as pd
from collections import Counter


# ============================================================
# 경로 설정
# ============================================================

# 입력 데이터
INPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\DEDUP8_affinity_relation_mean.csv"
)

# 현재 작업 폴더
SCRIPT_DIR = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\scripts\csv merge & normalization\affinity"
)


# ============================================================
# Affinity Relation QC
# ============================================================

counter = Counter()
total_rows = 0

print("=" * 80)
print("Affinity Relation QC")
print("=" * 80)
print(f"Input : {INPUT_FILE}")
print()


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=["affinity_relation"],
        chunksize=200_000,
        low_memory=False
    ),
    start=1
):

    values = chunk["affinity_relation"].fillna("MISSING")

    counter.update(values)
    total_rows += len(chunk)

    print(
        f"Chunk {chunk_idx}: "
        f"{len(chunk):,} rows"
    )


# ============================================================
# 결과 출력
# ============================================================

print()
print("=" * 80)
print("Affinity Relation 전체 분포")
print("=" * 80)

for relation, count in counter.most_common():

    percentage = count / total_rows * 100

    print(
        f"{relation:>10} : "
        f"{count:>12,} "
        f"({percentage:6.2f}%)"
    )

print("-" * 80)

print(
    f"{'TOTAL':>10} : "
    f"{total_rows:>12,} "
    f"(100.00%)"
)

print("=" * 80)