import os
import pandas as pd

# ============================================================
# 경로 설정
# ============================================================
INPUT_PATH = "../../../data/output/merged_dti_DEDUP7.csv"
DUPLICATE_SUMMARY_PATH = (
    "../../../data/output/dedup8_duplicate_type_summary.csv"
)
AFFINITY_MEAN_PATH = (
    "../../../data/output/dedup8_affinity_relation_mean.csv"
)

CHUNKSIZE = 200_000

# 출력 디렉토리 생성
os.makedirs(os.path.dirname(DUPLICATE_SUMMARY_PATH), exist_ok=True)
os.makedirs(os.path.dirname(AFFINITY_MEAN_PATH), exist_ok=True)

# ============================================================
# 기준 컬럼
# ============================================================
BASE_KEYS = [
    "smiles",
    "uniprot_id",
    "target_name",
    "affinity_type",
]
RELATION_KEY = "affinity_relation"
GROUP_KEYS = BASE_KEYS + [RELATION_KEY]

# ============================================================
# 집계용 변수
# ============================================================
base_group_counts = {}

affinity_sum = {}
affinity_count = {}
affinity_min = {}
affinity_max = {}

total_rows = 0
valid_affinity_rows = 0
missing_affinity_rows = 0

print("=" * 90)
print("약물-단백질 중복 통계 분석 및 affinity_relation별 평균 계산")
print("=" * 90)

# ============================================================
# CHUNK Processing (Pass 1)
# ============================================================
for chunk_idx, chunk in enumerate(
    pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, low_memory=False)
):
    print(f"\nChunk {chunk_idx + 1} 처리 중...")
    total_rows += len(chunk)

    # 1. Key 컬럼 문자열 정규화 (Dictionary Hash 오류 방지)
    for col in GROUP_KEYS:
        chunk[col] = chunk[col].fillna("").astype(str).str.strip()

    # 2. affinity_value 수치 변환
    chunk["affinity_value_numeric"] = pd.to_numeric(
        chunk["affinity_value"], errors="coerce"
    )

    valid_chunk_count = int(chunk["affinity_value_numeric"].notna().sum())
    missing_chunk_count = int(chunk["affinity_value_numeric"].isna().sum())

    valid_affinity_rows += valid_chunk_count
    missing_affinity_rows += missing_chunk_count

    # 3. BASE_KEYS 기준 중복 수 집계
    base_counts = chunk.groupby(BASE_KEYS, dropna=False).size()
    for key, count in base_counts.items():
        base_group_counts[key] = base_group_counts.get(key, 0) + int(count)

    # 4. GROUP_KEYS (BASE_KEYS + affinity_relation) 기준 수치 집계
    valid_chunk = chunk[chunk["affinity_value_numeric"].notna()].copy()
    if not valid_chunk.empty:
        grouped = (
            valid_chunk.groupby(GROUP_KEYS, dropna=False)[
                "affinity_value_numeric"
            ].agg(["sum", "count", "min", "max"])
        )

        for key, row in grouped.iterrows():
            affinity_sum[key] = affinity_sum.get(key, 0.0) + float(row["sum"])
            affinity_count[key] = affinity_count.get(key, 0) + int(row["count"])

            if key not in affinity_min:
                affinity_min[key] = float(row["min"])
            else:
                affinity_min[key] = min(affinity_min[key], float(row["min"]))

            if key not in affinity_max:
                affinity_max[key] = float(row["max"])
            else:
                affinity_max[key] = max(affinity_max[key], float(row["max"]))

    print(f"  Chunk {chunk_idx + 1} 완료 | 행 수: {len(chunk):,} | 유효 수치: {valid_chunk_count:,}")

# ============================================================
# 1. BASE_KEYS 중복 분포 요약
# ============================================================
print("\n" + "=" * 90)
print("1. 약물-단백질 중복 통계 생성")
print("=" * 90)

duplicate_summary = []
for key, count in base_group_counts.items():
    record = dict(zip(BASE_KEYS, key))
    record["row_count"] = count

    if count == 1:
        record["duplicate_type"] = "1"
    elif count == 2:
        record["duplicate_type"] = "2"
    elif count == 3:
        record["duplicate_type"] = "3"
    elif count == 4:
        record["duplicate_type"] = "4"
    elif count == 5:
        record["duplicate_type"] = "5"
    else:
        record["duplicate_type"] = "6+"

    duplicate_summary.append(record)

duplicate_summary_df = pd.DataFrame(duplicate_summary)

# 중복 종류별 총 그룹 수 및 전체 행 수 집계
duplicate_type_count = (
    duplicate_summary_df.groupby("duplicate_type")
    .agg(group_count=("row_count", "size"), total_rows=("row_count", "sum"))
    .reset_index()
)

duplicate_order = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6+": 6}
duplicate_type_count["_order"] = duplicate_type_count["duplicate_type"].map(duplicate_order)
duplicate_type_count = duplicate_type_count.sort_values("_order").drop(columns="_order")

duplicate_summary_df.to_csv(
    DUPLICATE_SUMMARY_PATH, index=False, encoding="utf-8-sig"
)

# ============================================================
# 2. affinity_relation별 독립 평균 계산
# ============================================================
print("\n" + "=" * 90)
print("2. affinity_relation별 affinity_value 평균 계산")
print("=" * 90)

affinity_mean_records = []
for key in affinity_sum.keys():
    record = dict(zip(GROUP_KEYS, key))
    count = affinity_count[key]
    total = affinity_sum[key]

    record["measurement_count"] = count
    record["affinity_value_mean"] = total / count
    record["affinity_value_min"] = affinity_min[key]
    record["affinity_value_max"] = affinity_max[key]

    affinity_mean_records.append(record)

affinity_mean_df = pd.DataFrame(affinity_mean_records)

# BASE 그룹별 부등호 종수 및 총 측정 횟수 병합
if not affinity_mean_df.empty:
    relation_count_df = (
        affinity_mean_df.groupby(BASE_KEYS, dropna=False)
        .agg(
            relation_type_count=(RELATION_KEY, "nunique"),
            total_measurement_count=("measurement_count", "sum"),
        )
        .reset_index()
    )

    affinity_mean_df = affinity_mean_df.merge(
        relation_count_df, on=BASE_KEYS, how="left"
    )

affinity_mean_df.to_csv(
    AFFINITY_MEAN_PATH, index=False, encoding="utf-8-sig"
)

# ============================================================
# 최종 검산 및 결과 출력
# ============================================================
print("\n" + "=" * 90)
print("분석 완료")
print("=" * 90)
print(f"전체 원본 행 수          : {total_rows:,}")
print(f"Valid Numeric Affinity    : {valid_affinity_rows:,}")
print(f"Missing Numeric Affinity  : {missing_affinity_rows:,}")
print(f"\nBASE 기준 Unique Group    : {len(duplicate_summary_df):,}")
print("\n[중복 종류별 요약]")
print(duplicate_type_count.to_string(index=False))
print(f"\nAffinity Relation별 Group  : {len(affinity_mean_df):,}")
print("\n[저장 완료 파일]")
print(f"1. {DUPLICATE_SUMMARY_PATH}")
print(f"2. {AFFINITY_MEAN_PATH}")