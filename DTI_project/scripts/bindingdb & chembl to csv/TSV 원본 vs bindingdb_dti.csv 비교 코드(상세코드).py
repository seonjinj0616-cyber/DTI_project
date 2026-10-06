import pandas as pd
import numpy as np
import os

# ============================================================
# 0. 경로
# ============================================================

TSV_PATH = r"C:\workspace\python\project_personal\DTI_project\data\bindingdb\BindingDB_All.tsv"
CSV_PATH = r"C:\workspace\python\project_personal\DTI_project\data\output\bindingdb_dti.csv"

OUTPUT_DIR = r"C:\workspace\python\project_personal\DTI_project\data\output\bindingdb_qc"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. BindingDB 원본 TSV → 현재 parser와 동일하게 재구성
# ============================================================

AFFINITY_COLS = [
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)"
]

USECOLS = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "ChEMBL ID of Ligand",
    "BindingDB MonomerID",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
] + AFFINITY_COLS


print("=" * 90)
print("1. BindingDB ORIGINAL TSV 재구성")
print("=" * 90)

tsv_parts = []

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        TSV_PATH,
        sep="\t",
        dtype=str,
        usecols=USECOLS,
        chunksize=200_000,
        low_memory=False
    )
):

    print(f"Chunk {chunk_idx}: {len(chunk):,}")

    # --------------------------------------------------------
    # Human target만
    # --------------------------------------------------------

    organism_col = "Target Source Organism According to Curator or DataSource"

    chunk = chunk[
        chunk[organism_col]
        .fillna("")
        .str.contains("Homo sapiens", case=False, na=False)
    ].copy()

    if len(chunk) == 0:
        continue

    # --------------------------------------------------------
    # affinity long format
    # --------------------------------------------------------

    id_cols = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "ChEMBL ID of Ligand",
        "BindingDB MonomerID",
        "Target Name",
        organism_col
    ]

    melted = chunk.melt(
        id_vars=id_cols,
        value_vars=AFFINITY_COLS,
        var_name="affinity_type",
        value_name="affinity_str"
    )

    # 빈 값 제거
    melted["affinity_str"] = (
        melted["affinity_str"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    melted = melted[melted["affinity_str"] != ""].copy()

    if len(melted) == 0:
        continue

    # --------------------------------------------------------
    # Relation
    #
    # 현재 BindingDB 실제 데이터에서는
    # = / > / < 만 존재한다고 QC 확인됨
    # --------------------------------------------------------

    melted["affinity_relation"] = "="

    melted.loc[
        melted["affinity_str"].str.startswith(">"),
        "affinity_relation"
    ] = ">"

    melted.loc[
        melted["affinity_str"].str.startswith("<"),
        "affinity_relation"
    ] = "<"

    # --------------------------------------------------------
    # 숫자 부분
    # --------------------------------------------------------

    clean_vals = (
        melted["affinity_str"]
        .str.replace(">", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.strip()
    )

    melted["affinity_value"] = pd.to_numeric(
        clean_vals,
        errors="coerce"
    )

    melted = melted.dropna(
        subset=["affinity_value"]
    ).copy()

    melted = melted[
        melted["affinity_value"] > 0
    ].copy()

    if len(melted) == 0:
        continue

    # --------------------------------------------------------
    # Unified schema
    # --------------------------------------------------------

    result = pd.DataFrame({
        "source": "BindingDB",

        "source_record_id":
            melted["BindingDB Reactant_set_id"],

        "chembl_id":
            melted["ChEMBL ID of Ligand"],

        "bindingdb_id":
            melted["BindingDB MonomerID"],

        "smiles":
            melted["Ligand SMILES"],

        "target_name":
            melted["Target Name"],

        "organism":
            melted[organism_col],

        "affinity_type":
            melted["affinity_type"],

        "affinity_relation":
            melted["affinity_relation"],

        "affinity_value":
            melted["affinity_value"],
    })

    tsv_parts.append(result)


tsv_reconstructed = pd.concat(
    tsv_parts,
    ignore_index=True
)

print()
print(f"TSV reconstructed rows: {len(tsv_reconstructed):,}")


# ============================================================
# 2. 현재 CSV 읽기
# ============================================================

print()
print("=" * 90)
print("2. 현재 bindingdb_dti.csv 읽기")
print("=" * 90)

csv = pd.read_csv(
    CSV_PATH,
    dtype=str,
    low_memory=False
)

print(f"CSV rows: {len(csv):,}")
print(f"CSV columns: {len(csv.columns)}")


# ============================================================
# 3. 숫자형 변환
# ============================================================

tsv_reconstructed["affinity_value"] = pd.to_numeric(
    tsv_reconstructed["affinity_value"],
    errors="coerce"
)

csv["affinity_value"] = pd.to_numeric(
    csv["affinity_value"],
    errors="coerce"
)


# ============================================================
# 4. 비교용 key
#
# 기존에는
# source_record_id + bindingdb_id + affinity_type
# 를 사용했는데,
#
# BindingDB에서는 같은 Reactant_set_id / MonomerID가
# 여러 measurement에 걸릴 수 있으므로
# target_name + value + relation까지 포함한
# measurement-level key를 추가한다.
# ============================================================

KEY_COLS = [
    "source_record_id",
    "bindingdb_id",
    "affinity_type",
    "target_name",
    "smiles",
    "chembl_id",
    "affinity_relation",
    "affinity_value"
]


# ============================================================
# 5. 문자열 정규화
# ============================================================

def normalize_str(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


for col in [
    "source_record_id",
    "bindingdb_id",
    "affinity_type",
    "target_name",
    "smiles",
    "chembl_id",
    "affinity_relation"
]:
    tsv_reconstructed[col] = normalize_str(
        tsv_reconstructed[col]
    )

    csv[col] = normalize_str(
        csv[col]
    )


# ============================================================
# 6. measurement key 생성
# ============================================================

def make_key(df):

    value = (
        df["affinity_value"]
        .map(lambda x: "" if pd.isna(x) else f"{float(x):.12g}")
    )

    return (
        df["source_record_id"] + "||" +
        df["bindingdb_id"] + "||" +
        df["affinity_type"] + "||" +
        df["target_name"] + "||" +
        df["smiles"] + "||" +
        df["chembl_id"] + "||" +
        df["affinity_relation"] + "||" +
        value
    )


tsv_reconstructed["measurement_key"] = make_key(
    tsv_reconstructed
)

csv["measurement_key"] = make_key(
    csv
)


# ============================================================
# 7. Key 중복 확인
# ============================================================

print()
print("=" * 90)
print("3. Measurement key 중복 QC")
print("=" * 90)

tsv_dup = (
    tsv_reconstructed["measurement_key"]
    .duplicated(keep=False)
    .sum()
)

csv_dup = (
    csv["measurement_key"]
    .duplicated(keep=False)
    .sum()
)

print(f"TSV duplicated measurement rows: {tsv_dup:,}")
print(f"CSV duplicated measurement rows: {csv_dup:,}")


# ============================================================
# 8. set 비교
# ============================================================

tsv_keys = set(
    tsv_reconstructed["measurement_key"]
)

csv_keys = set(
    csv["measurement_key"]
)

both_keys = tsv_keys & csv_keys
tsv_only_keys = tsv_keys - csv_keys
csv_only_keys = csv_keys - tsv_keys

print()
print("=" * 90)
print("4. Measurement key 비교")
print("=" * 90)

print(f"TSV unique keys : {len(tsv_keys):,}")
print(f"CSV unique keys : {len(csv_keys):,}")
print(f"BOTH            : {len(both_keys):,}")
print(f"TSV ONLY        : {len(tsv_only_keys):,}")
print(f"CSV ONLY        : {len(csv_only_keys):,}")


# ============================================================
# 9. left_only / right_only dataframe
# ============================================================

tsv_only = tsv_reconstructed[
    tsv_reconstructed["measurement_key"].isin(tsv_only_keys)
].copy()

csv_only = csv[
    csv["measurement_key"].isin(csv_only_keys)
].copy()


# ============================================================
# 10. TSV ONLY 원인 분석
# ============================================================

print()
print("=" * 90)
print("5. TSV ONLY 분석")
print("=" * 90)

print(f"TSV ONLY rows: {len(tsv_only):,}")


# ------------------------------------------------------------
# 같은 source_record_id가 CSV에 존재하는가?
# ------------------------------------------------------------

csv_source_ids = set(
    csv["source_record_id"]
)

tsv_only["source_record_exists_in_csv"] = (
    tsv_only["source_record_id"]
    .isin(csv_source_ids)
)

print()
print(
    "TSV ONLY 중 source_record_id가 CSV에도 존재:",
    tsv_only["source_record_exists_in_csv"].sum()
)

print(
    "TSV ONLY 중 source_record_id 자체가 CSV에 없음:",
    (~tsv_only["source_record_exists_in_csv"]).sum()
)


# ============================================================
# 11. CSV ONLY 원인 분석
# ============================================================

print()
print("=" * 90)
print("6. CSV ONLY 분석")
print("=" * 90)

print(f"CSV ONLY rows: {len(csv_only):,}")

tsv_source_ids = set(
    tsv_reconstructed["source_record_id"]
)

csv_only["source_record_exists_in_tsv"] = (
    csv_only["source_record_id"]
    .isin(tsv_source_ids)
)

print()
print(
    "CSV ONLY 중 source_record_id가 TSV에도 존재:",
    csv_only["source_record_exists_in_tsv"].sum()
)

print(
    "CSV ONLY 중 source_record_id 자체가 TSV에 없음:",
    (~csv_only["source_record_exists_in_tsv"]).sum()
)


# ============================================================
# 12. source_record_id 기준으로 다시 비교
#
# 핵심:
# 같은 Reactant_set_id가 존재하지만
# target / measurement 조합이 다른지 확인
# ============================================================

print()
print("=" * 90)
print("7. source_record_id 기준 분석")
print("=" * 90)

tsv_counts = (
    tsv_reconstructed
    .groupby("source_record_id")
    .size()
    .rename("tsv_rows")
)

csv_counts = (
    csv
    .groupby("source_record_id")
    .size()
    .rename("csv_rows")
)

source_compare = pd.concat(
    [tsv_counts, csv_counts],
    axis=1
).fillna(0)

source_compare["row_diff"] = (
    source_compare["tsv_rows"]
    - source_compare["csv_rows"]
)

print(
    "동일 source_record_id 존재:",
    (
        (source_compare["tsv_rows"] > 0) &
        (source_compare["csv_rows"] > 0)
    ).sum()
)

print(
    "TSV에만 존재:",
    (
        (source_compare["tsv_rows"] > 0) &
        (source_compare["csv_rows"] == 0)
    ).sum()
)

print(
    "CSV에만 존재:",
    (
        (source_compare["tsv_rows"] == 0) &
        (source_compare["csv_rows"] > 0)
    ).sum()
)

print(
    "row count가 다른 source_record_id:",
    (source_compare["row_diff"] != 0).sum()
)


# ============================================================
# 13. TSV ONLY를 source_record_id별로 가장 많이 발생한 순서
# ============================================================

print()
print("=" * 90)
print("8. TSV ONLY - source_record_id별 빈도")
print("=" * 90)

top_tsv_only = (
    tsv_only
    .groupby("source_record_id")
    .size()
    .sort_values(ascending=False)
    .head(30)
)

print(top_tsv_only)


# ============================================================
# 14. CSV ONLY 빈도
# ============================================================

print()
print("=" * 90)
print("9. CSV ONLY - source_record_id별 빈도")
print("=" * 90)

top_csv_only = (
    csv_only
    .groupby("source_record_id")
    .size()
    .sort_values(ascending=False)
    .head(30)
)

print(top_csv_only)


# ============================================================
# 15. source_record_id가 동일하지만 measurement가 다른 경우
# ============================================================

print()
print("=" * 90)
print("10. 동일 source_record_id 내부 measurement 차이 분석")
print("=" * 90)

common_source_ids = (
    set(tsv_reconstructed["source_record_id"])
    &
    set(csv["source_record_id"])
)

tsv_common_source = tsv_reconstructed[
    tsv_reconstructed["source_record_id"]
    .isin(common_source_ids)
]

csv_common_source = csv[
    csv["source_record_id"]
    .isin(common_source_ids)
]


# ------------------------------------------------------------
# target_name 기준
# ------------------------------------------------------------

tsv_targets = (
    tsv_common_source
    .groupby("source_record_id")["target_name"]
    .nunique()
    .rename("tsv_target_count")
)

csv_targets = (
    csv_common_source
    .groupby("source_record_id")["target_name"]
    .nunique()
    .rename("csv_target_count")
)

target_compare = pd.concat(
    [tsv_targets, csv_targets],
    axis=1
).fillna(0)

target_compare["target_count_diff"] = (
    target_compare["tsv_target_count"]
    -
    target_compare["csv_target_count"]
)

print(
    "target 개수가 다른 source_record_id:",
    (target_compare["target_count_diff"] != 0).sum()
)


# ============================================================
# 16. 결과 저장
# ============================================================

print()
print("=" * 90)
print("11. QC 결과 저장")
print("=" * 90)

tsv_only.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tsv_only_measurements.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

csv_only.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "csv_only_measurements.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

source_compare.reset_index().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "source_record_id_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

target_compare.reset_index().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "source_record_target_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 17. 최종 요약
# ============================================================

print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

print(f"TSV reconstructed rows : {len(tsv_reconstructed):,}")
print(f"CSV rows               : {len(csv):,}")
print(f"TSV unique measurements: {len(tsv_keys):,}")
print(f"CSV unique measurements: {len(csv_keys):,}")
print(f"Matched                : {len(both_keys):,}")
print(f"TSV ONLY                : {len(tsv_only_keys):,}")
print(f"CSV ONLY                : {len(csv_only_keys):,}")

print()
print("결과 파일:")
print(
    os.path.join(
        OUTPUT_DIR,
        "tsv_only_measurements.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "csv_only_measurements.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "source_record_id_comparison.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "source_record_target_comparison.csv"
    )
)