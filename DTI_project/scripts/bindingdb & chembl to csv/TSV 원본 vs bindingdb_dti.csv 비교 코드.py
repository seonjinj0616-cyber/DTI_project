# qc_bindingdb_tsv_vs_csv.py

import pandas as pd
import numpy as np
from pathlib import Path

TSV_PATH = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\bindingdb\BindingDB_All.tsv"
)

CSV_PATH = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\output\bindingdb_dti.csv"
)

CHUNK_SIZE = 200_000

print("=" * 100)
print("BindingDB ORIGINAL TSV vs PARSED CSV 비교 QC")
print("=" * 100)

# ================================================================
# 1. CSV 읽기
# ================================================================

csv = pd.read_csv(
    CSV_PATH,
    dtype=str,
    low_memory=False
)

print(f"\nCSV row 수: {len(csv):,}")

# 숫자 변환
csv["affinity_value"] = pd.to_numeric(
    csv["affinity_value"],
    errors="coerce"
)

# ================================================================
# 2. CSV source_record_id 기준 확인
# ================================================================

print("\nCSV source_record_id:")
print(f"  unique: {csv['source_record_id'].nunique():,}")
print(f"  total : {len(csv):,}")


# ================================================================
# 3. TSV에 필요한 컬럼 확인
# ================================================================

header = pd.read_csv(
    TSV_PATH,
    sep="\t",
    nrows=0
)

required_columns = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "ChEMBL ID of Ligand",
    "BindingDB MonomerID",
    "Target Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
]

missing = [
    col for col in required_columns
    if col not in header.columns
]

if missing:
    raise ValueError(
        "TSV에서 필요한 컬럼이 없습니다:\n"
        + "\n".join(missing)
    )

print("\nTSV 필요한 컬럼 확인 완료")


# ================================================================
# 4. TSV → 현재 parser와 동일한 방식으로 affinity 생성
# ================================================================

print("\n원본 TSV 읽는 중...")

raw_rows = []

usecols = required_columns

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        TSV_PATH,
        sep="\t",
        usecols=usecols,
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
):

    # ------------------------------------------------------------
    # 현재 parser의 melt 구조 재현
    # ------------------------------------------------------------

    id_columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "ChEMBL ID of Ligand",
        "BindingDB MonomerID",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
    ]

    affinity_columns = [
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
    ]

    melted = chunk.melt(
        id_vars=id_columns,
        value_vars=affinity_columns,
        var_name="affinity_column",
        value_name="affinity_str"
    )

    melted = melted.dropna(
        subset=["affinity_str"]
    )

    melted["affinity_str"] = (
        melted["affinity_str"]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------
    # 현재 코드와 동일
    # ------------------------------------------------------------

    melted["affinity_relation"] = "="

    melted.loc[
        melted["affinity_str"].str.startswith(">"),
        "affinity_relation"
    ] = ">"

    melted.loc[
        melted["affinity_str"].str.startswith("<"),
        "affinity_relation"
    ] = "<"

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
    )

    melted = melted[
        melted["affinity_value"] > 0
    ]

    if len(melted) == 0:
        continue

    # affinity type
    melted["affinity_type"] = (
        melted["affinity_column"]
        .str.replace(" (nM)", "", regex=False)
    )

    # 비교에 필요한 컬럼
    result = melted[[
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "ChEMBL ID of Ligand",
        "BindingDB MonomerID",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
        "affinity_type",
        "affinity_relation",
        "affinity_value",
    ]].copy()

    raw_rows.append(result)

    print(
        f"chunk {chunk_idx}: "
        f"{len(result):,} rows"
    )


raw = pd.concat(
    raw_rows,
    ignore_index=True
)

print("\nTSV에서 parser 방식으로 생성된 row:")
print(f"{len(raw):,}")


# ================================================================
# 5. CSV와 column 이름 맞추기
# ================================================================

raw = raw.rename(columns={
    "BindingDB Reactant_set_id": "source_record_id",
    "Ligand SMILES": "smiles",
    "ChEMBL ID of Ligand": "chembl_id",
    "BindingDB MonomerID": "bindingdb_id",
    "Target Name": "target_name",
    "Target Source Organism According to Curator or DataSource":
        "organism",
})


# ================================================================
# 6. 비교 key 생성
# ================================================================

COMPARE_KEYS = [
    "source_record_id",
    "bindingdb_id",
    "affinity_type",
]

print("\n비교 key:")
print(COMPARE_KEYS)


# ================================================================
# 7. relation 비교
# ================================================================

merged = raw.merge(
    csv[
        COMPARE_KEYS
        + [
            "affinity_relation",
            "affinity_value",
            "smiles",
            "chembl_id",
            "target_name",
        ]
    ],
    on=COMPARE_KEYS,
    how="outer",
    suffixes=("_raw", "_csv"),
    indicator=True
)

print("\n" + "=" * 100)
print("1. ROW MATCH QC")
print("=" * 100)

print(
    merged["_merge"]
    .value_counts(dropna=False)
)


# ================================================================
# 8. 양쪽 모두 존재하는 데이터만 비교
# ================================================================

matched = merged[
    merged["_merge"] == "both"
].copy()

print(
    f"\n양쪽 모두 존재하는 measurement: "
    f"{len(matched):,}"
)


# ================================================================
# 9. Relation 비교
# ================================================================

matched["relation_same"] = (
    matched["affinity_relation_raw"]
    ==
    matched["affinity_relation_csv"]
)

relation_diff = matched[
    ~matched["relation_same"]
]

print("\n" + "=" * 100)
print("2. AFFINITY RELATION 비교")
print("=" * 100)

print(
    f"동일: {matched['relation_same'].sum():,}"
)

print(
    f"불일치: {len(relation_diff):,}"
)

if len(relation_diff) > 0:

    print("\nRelation 변환 예시:")

    print(
        relation_diff[
            [
                "source_record_id",
                "affinity_type",
                "affinity_relation_raw",
                "affinity_relation_csv",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ================================================================
# 10. Affinity 값 비교
# ================================================================

matched["affinity_value_raw"] = pd.to_numeric(
    matched["affinity_value_raw"],
    errors="coerce"
)

matched["affinity_value_csv"] = pd.to_numeric(
    matched["affinity_value_csv"],
    errors="coerce"
)

matched["value_diff"] = (
    matched["affinity_value_csv"]
    -
    matched["affinity_value_raw"]
).abs()

matched["value_same"] = (
    matched["value_diff"] <= 1e-12
)

print("\n" + "=" * 100)
print("3. AFFINITY VALUE 비교")
print("=" * 100)

print(
    f"동일: {matched['value_same'].sum():,}"
)

print(
    f"불일치: "
    f"{(~matched['value_same']).sum():,}"
)

print(
    f"Max absolute difference: "
    f"{matched['value_diff'].max()}"
)

print(
    f"Mean absolute difference: "
    f"{matched['value_diff'].mean()}"
)


# ================================================================
# 11. SMILES 비교
# ================================================================

matched["smiles_same"] = (
    matched["smiles_raw"].fillna("")
    ==
    matched["smiles_csv"].fillna("")
)

print("\n" + "=" * 100)
print("4. SMILES 비교")
print("=" * 100)

print(
    f"동일: {matched['smiles_same'].sum():,}"
)

print(
    f"불일치: {(~matched['smiles_same']).sum():,}"
)


# ================================================================
# 12. ChEMBL ID 비교
# ================================================================

matched["chembl_same"] = (
    matched["chembl_id_raw"].fillna("")
    ==
    matched["chembl_id_csv"].fillna("")
)

print("\n" + "=" * 100)
print("5. ChEMBL ID 비교")
print("=" * 100)

print(
    f"동일: {matched['chembl_same'].sum():,}"
)

print(
    f"불일치: {(~matched['chembl_same']).sum():,}"
)


# ================================================================
# 13. Target Name 비교
# ================================================================

matched["target_name_same"] = (
    matched["target_name_raw"].fillna("")
    ==
    matched["target_name_csv"].fillna("")
)

print("\n" + "=" * 100)
print("6. TARGET NAME 비교")
print("=" * 100)

print(
    f"동일: {matched['target_name_same'].sum():,}"
)

print(
    f"불일치: "
    f"{(~matched['target_name_same']).sum():,}"
)


# ================================================================
# 14. 종합
# ================================================================

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(f"원본 TSV parser 기준 row : {len(raw):,}")
print(f"현재 CSV row             : {len(csv):,}")
print(f"Matched row              : {len(matched):,}")

print(
    f"\nRelation 불일치          : "
    f"{len(relation_diff):,}"
)

print(
    f"Affinity value 불일치    : "
    f"{(~matched['value_same']).sum():,}"
)

print(
    f"SMILES 불일치            : "
    f"{(~matched['smiles_same']).sum():,}"
)

print(
    f"ChEMBL ID 불일치         : "
    f"{(~matched['chembl_same']).sum():,}"
)

print(
    f"Target name 불일치       : "
    f"{(~matched['target_name_same']).sum():,}"
)