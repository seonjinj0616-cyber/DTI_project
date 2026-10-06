import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINDINGDB_TSV = (
    PROJECT_ROOT
    / "data"
    / "bindingdb"
    / "BindingDB_All.tsv"
)

BINDINGDB_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "bindingdb_dti.csv"
)

CHEMBL_CSV = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "chembl_dti.csv"
)

CHEMBL_DB = (
    PROJECT_ROOT
    / "data"
    / "chembl"
    / "chembl_37_sqlite"
    / "chembl_37"
    / "chembl_37_sqlite"
    / "chembl_37.db"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def section(title):
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# PATH CHECK
# ============================================================

section("0. FILE PATH CHECK")

paths = {
    "BindingDB RAW TSV": BINDINGDB_TSV,
    "BindingDB CSV": BINDINGDB_CSV,
    "ChEMBL CSV": CHEMBL_CSV,
    "ChEMBL SQLite": CHEMBL_DB,
}

for name, path in paths.items():
    print(f"{name:25s}: {path}")
    print(f"{'EXISTS':25s}: {path.exists()}")
    print()


# ============================================================
# 1. BINDINGDB RAW TSV
# ============================================================

section("1. BindingDB RAW TSV QC")


if not BINDINGDB_TSV.exists():
    print("BindingDB RAW TSV를 찾을 수 없습니다.")
else:

    affinity_cols = [
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
    ]

    # 필요한 컬럼만 읽음
    raw_bdb = pd.read_csv(
        BINDINGDB_TSV,
        sep="\t",
        usecols=lambda c: c in affinity_cols,
        dtype=str,
        low_memory=False
    )

    print(f"전체 RAW Row 수: {len(raw_bdb):,}")

    raw_counts = {}

    for col in affinity_cols:

        if col not in raw_bdb.columns:
            print(f"\n{col}: COLUMN NOT FOUND")
            raw_counts[col] = 0
            continue

        s = raw_bdb[col].dropna().astype(str).str.strip()

        raw_counts[col] = len(s)

        print(f"\n[{col}]")
        print(f"Non-null: {len(s):,}")

        # relation
        gt = s.str.startswith(">")
        lt = s.str.startswith("<")

        print(f"  > 값: {gt.sum():,}")
        print(f"  < 값: {lt.sum():,}")

        # 숫자 변환
        cleaned = (
            s.str.replace(">", "", regex=False)
             .str.replace("<", "", regex=False)
             .str.strip()
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        print(f"  Numeric: {numeric.notna().sum():,}")
        print(f"  Non-numeric: {numeric.isna().sum():,}")

        if numeric.isna().sum() > 0:
            print("\n  Non-numeric examples:")
            print(
                s[numeric.isna()]
                .head(20)
                .to_string(index=False)
            )

        # unit-like 문자열 탐색
        unit_pattern = (
            r"(nM|uM|µM|μM|mM|pM|fM|"
            r"mol|g/L|mg|ug|µg)"
        )

        unit_like = s.str.contains(
            unit_pattern,
            case=False,
            regex=True,
            na=False
        )

        print(
            f"  Unit-like 문자열 포함: "
            f"{unit_like.sum():,}"
        )

        if unit_like.sum() > 0:
            print("\n  Unit-like examples:")
            print(
                s[unit_like]
                .head(20)
                .to_string(index=False)
            )


# ============================================================
# 2. BINDINGDB CSV QC
# ============================================================

section("2. BindingDB PARSED CSV QC")


if not BINDINGDB_CSV.exists():

    print("BindingDB CSV를 찾을 수 없습니다.")

else:

    bdb = pd.read_csv(
        BINDINGDB_CSV,
        low_memory=False
    )

    print(f"Rows: {len(bdb):,}")
    print(f"Columns: {len(bdb.columns)}")

    subsection("2-1. Affinity Type")

    print(
        bdb["affinity_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("2-2. Affinity Unit")

    print(
        bdb["affinity_unit"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("2-3. Affinity Relation")

    print(
        bdb["affinity_relation"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("2-4. Affinity Statistics")

    print(
        bdb["affinity_value"]
        .describe()
        .to_string()
    )

    subsection("2-5. pAffinity Statistics")

    print(
        bdb["pchembl_value"]
        .describe()
        .to_string()
    )

    subsection("2-6. Drug-Target Pair")

    valid = bdb.dropna(
        subset=["smiles", "uniprot_id"]
    )

    pair_counts = (
        valid
        .groupby(["smiles", "uniprot_id"])
        .size()
    )

    print(
        f"Unique Drug-Target Pair: "
        f"{len(pair_counts):,}"
    )

    if len(pair_counts) > 0:

        print(
            f"Mean measurement/pair: "
            f"{pair_counts.mean():.2f}"
        )

        print(
            f"Median measurement/pair: "
            f"{pair_counts.median():.0f}"
        )

        print(
            f"Max measurement/pair: "
            f"{pair_counts.max():,}"
        )

        print(
            f"Multi-measurement pair: "
            f"{(pair_counts > 1).sum():,}"
        )


# ============================================================
# 3. BINDINGDB RAW VS CSV COUNT COMPARISON
# ============================================================

section("3. BindingDB RAW vs CSV COUNT COMPARISON")


if BINDINGDB_CSV.exists():

    bdb = pd.read_csv(
        BINDINGDB_CSV,
        low_memory=False
    )

    csv_counts = (
        bdb["affinity_type"]
        .value_counts()
        .to_dict()
    )

    mapping = {
        "Ki (nM)": "Ki",
        "IC50 (nM)": "IC50",
        "Kd (nM)": "Kd",
    }

    print(
        f"{'Affinity':<10}"
        f"{'RAW':>15}"
        f"{'CSV':>15}"
        f"{'Difference':>15}"
        f"{'Status':>12}"
    )

    print("-" * 70)

    for raw_col, affinity_type in mapping.items():

        raw_count = raw_counts.get(
            raw_col,
            0
        )

        csv_count = csv_counts.get(
            affinity_type,
            0
        )

        diff = csv_count - raw_count

        status = (
            "OK"
            if diff == 0
            else "CHECK"
        )

        print(
            f"{affinity_type:<10}"
            f"{raw_count:>15,}"
            f"{csv_count:>15,}"
            f"{diff:>15,}"
            f"{status:>12}"
        )


# ============================================================
# 4. CHEMBL CSV UNIT DISTRIBUTION
# ============================================================

section("4. ChEMBL PARSED CSV UNIT QC")


if not CHEMBL_CSV.exists():

    print("ChEMBL CSV를 찾을 수 없습니다.")

else:

    chembl = pd.read_csv(
        CHEMBL_CSV,
        low_memory=False
    )

    print(f"Rows: {len(chembl):,}")
    print(f"Columns: {len(chembl.columns)}")

    subsection("4-1. Affinity Type")

    print(
        chembl["affinity_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("4-2. Affinity Unit")

    print(
        chembl["affinity_unit"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("4-3. Affinity Type x Unit")

    print(
        pd.crosstab(
            chembl["affinity_type"],
            chembl["affinity_unit"],
            dropna=False
        ).to_string()
    )


# ============================================================
# 5. CHEMBL EXTREME VALUE INVESTIGATION
# ============================================================

section("5. ChEMBL EXTREME VALUE INVESTIGATION")


if CHEMBL_CSV.exists():

    cols = [
        "chembl_id",
        "target_chembl_id",
        "uniprot_id",
        "affinity_type",
        "affinity_relation",
        "affinity_value",
        "affinity_unit",
        "pchembl_value",
    ]

    available_cols = [
        c for c in cols
        if c in chembl.columns
    ]

    subsection("5-1. Lowest 30 Affinity Values")

    print(
        chembl
        .sort_values("affinity_value")
        [available_cols]
        .head(30)
        .to_string(index=False)
    )

    subsection("5-2. Highest 30 Affinity Values")

    print(
        chembl
        .sort_values(
            "affinity_value",
            ascending=False
        )
        [available_cols]
        .head(30)
        .to_string(index=False)
    )

    subsection("5-3. pChEMBL < 0")

    low = chembl[
        chembl["pchembl_value"] < 0
    ]

    print(f"Count: {len(low):,}")

    print("\nUnit distribution:")
    print(
        low["affinity_unit"]
        .value_counts(dropna=False)
        .to_string()
    )

    subsection("5-4. pChEMBL > 12")

    high = chembl[
        chembl["pchembl_value"] > 12
    ]

    print(f"Count: {len(high):,}")

    print("\nUnit distribution:")
    print(
        high["affinity_unit"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nHighest 30 pChEMBL:")

    print(
        chembl
        .sort_values(
            "pchembl_value",
            ascending=False
        )
        [available_cols]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 6. CHEMBL pChEMBL CONSISTENCY
# ============================================================

section("6. ChEMBL pChEMBL CONSISTENCY QC")


if CHEMBL_CSV.exists():

    valid = chembl[
        (chembl["affinity_unit"] == "nM") &
        (chembl["affinity_value"] > 0) &
        (chembl["pchembl_value"].notna())
    ].copy()

    calculated = (
        -np.log10(
            valid["affinity_value"] * 1e-9
        )
    )

    diff = (
        calculated
        - valid["pchembl_value"]
    ).abs()

    print(
        f"검사 대상: {len(valid):,}"
    )

    print(
        f"|difference| > 0.01: "
        f"{(diff > 0.01).sum():,}"
    )

    if len(diff) > 0:

        print(
            f"Mean difference: "
            f"{diff.mean():.6f}"
        )

        print(
            f"Max difference: "
            f"{diff.max():.6f}"
        )


# ============================================================
# 7. CHEMBL ORIGINAL SQLITE EXTREME VALUES
# ============================================================

section("7. ChEMBL ORIGINAL SQLite EXTREME VALUE CHECK")


if not CHEMBL_DB.exists():

    print("ChEMBL SQLite DB를 찾을 수 없습니다.")

else:

    conn = sqlite3.connect(
        CHEMBL_DB
    )

    query = """
    SELECT
        a.activity_id,
        md.chembl_id AS chembl_id,
        td.chembl_id AS target_chembl_id,
        cs.accession AS uniprot_id,
        a.standard_type,
        a.standard_relation,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.data_validity_comment,
        a.potential_duplicate,
        a.activity_comment
    FROM activities a
    JOIN assays ass
        ON a.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    JOIN target_components tc
        ON td.tid = tc.tid
    JOIN component_sequences cs
        ON tc.component_id = cs.component_id
    JOIN molecule_dictionary md
        ON a.molregno = md.molregno
    WHERE
        td.target_type = 'SINGLE PROTEIN'
        AND td.organism = 'Homo sapiens'
        AND ass.assay_type = 'B'
        AND a.standard_type IN ('Ki', 'IC50', 'Kd')
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
    ORDER BY a.standard_value ASC
    LIMIT 30
    """

    subsection(
        "7-1. ORIGINAL DB Lowest 30"
    )

    lowest = pd.read_sql_query(
        query,
        conn
    )

    print(
        lowest.to_string(index=False)
    )

    query_high = """
    SELECT
        a.activity_id,
        md.chembl_id AS chembl_id,
        td.chembl_id AS target_chembl_id,
        cs.accession AS uniprot_id,
        a.standard_type,
        a.standard_relation,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.data_validity_comment,
        a.potential_duplicate,
        a.activity_comment
    FROM activities a
    JOIN assays ass
        ON a.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    JOIN target_components tc
        ON td.tid = tc.tid
    JOIN component_sequences cs
        ON tc.component_id = cs.component_id
    JOIN molecule_dictionary md
        ON a.molregno = md.molregno
    WHERE
        td.target_type = 'SINGLE PROTEIN'
        AND td.organism = 'Homo sapiens'
        AND ass.assay_type = 'B'
        AND a.standard_type IN ('Ki', 'IC50', 'Kd')
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
    ORDER BY a.standard_value DESC
    LIMIT 30
    """

    subsection(
        "7-2. ORIGINAL DB Highest 30"
    )

    highest = pd.read_sql_query(
        query_high,
        conn
    )

    print(
        highest.to_string(index=False)
    )

    # --------------------------------------------------------
    # Extreme unit distribution
    # --------------------------------------------------------

    subsection(
        "7-3. ORIGINAL DB Unit Distribution"
    )

    unit_query = """
    SELECT
        standard_units,
        COUNT(*) AS count
    FROM activities a
    JOIN assays ass
        ON a.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    WHERE
        td.target_type = 'SINGLE PROTEIN'
        AND td.organism = 'Homo sapiens'
        AND ass.assay_type = 'B'
        AND a.standard_type IN ('Ki', 'IC50', 'Kd')
        AND a.standard_value IS NOT NULL
        AND a.standard_value > 0
    GROUP BY standard_units
    ORDER BY count DESC
    """

    units = pd.read_sql_query(
        unit_query,
        conn
    )

    print(
        units.to_string(index=False)
    )

    # --------------------------------------------------------
    # pChEMBL > 12 원본
    # --------------------------------------------------------

    subsection(
        "7-4. ORIGINAL DB pChEMBL > 12"
    )

    # 위 query의 JOIN 순서를 안전하게 다시 작성
    high_p_query = """
    SELECT
        a.activity_id,
        md.chembl_id AS chembl_id,
        td.chembl_id AS target_chembl_id,
        cs.accession AS uniprot_id,
        a.standard_type,
        a.standard_relation,
        a.standard_value,
        a.standard_units,
        a.pchembl_value,
        a.data_validity_comment,
        a.potential_duplicate,
        a.activity_comment
    FROM activities a
    JOIN assays ass
        ON a.assay_id = ass.assay_id
    JOIN target_dictionary td
        ON ass.tid = td.tid
    JOIN target_components tc
        ON td.tid = tc.tid
    JOIN component_sequences cs
        ON tc.component_id = cs.component_id
    JOIN molecule_dictionary md
        ON a.molregno = md.molregno
    WHERE
        td.target_type = 'SINGLE PROTEIN'
        AND td.organism = 'Homo sapiens'
        AND ass.assay_type = 'B'
        AND a.standard_type IN ('Ki', 'IC50', 'Kd')
        AND a.standard_value IS NOT NULL
        AND a.pchembl_value > 12
    ORDER BY a.pchembl_value DESC
    LIMIT 50
    """

    high_p = pd.read_sql_query(
        high_p_query,
        conn
    )

    print(
        high_p.to_string(index=False)
    )

    conn.close()


# ============================================================
# 8. FINAL DIAGNOSIS
# ============================================================

section("8. FINAL QC SUMMARY")

print(
    """
[BindingDB]

1. Ki / IC50 / Kd는 원본 컬럼명이 각각 '(nM)'이므로
   현재 parsing 대상 affinity는 nM 기반으로 처리.

2. RAW TSV의 Ki / IC50 / Kd non-null 개수와
   parsed CSV의 affinity_type 개수를 비교하여
   parsing 누락 여부 확인.

3. CSV의 affinity_unit은 현재 parser가 'nM'으로
   지정하고 있으므로, CSV의 nM 100% 자체는
   원본 unit 검증 결과가 아니라 parser의 결과임.

4. pchembl_value는 BindingDB 원본 pChEMBL이 아니라
   affinity_value로부터 계산된 값인지 확인 필요.


[ChEMBL]

1. affinity_unit 분포를 먼저 확인.

2. pChEMBL extreme 값이 어떤 standard_units에서
   발생하는지 확인.

3. CSV extreme 값과 원본 SQLite의
   standard_value / standard_units / pchembl_value를
   직접 비교.

4. data_validity_comment,
   potential_duplicate,
   activity_comment를 함께 확인.

5. 실제 원본 unit이 nM인지 확인한 후
   unit normalization 및 extreme-value 처리 진행.
"""
)

print("\n")
print("=" * 80)
print("ALL QC COMPLETE")
print("=" * 80)