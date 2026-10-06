import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"
)

REFERENCE_FILE = (
    BASE_DIR /
    "uniprot_protein_reference_sequence_fixed_api14.csv"
)

INPUT_FILES = {
    "IC50": BASE_DIR / "IC50.csv",
    "Ki": BASE_DIR / "Ki.csv",
    "Kd": BASE_DIR / "Kd.csv",
}

OUTPUT_FILES = {
    "IC50": BASE_DIR / "IC50_protein_matched.csv",
    "Ki": BASE_DIR / "Ki_protein_matched.csv",
    "Kd": BASE_DIR / "Kd_protein_matched.csv",
}


# ============================================================
# REFERENCE COLUMNS
# ============================================================

REFERENCE_COLUMNS = [
    "uniprot_id",
    "protein_name",
    "gene_name",
    "organism",
    "reviewed",
    "protein_sequence",
    "sequence_length",
]


# ============================================================
# LOAD REFERENCE
# ============================================================

print("=" * 100)
print("Protein Sequence Matching to IC50 / Ki / Kd")
print("=" * 100)

print()
print("Reference:")
print(REFERENCE_FILE)

reference_df = pd.read_csv(
    REFERENCE_FILE,
    low_memory=False
)

print()
print(
    f"Reference rows    : {len(reference_df):,}"
)
print(
    f"Reference columns : {len(reference_df.columns):,}"
)


# ============================================================
# REFERENCE VALIDATION
# ============================================================

if "uniprot_id" not in reference_df.columns:
    raise ValueError(
        "Reference에 'uniprot_id'가 없습니다."
    )

if reference_df.columns.tolist().count("uniprot_id") != 1:
    raise ValueError(
        "Reference의 uniprot_id 컬럼이 중복되어 있습니다."
    )


# ------------------------------------------------------------
# Reference duplicate UniProt ID 확인
# ------------------------------------------------------------

reference_dup = reference_df[
    reference_df["uniprot_id"].duplicated(
        keep=False
    )
]

if len(reference_dup) > 0:

    print()
    print(
        "[ERROR] Reference에 중복 UniProt ID가 존재합니다."
    )

    print(
        reference_dup[
            ["uniprot_id"]
        ].head(20)
    )

    raise ValueError(
        "Reference의 UniProt ID는 반드시 unique해야 합니다."
    )


# ============================================================
# SELECT REFERENCE COLUMNS
# ============================================================

missing_reference_columns = [
    col
    for col in REFERENCE_COLUMNS
    if col not in reference_df.columns
]

if missing_reference_columns:

    raise ValueError(
        "Reference에 필요한 컬럼이 없습니다: "
        + ", ".join(missing_reference_columns)
    )


reference_subset = reference_df[
    REFERENCE_COLUMNS
].copy()


# ============================================================
# NORMALIZE REFERENCE UNIPROT ID
# ============================================================

reference_subset["uniprot_id"] = (
    reference_subset["uniprot_id"]
    .astype("string")
    .str.strip()
)


# ============================================================
# REFERENCE SEQUENCE QC
# ============================================================

reference_sequence_available = (
    reference_subset["protein_sequence"]
    .notna()
    &
    reference_subset["protein_sequence"]
    .astype("string")
    .str.strip()
    .ne("")
)

print()
print("=" * 100)
print("REFERENCE QC")
print("=" * 100)

print(
    f"Total UniProt IDs      : "
    f"{len(reference_subset):,}"
)

print(
    f"Sequence available     : "
    f"{reference_sequence_available.sum():,}"
)

print(
    f"Sequence missing       : "
    f"{(~reference_sequence_available).sum():,}"
)


# ============================================================
# PROCESS EACH DATASET
# ============================================================

for dataset_name in ["IC50", "Ki", "Kd"]:

    input_file = INPUT_FILES[dataset_name]
    output_file = OUTPUT_FILES[dataset_name]

    print()
    print()
    print("=" * 100)
    print(f"{dataset_name} PROCESSING")
    print("=" * 100)

    print()
    print(
        f"Input : {input_file}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    original_row_count = len(df)

    print(
        f"Rows before matching : "
        f"{original_row_count:,}"
    )

    print(
        f"Columns before       : "
        f"{len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Validate uniprot_id
    # --------------------------------------------------------

    if "uniprot_id" not in df.columns:

        raise ValueError(
            f"{dataset_name}: "
            "'uniprot_id' 컬럼이 없습니다."
        )

    if df.columns.tolist().count("uniprot_id") != 1:

        raise ValueError(
            f"{dataset_name}: "
            "'uniprot_id' 컬럼이 중복되어 있습니다."
        )

    # --------------------------------------------------------
    # Preserve original IDs/order
    # --------------------------------------------------------

    original_ids = (
        df["uniprot_id"]
        .astype("string")
        .tolist()
    )

    # --------------------------------------------------------
    # Normalize ID for matching
    # --------------------------------------------------------

    df["_match_uniprot_id"] = (
        df["uniprot_id"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    dataset_unique_ids = set(
        df["_match_uniprot_id"]
        .dropna()
        .tolist()
    )

    reference_ids = set(
        reference_subset["uniprot_id"]
        .dropna()
        .tolist()
    )

    missing_reference_ids = (
        dataset_unique_ids
        - reference_ids
    )

    print()
    print(
        f"Unique UniProt IDs    : "
        f"{len(dataset_unique_ids):,}"
    )

    print(
        f"IDs found in reference: "
        f"{len(dataset_unique_ids - missing_reference_ids):,}"
    )

    print(
        f"IDs missing reference : "
        f"{len(missing_reference_ids):,}"
    )

    # --------------------------------------------------------
    # Missing ID report
    # --------------------------------------------------------

    if missing_reference_ids:

        print()
        print(
            "[WARNING] Reference에 없는 UniProt ID:"
        )

        for uid in sorted(
            missing_reference_ids
        )[:100]:

            print(
                f"  {uid}"
            )

    # --------------------------------------------------------
    # Remove existing protein columns if they exist
    # --------------------------------------------------------
    #
    # 중요:
    # 기존 protein 정보가 있더라도 reference의 최종 정보를
    # 기준으로 다시 붙이기 위해 제거합니다.
    #
    # 단, 원본 데이터의 uniprot_id는 절대 제거하지 않습니다.
    #

    protein_columns_to_replace = [
        "protein_name",
        "gene_name",
        "organism",
        "reviewed",
        "protein_sequence",
        "sequence_length",
    ]

    existing_protein_columns = [
        col
        for col in protein_columns_to_replace
        if col in df.columns
    ]

    if existing_protein_columns:

        print()
        print(
            "Existing protein columns found:"
        )

        for col in existing_protein_columns:
            print(
                f"  - {col}"
            )

        df = df.drop(
            columns=existing_protein_columns
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = df.merge(
        reference_subset,
        how="left",
        left_on="_match_uniprot_id",
        right_on="uniprot_id",
        validate="many_to_one",
        sort=False,
        suffixes=("", "_reference")
    )

    # --------------------------------------------------------
    # Remove duplicate right-side ID
    # --------------------------------------------------------

    if "uniprot_id_reference" in df.columns:

        # 원본 uniprot_id 보존
        df = df.drop(
            columns=["uniprot_id_reference"]
        )

    # --------------------------------------------------------
    # Remove helper column
    # --------------------------------------------------------

    df = df.drop(
        columns=["_match_uniprot_id"]
    )

    # --------------------------------------------------------
    # Row count validation
    # --------------------------------------------------------

    if len(df) != original_row_count:

        raise RuntimeError(
            f"{dataset_name}: "
            f"merge 후 row count가 변경되었습니다. "
            f"{original_row_count:,} -> {len(df):,}"
        )

    print()
    print(
        "[PASS] Row count preserved."
    )

    # --------------------------------------------------------
    # UniProt ID / order validation
    # --------------------------------------------------------

    new_ids = (
        df["uniprot_id"]
        .astype("string")
        .tolist()
    )

    if new_ids != original_ids:

        raise RuntimeError(
            f"{dataset_name}: "
            "UniProt ID 또는 row order가 변경되었습니다."
        )

    print(
        "[PASS] UniProt ID/order preserved."
    )

    # --------------------------------------------------------
    # Sequence coverage
    # --------------------------------------------------------

    sequence_available = (
        df["protein_sequence"]
        .notna()
        &
        df["protein_sequence"]
        .astype("string")
        .str.strip()
        .ne("")
    )

    sequence_missing = (
        ~sequence_available
    )

    print()
    print(
        "SEQUENCE MATCHING"
    )

    print(
        f"Rows with sequence : "
        f"{sequence_available.sum():,}"
    )

    print(
        f"Rows without       : "
        f"{sequence_missing.sum():,}"
    )

    print(
        f"Match rate         : "
        f"{sequence_available.mean() * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Missing sequence IDs
    # --------------------------------------------------------

    missing_sequence_ids = sorted(
        set(
            df.loc[
                sequence_missing,
                "uniprot_id"
            ]
            .astype("string")
            .dropna()
            .tolist()
        )
    )

    print()
    print(
        f"Unique IDs without sequence : "
        f"{len(missing_sequence_ids):,}"
    )

    if missing_sequence_ids:

        print(
            "Examples:"
        )

        for uid in missing_sequence_ids[:30]:

            print(
                f"  {uid}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"Saved:"
    )

    print(
        output_file
    )

    # --------------------------------------------------------
    # Post-save verification
    # --------------------------------------------------------

    check_df = pd.read_csv(
        output_file,
        low_memory=False
    )

    if len(check_df) != original_row_count:

        raise RuntimeError(
            f"{dataset_name}: "
            "저장 후 row count가 변경되었습니다."
        )

    check_ids = (
        check_df["uniprot_id"]
        .astype("string")
        .tolist()
    )

    if check_ids != original_ids:

        raise RuntimeError(
            f"{dataset_name}: "
            "저장 후 UniProt ID/order가 변경되었습니다."
        )

    print(
        "[PASS] Post-save verification."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 100)
print("ALL DATASETS COMPLETE")
print("=" * 100)

for dataset_name in ["IC50", "Ki", "Kd"]:

    print()
    print(
        f"{dataset_name}:"
    )

    print(
        f"  Input : {INPUT_FILES[dataset_name]}"
    )

    print(
        f"  Output: {OUTPUT_FILES[dataset_name]}"
    )

print()
print("=" * 100)
print("DONE")
print("=" * 100)