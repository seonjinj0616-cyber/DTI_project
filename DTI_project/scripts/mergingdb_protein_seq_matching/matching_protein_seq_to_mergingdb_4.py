import os
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\esm2_bindingdb_chembl vectorDB"
)

REFERENCE_FILE = os.path.join(
    BASE_DIR,
    "uniprot_protein_reference_sequence_fixed_api14.csv"
)

API4_FILE = os.path.join(
    BASE_DIR,
    "uniprot_api_4_recovery.csv"
)

DATASET_FILES = {
    "IC50": os.path.join(
        BASE_DIR,
        "IC50_protein_matched_idqc_api.csv"
    ),
    "Ki": os.path.join(
        BASE_DIR,
        "Ki_protein_matched_idqc_api.csv"
    ),
    "Kd": os.path.join(
        BASE_DIR,
        "Kd_protein_matched_idqc_api.csv"
    ),
}


# ============================================================
# OUTPUT
# ============================================================

FINAL_REFERENCE_FILE = os.path.join(
    BASE_DIR,
    "uniprot_protein_reference_final.csv"
)

FINAL_DATASET_FILES = {
    "IC50": os.path.join(
        BASE_DIR,
        "IC50_protein_matched_final.csv"
    ),
    "Ki": os.path.join(
        BASE_DIR,
        "Ki_protein_matched_final.csv"
    ),
    "Kd": os.path.join(
        BASE_DIR,
        "Kd_protein_matched_final.csv"
    ),
}


# ============================================================
# 처리 대상
# ============================================================

# API에서 정상적으로 sequence 확보
UPDATE_IDS = [
    "P02795",
    "P03928",
]

# API에서 sequence 확보 실패
DELETE_IDS = [
    "Q2L8D9",
    "O60361",
]


# ============================================================
# Utility
# ============================================================

def normalize_id(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value == "":
        return ""

    return value.split()[0]


def has_sequence(value):

    if pd.isna(value):
        return False

    return str(value).strip() != ""


# ============================================================
# START
# ============================================================

print("=" * 100)
print("FINAL PROTEIN SEQUENCE APPLY")
print("=" * 100)


# ============================================================
# 1. API recovery 결과
# ============================================================

print("\n[1] API recovery 결과 로드")
print("-" * 100)

api_df = pd.read_csv(
    API4_FILE,
    low_memory=False
)

api_df["uniprot_id"] = (
    api_df["uniprot_id"]
    .apply(normalize_id)
)

print(
    api_df[
        [
            "uniprot_id",
            "protein_name",
            "gene_name",
            "organism",
            "reviewed",
            "sequence_length",
            "status",
        ]
    ].to_string(index=False)
)


# ============================================================
# 2. Reference 로드
# ============================================================

print("\n" + "=" * 100)
print("[2] Reference 로드")
print("=" * 100)

ref = pd.read_csv(
    REFERENCE_FILE,
    low_memory=False
)

ref["uniprot_id"] = (
    ref["uniprot_id"]
    .apply(normalize_id)
)

original_reference_rows = len(ref)

print(
    f"Original reference rows : "
    f"{original_reference_rows:,}"
)


# ============================================================
# 3. Reference dtype 안전 처리
# ============================================================

# API 결과를 넣을 컬럼들은 문자열로 통일
# reviewed 역시 문자열로 저장해서 dtype 충돌 방지

metadata_columns = [
    "protein_name",
    "gene_name",
    "organism",
    "reviewed",
]

for col in metadata_columns:

    if col in ref.columns:

        ref[col] = (
            ref[col]
            .astype("string")
        )


# sequence도 string으로 통일
if "protein_sequence" in ref.columns:

    ref["protein_sequence"] = (
        ref["protein_sequence"]
        .astype("string")
    )


# sequence_length는 numeric으로 통일
if "sequence_length" in ref.columns:

    ref["sequence_length"] = pd.to_numeric(
        ref["sequence_length"],
        errors="coerce"
    )


# ============================================================
# 4. P02795 / P03928 업데이트
# ============================================================

print("\n" + "=" * 100)
print("[3] API 정상 ID 업데이트")
print("=" * 100)


for uid in UPDATE_IDS:

    api_match = api_df[
        api_df["uniprot_id"] == uid
    ]

    if len(api_match) == 0:

        raise ValueError(
            f"[ERROR] API 결과에 {uid}가 없습니다."
        )

    api_row = api_match.iloc[0]

    sequence = api_row["protein_sequence"]

    if not has_sequence(sequence):

        raise ValueError(
            f"[ERROR] {uid}의 sequence가 없습니다."
        )

    sequence = str(sequence).strip()

    mask = (
        ref["uniprot_id"] == uid
    )

    reference_count = int(mask.sum())

    if reference_count == 0:

        raise ValueError(
            f"[ERROR] Reference에 {uid}가 없습니다."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if "protein_name" in ref.columns:

        value = api_row["protein_name"]

        if pd.notna(value):

            ref.loc[
                mask,
                "protein_name"
            ] = str(value)

    if "gene_name" in ref.columns:

        value = api_row["gene_name"]

        if pd.notna(value):

            ref.loc[
                mask,
                "gene_name"
            ] = str(value)

    if "organism" in ref.columns:

        value = api_row["organism"]

        if pd.notna(value):

            ref.loc[
                mask,
                "organism"
            ] = str(value)

    if "reviewed" in ref.columns:

        value = api_row["reviewed"]

        if pd.notna(value):

            # True/False를 문자열로 저장
            value = str(value)

            ref.loc[
                mask,
                "reviewed"
            ] = value

    # --------------------------------------------------------
    # Sequence
    # --------------------------------------------------------

    ref.loc[
        mask,
        "protein_sequence"
    ] = sequence

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if "sequence_length" in ref.columns:

        ref.loc[
            mask,
            "sequence_length"
        ] = len(sequence)

    print(
        f"[UPDATED] {uid}"
    )

    print(
        f"  sequence length : {len(sequence)}"
    )

    print(
        f"  reference rows  : {reference_count}"
    )


# ============================================================
# 5. Q2L8D9 / O60361 reference에서 삭제
# ============================================================

print("\n" + "=" * 100)
print("[4] API 미연동 ID Reference 삭제")
print("=" * 100)


for uid in DELETE_IDS:

    mask = (
        ref["uniprot_id"] == uid
    )

    remove_count = int(mask.sum())

    ref = ref.loc[
        ~mask
    ].copy()

    print(
        f"[DELETED] {uid} "
        f"| removed rows = {remove_count}"
    )


# ============================================================
# 6. Reference 저장
# ============================================================

print("\n" + "=" * 100)
print("[5] Reference 저장")
print("=" * 100)

ref.to_csv(
    FINAL_REFERENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Output : {FINAL_REFERENCE_FILE}"
)

print(
    f"Rows   : "
    f"{original_reference_rows:,} -> {len(ref):,}"
)


# ============================================================
# 7. IC50 / Ki / Kd 처리
# ============================================================

print("\n" + "=" * 100)
print("[6] IC50 / Ki / Kd 처리")
print("=" * 100)


dataset_summary = []


for dataset_name, input_file in DATASET_FILES.items():

    print("\n" + "-" * 100)
    print(dataset_name)

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    original_rows = len(df)

    df["uniprot_id"] = (
        df["uniprot_id"]
        .apply(normalize_id)
    )

    # --------------------------------------------------------
    # 삭제 대상
    # --------------------------------------------------------

    delete_mask = (
        df["uniprot_id"]
        .isin(DELETE_IDS)
    )

    removed_df = df.loc[
        delete_mask
    ].copy()

    final_df = df.loc[
        ~delete_mask
    ].copy()

    # --------------------------------------------------------
    # ID별 삭제 수
    # --------------------------------------------------------

    q2l8d9_count = int(
        (
            removed_df["uniprot_id"]
            == "Q2L8D9"
        ).sum()
    )

    o60361_count = int(
        (
            removed_df["uniprot_id"]
            == "O60361"
        ).sum()
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    output_file = FINAL_DATASET_FILES[
        dataset_name
    ]

    final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Original rows : {original_rows:,}"
    )

    print(
        f"Q2L8D9 removed: {q2l8d9_count:,}"
    )

    print(
        f"O60361 removed: {o60361_count:,}"
    )

    print(
        f"Total removed : {len(removed_df):,}"
    )

    print(
        f"Final rows    : {len(final_df):,}"
    )

    print(
        f"Saved         : {output_file}"
    )

    dataset_summary.append({
        "dataset": dataset_name,
        "original_rows": original_rows,
        "Q2L8D9_removed": q2l8d9_count,
        "O60361_removed": o60361_count,
        "total_removed": len(removed_df),
        "final_rows": len(final_df),
    })


# ============================================================
# 8. Reference 최종 QC
# ============================================================

print("\n" + "=" * 100)
print("[7] FINAL REFERENCE QC")
print("=" * 100)


sequence_mask = (
    ref["protein_sequence"]
    .notna()
    &
    (
        ref["protein_sequence"]
        .astype(str)
        .str.strip()
        != ""
    )
)

print(
    f"Reference rows     : {len(ref):,}"
)

print(
    f"Sequence available : {sequence_mask.sum():,}"
)

print(
    f"Sequence missing   : {(~sequence_mask).sum():,}"
)

print(
    f"Unique UniProt IDs : "
    f"{ref['uniprot_id'].nunique():,}"
)


# ============================================================
# 9. P02795 / P03928 확인
# ============================================================

print("\nUpdated IDs:")

for uid in UPDATE_IDS:

    rows = ref[
        ref["uniprot_id"] == uid
    ]

    if len(rows) == 0:

        print(
            f"  [ERROR] {uid} not found"
        )

        continue

    seq = rows.iloc[0][
        "protein_sequence"
    ]

    print(
        f"  [OK] {uid} "
        f"| length = {len(str(seq))}"
    )


# ============================================================
# 10. Q2L8D9 / O60361 확인
# ============================================================

print("\nDeleted IDs:")

for uid in DELETE_IDS:

    exists = (
        ref["uniprot_id"] == uid
    ).any()

    print(
        f"  {uid}: "
        f"{'STILL EXISTS' if exists else 'REMOVED'}"
    )


# ============================================================
# 11. Dataset Summary
# ============================================================

print("\n" + "=" * 100)
print("[8] FINAL DATASET SUMMARY")
print("=" * 100)

summary_df = pd.DataFrame(
    dataset_summary
)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 12. Dataset에 삭제 ID가 남아있는지 최종 확인
# ============================================================

print("\nDeleted ID remaining check:")

for dataset_name, output_file in FINAL_DATASET_FILES.items():

    check_df = pd.read_csv(
        output_file,
        usecols=["uniprot_id"],
        low_memory=False
    )

    check_df["uniprot_id"] = (
        check_df["uniprot_id"]
        .apply(normalize_id)
    )

    remaining = check_df[
        check_df["uniprot_id"]
        .isin(DELETE_IDS)
    ]

    print(
        f"  {dataset_name}: "
        f"{len(remaining):,} rows"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 100)
print("FINAL PROCESS COMPLETE")
print("=" * 100)

print("\nGenerated files:")

print(
    f"Reference:\n{FINAL_REFERENCE_FILE}"
)

for dataset_name, output_file in FINAL_DATASET_FILES.items():

    print(
        f"{dataset_name}:\n{output_file}"
    )