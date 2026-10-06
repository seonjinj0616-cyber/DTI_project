import os
import pandas as pd
import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\esm2_active_site_preprocessing"

INPUT_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "final_qc")

os.makedirs(OUTPUT_DIR, exist_ok=True)


FILES = {
    "IC50": os.path.join(INPUT_DIR, "IC50_esm2_ready_final.csv"),
    "Ki":   os.path.join(INPUT_DIR, "Ki_esm2_ready_final.csv"),
    "Kd":   os.path.join(INPUT_DIR, "Kd_esm2_ready_final.csv"),
}


# ============================================================
# QC FUNCTION
# ============================================================

def process_qc(affinity_type, input_path):

    print("\n" + "#" * 100)
    print(f"PROCESSING: {affinity_type}")
    print("#" * 100)

    df = pd.read_csv(input_path)

    print("\n" + "=" * 100)
    print(f"{affinity_type} - QC START")
    print("=" * 100)

    total_before = len(df)

    # --------------------------------------------------------
    # Sequence column 확인
    # --------------------------------------------------------

    if "esm2_sequence" not in df.columns:
        raise ValueError(
            f"[ERROR] esm2_sequence column not found: {affinity_type}"
        )

    if "uniprot_id" not in df.columns:
        raise ValueError(
            f"[ERROR] uniprot_id column not found: {affinity_type}"
        )

    # --------------------------------------------------------
    # 빈 sequence 판정
    # --------------------------------------------------------

    sequence_missing_mask = (
        df["esm2_sequence"].isna()
        | df["esm2_sequence"].astype(str).str.strip().eq("")
        | df["esm2_sequence"].astype(str).str.lower().isin(
            ["nan", "none", "null"]
        )
    )

    sequence_available_mask = ~sequence_missing_mask

    # --------------------------------------------------------
    # Sequence length 계산
    # --------------------------------------------------------

    sequence_length = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    sequence_length.loc[sequence_available_mask] = (
        df.loc[sequence_available_mask, "esm2_sequence"]
        .astype(str)
        .str.len()
    )

    df["_sequence_length_before_qc"] = sequence_length

    # --------------------------------------------------------
    # QC BEFORE FIX
    # --------------------------------------------------------

    sequence_available = int(sequence_available_mask.sum())
    sequence_missing = int(sequence_missing_mask.sum())

    exactly_1024 = int((sequence_length == 1024).sum())
    under_1024 = int((sequence_length < 1024).sum())
    over_1024 = int((sequence_length > 1024).sum())

    print("\n[QC BEFORE FIX]")
    print(f"Total rows                 : {total_before:,}")
    print(f"Sequence available         : {sequence_available:,}")
    print(f"Sequence missing           : {sequence_missing:,}")
    print(f"Exactly 1024 aa            : {exactly_1024:,}")
    print(f"Less than 1024 aa          : {under_1024:,}")
    print(f"Greater than 1024 aa       : {over_1024:,}")

    # --------------------------------------------------------
    # >1024 aa 확인
    # --------------------------------------------------------

    over_1024_mask = sequence_length > 1024

    print(f"\n>1024 aa rows found        : {int(over_1024_mask.sum()):,}")

    if over_1024_mask.any():

        print("\n[>1024 aa length distribution]")

        print(
            df.loc[
                over_1024_mask,
                "_sequence_length_before_qc"
            ].value_counts().sort_index()
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # alignment/crop 시작점은 이미 앞쪽에서 결정되었으므로
        # 여기서는 sequence 앞부분을 건드리지 않고
        # 뒤쪽 residue만 제거
        #
        # 1025 -> first 1024
        # 1030 -> first 1024
        # ----------------------------------------------------

        df.loc[over_1024_mask, "esm2_sequence"] = (
            df.loc[over_1024_mask, "esm2_sequence"]
            .astype(str)
            .str.slice(0, 1024)
        )

    # --------------------------------------------------------
    # 새로운 sequence length
    # --------------------------------------------------------

    final_sequence_missing_mask = (
        df["esm2_sequence"].isna()
        | df["esm2_sequence"].astype(str).str.strip().eq("")
        | df["esm2_sequence"].astype(str).str.lower().isin(
            ["nan", "none", "null"]
        )
    )

    final_sequence_length = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64"
    )

    final_available_mask = ~final_sequence_missing_mask

    final_sequence_length.loc[final_available_mask] = (
        df.loc[final_available_mask, "esm2_sequence"]
        .astype(str)
        .str.len()
    )

    df["_sequence_length_after_qc"] = final_sequence_length

    # --------------------------------------------------------
    # AFTER FIX statistics
    # --------------------------------------------------------

    final_sequence_available = int(final_available_mask.sum())
    final_sequence_missing = int(final_sequence_missing_mask.sum())

    final_exactly_1024 = int(
        (final_sequence_length == 1024).sum()
    )

    final_under_1024 = int(
        (final_sequence_length < 1024).sum()
    )

    final_over_1024 = int(
        (final_sequence_length > 1024).sum()
    )

    trimmed_to_1024 = int(
        (
            (sequence_length > 1024)
            & (final_sequence_length == 1024)
        ).sum()
    )

    print("\n[QC AFTER 1024 FIX]")
    print(f"Total rows                 : {len(df):,}")
    print(f"Sequence available         : {final_sequence_available:,}")
    print(f"Sequence missing           : {final_sequence_missing:,}")
    print(f"Exactly 1024 aa            : {final_exactly_1024:,}")
    print(f"Less than 1024 aa          : {final_under_1024:,}")
    print(f"Greater than 1024 aa       : {final_over_1024:,}")
    print(f"Trimmed to 1024 aa         : {trimmed_to_1024:,}")

    if final_over_1024 == 0:
        print("\nPASS: 최종 sequence에 >1024 aa 데이터가 없습니다.")
    else:
        print("\nFAIL: 아직 >1024 aa sequence가 존재합니다.")

    # ========================================================
    # MISSING PROTEIN SEQUENCE ROW DELETE
    # ========================================================

    missing_rows = df[final_sequence_missing_mask].copy()

    missing_count = len(missing_rows)

    print("\n" + "=" * 100)
    print("REMOVE ROWS WITHOUT PROTEIN SEQUENCE")
    print("=" * 100)

    print(f"Rows without protein sequence : {missing_count:,}")

    if missing_count > 0:

        print("\n[ROWS TO DELETE BY UniProt ID]")

        missing_id_counts = (
            missing_rows["uniprot_id"]
            .value_counts()
            .sort_index()
        )

        print(missing_id_counts)

        # ----------------------------------------------------
        # 실제 삭제
        # ----------------------------------------------------

        df = df.loc[~final_sequence_missing_mask].copy()

        print(
            f"\nDeleted rows                : {missing_count:,}"
        )
        print(
            f"Rows remaining              : {len(df):,}"
        )

    else:

        print("\nNo rows with missing protein sequence.")

    # ========================================================
    # 최종 QC
    # ========================================================

    final_lengths = (
        df["esm2_sequence"]
        .astype(str)
        .str.len()
    )

    final_missing = (
        df["esm2_sequence"].isna()
        | df["esm2_sequence"].astype(str).str.strip().eq("")
        | df["esm2_sequence"].astype(str).str.lower().isin(
            ["nan", "none", "null"]
        )
    )

    final_over_1024_count = int(
        (final_lengths > 1024).sum()
    )

    final_missing_count = int(
        final_missing.sum()
    )

    final_exact_1024_count = int(
        (final_lengths == 1024).sum()
    )

    final_under_1024_count = int(
        (final_lengths < 1024).sum()
    )

    # --------------------------------------------------------
    # UniProt QC
    # --------------------------------------------------------

    total_uniprot_ids = df["uniprot_id"].nunique()

    unresolved_uniprot_ids = (
        df.loc[final_missing, "uniprot_id"]
        .dropna()
        .unique()
    )

    resolved_uniprot_ids = total_uniprot_ids

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("FINAL QC")
    print("=" * 100)

    print(f"Final total rows            : {len(df):,}")
    print(f"Protein sequence missing    : {final_missing_count:,}")
    print(f"Exactly 1024 aa             : {final_exact_1024_count:,}")
    print(f"Under 1024 aa               : {final_under_1024_count:,}")
    print(f"Over 1024 aa                : {final_over_1024_count:,}")
    print(f"Total UniProt IDs           : {total_uniprot_ids:,}")

    # --------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------

    pass_sequence_length = (
        final_over_1024_count == 0
    )

    pass_sequence_missing = (
        final_missing_count == 0
    )

    if pass_sequence_length:
        print(
            "\nPASS 1: 최종 데이터에서 >1024 aa sequence = 0"
        )
    else:
        print(
            "\nFAIL 1: >1024 aa sequence가 존재합니다."
        )

    if pass_sequence_missing:
        print(
            "PASS 2: protein sequence가 없는 row = 0"
        )
    else:
        print(
            f"FAIL 2: protein sequence가 없는 row = "
            f"{final_missing_count:,}"
        )

    # --------------------------------------------------------
    # 임시 QC columns 삭제
    # --------------------------------------------------------

    df.drop(
        columns=[
            "_sequence_length_before_qc",
            "_sequence_length_after_qc"
        ],
        inplace=True,
        errors="ignore"
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{affinity_type}_esm2_final_qc.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    return {
        "affinity_type": affinity_type,
        "total_rows_before": total_before,
        "deleted_missing_sequence_rows": missing_count,
        "final_rows": len(df),
        "sequence_missing": final_missing_count,
        "exactly_1024": final_exact_1024_count,
        "under_1024": final_under_1024_count,
        "over_1024": final_over_1024_count,
        "trimmed_to_1024": trimmed_to_1024,
        "total_uniprot_ids": total_uniprot_ids,
        "pass_over_1024": pass_sequence_length,
        "pass_sequence_complete": pass_sequence_missing,
    }


# ============================================================
# MAIN
# ============================================================

results = []

for affinity_type, input_path in FILES.items():

    result = process_qc(
        affinity_type,
        input_path
    )

    results.append(result)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("FINAL GLOBAL QC SUMMARY")
print("=" * 100)

print(
    summary_df[
        [
            "affinity_type",
            "total_rows_before",
            "deleted_missing_sequence_rows",
            "final_rows",
            "sequence_missing",
            "exactly_1024",
            "under_1024",
            "over_1024",
            "trimmed_to_1024",
            "total_uniprot_ids",
        ]
    ].to_string(index=False)
)

# ============================================================
# GLOBAL PASS / FAIL
# ============================================================

global_over_1024 = int(
    summary_df["over_1024"].sum()
)

global_sequence_missing = int(
    summary_df["sequence_missing"].sum()
)

global_deleted = int(
    summary_df["deleted_missing_sequence_rows"].sum()
)

print("\n" + "=" * 100)
print("GLOBAL QC RESULT")
print("=" * 100)

if global_over_1024 == 0:
    print(
        "PASS 1: 전체 데이터에서 >1024 aa sequence = 0"
    )
else:
    print(
        f"FAIL 1: >1024 aa sequence = "
        f"{global_over_1024:,}"
    )

if global_sequence_missing == 0:
    print(
        "PASS 2: 최종 데이터에서 protein sequence missing = 0"
    )
else:
    print(
        f"FAIL 2: protein sequence missing = "
        f"{global_sequence_missing:,}"
    )

print(
    f"REMOVED: protein sequence missing row = "
    f"{global_deleted:,}"
)

print("\n처리 완료.")