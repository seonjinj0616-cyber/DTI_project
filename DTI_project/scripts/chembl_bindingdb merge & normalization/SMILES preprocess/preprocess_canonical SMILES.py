import os
import pandas as pd
from rdkit import Chem


INPUT_PATH = "../../data/output/merged_dti_raw.csv"
CHUNK_SIZE = 200_000

INVALID_SMILES_OUTPUT = (
    "../../data/output/qc_invalid_smiles.csv"
)

UNIPROT_COUNT_OUTPUT = (
    "../../data/output/qc_uniprot_counts.csv"
)


# ============================================================
# SMILES
# ============================================================

def canonicalize_smiles(smiles):

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return None

def smiles_qc(df):
    """
    SMILES validation 및 canonicalization QC.
    삭제하지 않고 실패 행과 통계를 확인한다.
    """

    print("\n" + "=" * 90)
    print("SMILES QC")
    print("=" * 90)

    total = len(df)

    invalid_rows = []

    canonical_smiles = []

    for idx, smiles in df["smiles"].items():

        if pd.isna(smiles):
            canonical_smiles.append(None)

            invalid_rows.append({
                "row_index": idx,
                "smiles": smiles,
                "reason": "missing"
            })

            continue

        smiles = str(smiles).strip()

        if not smiles:
            canonical_smiles.append(None)

            invalid_rows.append({
                "row_index": idx,
                "smiles": smiles,
                "reason": "empty"
            })

            continue

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                canonical_smiles.append(None)

                invalid_rows.append({
                    "row_index": idx,
                    "smiles": smiles,
                    "reason": "RDKit_parse_failed"
                })

                continue

            canonical = Chem.MolToSmiles(
                mol,
                canonical=True
            )

            canonical_smiles.append(canonical)

        except Exception as e:

            canonical_smiles.append(None)

            invalid_rows.append({
                "row_index": idx,
                "smiles": smiles,
                "reason": f"exception: {str(e)}"
            })


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    df["canonical_smiles"] = canonical_smiles

    invalid_df = pd.DataFrame(invalid_rows)

    invalid_count = len(invalid_df)
    valid_count = total - invalid_count

    invalid_ratio = (
        invalid_count / total * 100
        if total > 0 else 0
    )

    print(f"전체 row              : {total:,}")
    print(f"Canonical SMILES 성공 : {valid_count:,}")
    print(f"Canonical SMILES 실패 : {invalid_count:,}")
    print(f"실패율                 : {invalid_ratio:.4f}%")

    # --------------------------------------------------------
    # 실패 SMILES 출력
    # --------------------------------------------------------

    if invalid_count > 0:

        print("\n[실패 원인]")
        print(
            invalid_df["reason"]
            .value_counts()
            .to_string()
        )

        print("\n[실패 SMILES 예시]")

        print(
            invalid_df.head(20).to_string(index=False)
        )

    else:

        print("\nCanonical SMILES 변환 실패 없음.")

    return df, invalid_df
# ============================================================
# UniProt
# ============================================================

def validate_uniprot(uniprot):

    if pd.isna(uniprot):
        return False

    uniprot = str(uniprot).strip()

    if not uniprot:
        return False

    if len(uniprot) < 5 or len(uniprot) > 12:
        return False

    allowed = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789_-"
    )

    return all(
        c in allowed
        for c in uniprot
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SMILES + UniProt QC")
    print("=" * 90)

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            INPUT_PATH
        )


    # --------------------------------------------------------
    # counters
    # --------------------------------------------------------

    total_rows = 0

    smiles_valid = 0
    smiles_invalid = 0

    smiles_invalid_rows = []

    uniprot_valid = 0
    uniprot_invalid = 0
    uniprot_missing = 0

    uniprot_counter = {}


    # ========================================================
    # CHUNK
    # ========================================================

    for chunk_idx, df in enumerate(
        pd.read_csv(
            INPUT_PATH,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        print(
            f"\nChunk {chunk_idx}: "
            f"{len(df):,} rows"
        )

        total_rows += len(df)


        # ====================================================
        # SMILES
        # ====================================================

        for row_idx, smiles in df["smiles"].items():

            canonical = canonicalize_smiles(
                smiles
            )

            if canonical is None:

                smiles_invalid += 1

                smiles_invalid_rows.append({
                    "row_index": row_idx,
                    "source": df.loc[row_idx, "source"],
                    "source_record_id":
                        df.loc[row_idx, "source_record_id"],
                    "smiles": smiles
                })

            else:

                smiles_valid += 1


        # ====================================================
        # UniProt
        # ====================================================

        for uniprot in df["uniprot_id"]:

            if pd.isna(uniprot):

                uniprot_missing += 1
                continue

            uniprot = str(uniprot).strip()

            if not uniprot:

                uniprot_missing += 1
                continue

            if validate_uniprot(uniprot):

                uniprot_valid += 1

                uniprot_counter[uniprot] = (
                    uniprot_counter.get(
                        uniprot,
                        0
                    ) + 1
                )

            else:

                uniprot_invalid += 1


        print(
            f"  SMILES invalid: "
            f"{smiles_invalid:,}"
        )

        print(
            f"  Unique UniProt so far: "
            f"{len(uniprot_counter):,}"
        )


    # ========================================================
    # FINAL SMILES QC
    # ========================================================

    print("\n")
    print("=" * 90)
    print("SMILES RESULT")
    print("=" * 90)

    print(
        f"Total rows        : {total_rows:,}"
    )

    print(
        f"Valid             : {smiles_valid:,}"
    )

    print(
        f"Invalid           : {smiles_invalid:,}"
    )

    print(
        f"Invalid ratio     : "
        f"{smiles_invalid / total_rows * 100:.4f}%"
    )


    # --------------------------------------------------------
    # 실패 SMILES 저장
    # --------------------------------------------------------

    if smiles_invalid_rows:

        invalid_smiles_df = pd.DataFrame(
            smiles_invalid_rows
        )

        invalid_smiles_df.to_csv(
            INVALID_SMILES_OUTPUT,
            index=False
        )

        print(
            f"\nInvalid SMILES saved:"
            f"\n{INVALID_SMILES_OUTPUT}"
        )

    else:

        print(
            "\nInvalid SMILES 없음."
        )


    # ========================================================
    # UNIPROT RESULT
    # ========================================================

    print("\n")
    print("=" * 90)
    print("UNIPROT RESULT")
    print("=" * 90)

    unique_uniprot = len(
        uniprot_counter
    )

    total_non_missing = (
        uniprot_valid
    )

    duplicate_rows = (
        total_non_missing
        - unique_uniprot
    )

    print(
        f"Total rows                  : "
        f"{total_rows:,}"
    )

    print(
        f"Valid UniProt rows          : "
        f"{uniprot_valid:,}"
    )

    print(
        f"Invalid UniProt rows        : "
        f"{uniprot_invalid:,}"
    )

    print(
        f"Missing UniProt             : "
        f"{uniprot_missing:,}"
    )

    print(
        f"Unique UniProt              : "
        f"{unique_uniprot:,}"
    )

    print(
        f"Duplicate UniProt rows     : "
        f"{duplicate_rows:,}"
    )


    # ========================================================
    # UNIPROT COUNT TABLE
    # ========================================================

    uniprot_count_df = (
        pd.DataFrame(
            [
                {
                    "uniprot_id": k,
                    "row_count": v
                }
                for k, v in uniprot_counter.items()
            ]
        )
        .sort_values(
            "row_count",
            ascending=False
        )
    )

    uniprot_count_df.to_csv(
        UNIPROT_COUNT_OUTPUT,
        index=False
    )


    print(
        f"\nUniProt count table saved:"
        f"\n{UNIPROT_COUNT_OUTPUT}"
    )


    print("\n[Top 20 UniProt]")

    print(
        uniprot_count_df
        .head(20)
        .to_string(index=False)
    )


    print("\n")
    print("=" * 90)
    print("QC COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()