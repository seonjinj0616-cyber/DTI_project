import os
import re
import pandas as pd
from rdkit import Chem

# ============================================================
# PATH
# ============================================================

INPUT_PATH = "../../data/output/merged_dti_raw.csv"

OUTPUT_PATH = "../../data/output/merged_dti_smiles_revalidated.csv"

CHUNK_SIZE = 200_000


# ============================================================
# BindingDB annotation 제거
# ============================================================

def clean_bindingdb_smiles(smiles):
    """
    BindingDB SMILES 뒤에 붙은 annotation 제거.

    예:
    CN... |TLB:10:9:6.7:20.15,...|
    ->
    CN...

    CN... |THB:9:7:1:3.4|
    ->
    CN...
    """

    if pd.isna(smiles):
        return smiles

    smiles = str(smiles).strip()

    # BindingDB annotation이 붙은 경우에만 제거
    # |TLB:...| / |THB:...|
    cleaned = re.sub(
        r"\s*\|(?:TLB|THB):.*?\|\s*$",
        "",
        smiles
    )

    return cleaned.strip()


# ============================================================
# RDKit validation
# ============================================================

def is_valid_smiles(smiles):
    """
    RDKit으로 SMILES 유효성 검사
    """

    if pd.isna(smiles):
        return False

    smiles = str(smiles).strip()

    if not smiles:
        return False

    mol = Chem.MolFromSmiles(smiles)

    return mol is not None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("BindingDB Annotation Removal + SMILES Revalidation")
    print("=" * 90)

    # 기존 결과 파일이 있으면 삭제
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    total_rows = 0

    bindingdb_rows = 0
    annotation_rows = 0

    valid_before = 0
    invalid_before = 0

    valid_after = 0
    invalid_after = 0

    first_write = True

    # ========================================================
    # Chunk processing
    # ========================================================

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            chunksize=CHUNK_SIZE,
            dtype=str
        )
    ):

        print(f"\nChunk {chunk_idx}: {len(chunk):,} rows")

        total_rows += len(chunk)

        # ----------------------------------------------------
        # BindingDB만 대상
        # ----------------------------------------------------

        bindingdb_mask = (
            chunk["source"]
            .fillna("")
            .str.strip()
            .eq("BindingDB")
        )

        bindingdb_rows += bindingdb_mask.sum()

        # ----------------------------------------------------
        # 기존 SMILES validation
        # ----------------------------------------------------

        before_valid = chunk["smiles"].apply(is_valid_smiles)

        valid_before += before_valid.sum()
        invalid_before += (~before_valid).sum()

        # ----------------------------------------------------
        # BindingDB annotation 제거
        # ----------------------------------------------------

        original_smiles = chunk["smiles"].copy()

        chunk.loc[bindingdb_mask, "smiles"] = (
            chunk.loc[bindingdb_mask, "smiles"]
            .apply(clean_bindingdb_smiles)
        )

        # annotation이 실제로 제거된 행
        changed_mask = (
            original_smiles.astype(str)
            != chunk["smiles"].astype(str)
        )

        annotation_rows += (
            changed_mask & bindingdb_mask
        ).sum()

        # ----------------------------------------------------
        # 다시 RDKit validation
        # ----------------------------------------------------

        after_valid = chunk["smiles"].apply(is_valid_smiles)

        valid_after += after_valid.sum()
        invalid_after += (~after_valid).sum()

        # ----------------------------------------------------
        # 진행상황
        # ----------------------------------------------------

        print(
            f"  BindingDB rows       : {bindingdb_mask.sum():,}"
        )

        print(
            f"  Annotation removed   : "
            f"{(changed_mask & bindingdb_mask).sum():,}"
        )

        print(
            f"  Invalid before       : "
            f"{(~before_valid).sum():,}"
        )

        print(
            f"  Invalid after        : "
            f"{(~after_valid).sum():,}"
        )

        # ----------------------------------------------------
        # 결과 저장
        # ----------------------------------------------------

        chunk.to_csv(
            OUTPUT_PATH,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    print(f"Total rows             : {total_rows:,}")

    print(f"BindingDB rows         : {bindingdb_rows:,}")

    print(f"Annotation removed     : {annotation_rows:,}")

    print("\n[Before annotation removal]")

    print(f"Valid SMILES           : {valid_before:,}")
    print(f"Invalid SMILES         : {invalid_before:,}")

    before_ratio = (
        invalid_before / total_rows * 100
        if total_rows > 0 else 0
    )

    print(f"Invalid ratio          : {before_ratio:.4f}%")

    print("\n[After annotation removal]")

    print(f"Valid SMILES           : {valid_after:,}")
    print(f"Invalid SMILES         : {invalid_after:,}")

    after_ratio = (
        invalid_after / total_rows * 100
        if total_rows > 0 else 0
    )

    print(f"Invalid ratio          : {after_ratio:.4f}%")

    print("\n[Improvement]")

    recovered = valid_after - valid_before

    print(f"Recovered SMILES       : {recovered:,}")

    print("\n")
    print("=" * 90)
    print("OUTPUT")
    print("=" * 90)

    print(OUTPUT_PATH)

    print("=" * 90)


if __name__ == "__main__":
    main()