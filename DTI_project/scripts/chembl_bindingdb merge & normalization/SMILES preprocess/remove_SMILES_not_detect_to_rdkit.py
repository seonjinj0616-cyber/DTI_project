import pandas as pd
from rdkit import Chem
import os

input_path = "../../../data/output/merged_dti_PRE_SMILES1.csv"
output_path = "../../../data/output/merged_dti_final.csv"

chunksize = 200_000

total_rows = 0
total_deleted = 0
total_valid = 0
total_missing = 0
total_parse_failed = 0

first_chunk = True

for chunk_idx, chunk in enumerate(
    pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
):

    print(f"\n{'='*70}")
    print(f"Chunk {chunk_idx + 1}")
    print(f"{'='*70}")

    delete_mask = []

    for smiles in chunk["smiles"]:

        # 결측
        if pd.isna(smiles) or str(smiles).strip() == "":
            total_missing += 1
            delete_mask.append(False)
            continue

        smiles = str(smiles).strip()

        # RDKit parsing
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # VALENCE_ERROR / KEKULIZE_ERROR 포함
            total_parse_failed += 1
            delete_mask.append(True)
        else:
            total_valid += 1
            delete_mask.append(False)

    delete_mask = pd.Series(delete_mask, index=chunk.index)

    deleted = int(delete_mask.sum())
    total_deleted += deleted
    total_rows += len(chunk)

    # 잘못된 SMILES 삭제
    cleaned_chunk = chunk.loc[~delete_mask]

    cleaned_chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    print(f"전체 행       : {len(chunk):,}")
    print(f"VALID         : {len(chunk) - deleted:,}")
    print(f"삭제          : {deleted:,}")

print("\n" + "="*70)
print("최종 결과")
print("="*70)

print(f"전체 행                  : {total_rows:,}")
print(f"VALID                    : {total_valid:,}")
print(f"MISSING                  : {total_missing:,}")
print(f"PARSE_FAILED             : {total_parse_failed:,}")
print(f"삭제                     : {total_deleted:,}")
print(f"최종 저장                : {total_rows - total_deleted:,}")

print(f"\n저장 위치: {output_path}")