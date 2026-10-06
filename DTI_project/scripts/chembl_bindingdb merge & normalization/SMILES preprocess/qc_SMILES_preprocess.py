import pandas as pd
from rdkit import Chem
from collections import Counter

INPUT_PATH = "../../data/output/merged_dti_smiles_cleaned.csv"

print("=" * 100)
print("REMAINING INVALID SMILES ANALYSIS")
print("=" * 100)

df = pd.read_csv(INPUT_PATH, dtype=str)

invalid_rows = []

for idx, row in df.iterrows():
    smiles = row["smiles"]

    if pd.isna(smiles) or not smiles.strip():
        invalid_rows.append({
            "row_index": idx,
            "source": row["source"],
            "source_record_id": row["source_record_id"],
            "smiles": smiles,
            "error_type": "EMPTY"
        })
        continue

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        invalid_rows.append({
            "row_index": idx,
            "source": row["source"],
            "source_record_id": row["source_record_id"],
            "smiles": smiles,
            "error_type": "PARSE_FAILED"
        })
        continue

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        error_msg = str(e)

        if "valence" in error_msg.lower():
            error_type = "VALENCE_ERROR"
        elif "kekulize" in error_msg.lower():
            error_type = "KEKULIZE_ERROR"
        elif "charge" in error_msg.lower():
            error_type = "CHARGE_ERROR"
        else:
            error_type = "SANITIZE_ERROR"

        invalid_rows.append({
            "row_index": idx,
            "source": row["source"],
            "source_record_id": row["source_record_id"],
            "smiles": smiles,
            "error_type": error_type
        })


# ============================================================================
# RESULT
# ============================================================================

invalid_df = pd.DataFrame(invalid_rows)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"전체 데이터       : {len(df):,}")
print(f"남은 invalid      : {len(invalid_df):,}")

print()
print("[Invalid Type]")
print(invalid_df["error_type"].value_counts())


print()
print("=" * 100)
print("SOURCE")
print("=" * 100)

print(invalid_df["source"].value_counts())


print()
print("=" * 100)
print("INVALID SMILES SAMPLE")
print("=" * 100)

pd.set_option("display.max_colwidth", 300)

print(
    invalid_df[
        [
            "row_index",
            "source",
            "source_record_id",
            "error_type",
            "smiles"
        ]
    ].head(50).to_string(index=False)
)


print()
print("=" * 100)
print("DONE")
print("=" * 100)