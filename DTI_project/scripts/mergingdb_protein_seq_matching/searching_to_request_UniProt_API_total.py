import os
import pandas as pd

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"

files = [
    "IC50.csv",
    "Ki.csv",
    "Kd.csv"
]

all_ids = set()

for file in files:

    path = os.path.join(DATA_DIR, file)

    df = pd.read_csv(
        path,
        usecols=["uniprot_id"],
        low_memory=False
    )

    ids = (
        df["uniprot_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    ids = ids[ids != ""]

    print(
        f"{file} : "
        f"{len(ids):,} rows / "
        f"{ids.nunique():,} unique UniProt"
    )

    all_ids.update(ids.tolist())


print()
print("=" * 80)
print(
    f"전체 unique UniProt ID : {len(all_ids):,}"
)
print("=" * 80)