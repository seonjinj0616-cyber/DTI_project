import os
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\final_final"

KI_PATH = os.path.join(BASE_DIR, "Ki_ems2_fi.csv")
KD_PATH = os.path.join(BASE_DIR, "Kd_ems2_fi.csv")


# ============================================================
# Load
# ============================================================

ki = pd.read_csv(
    KI_PATH,
    usecols=["smiles", "uniprot_id", "affinity_type"]
)

kd = pd.read_csv(
    KD_PATH,
    usecols=["smiles", "uniprot_id", "affinity_type"]
)


# ============================================================
# Normalize
# ============================================================

for df in [ki, kd]:
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["affinity_type"] = df["affinity_type"].astype(str).str.upper().str.strip()


# ============================================================
# Check affinity_type
# ============================================================

print("=" * 80)
print("Affinity type check")
print("=" * 80)

print("\nKi:")
print(ki["affinity_type"].value_counts())

print("\nKd:")
print(kd["affinity_type"].value_counts())


# ============================================================
# Unique (SMILES, UniProt) pairs
# ============================================================

ki_pairs = ki[
    ["smiles", "uniprot_id"]
].drop_duplicates()

kd_pairs = kd[
    ["smiles", "uniprot_id"]
].drop_duplicates()


# ============================================================
# Ki ∩ Kd
# 같은 SMILES + 같은 UniProt이 Ki와 Kd 모두 존재
# ============================================================

common = pd.merge(
    ki_pairs,
    kd_pairs,
    on=["smiles", "uniprot_id"],
    how="inner"
)


# ============================================================
# Ki ONLY
# Ki에는 있지만 Kd에는 없는 interaction
# ============================================================

ki_only = pd.merge(
    ki_pairs,
    kd_pairs,
    on=["smiles", "uniprot_id"],
    how="left",
    indicator=True
)

ki_only = ki_only[
    ki_only["_merge"] == "left_only"
].drop(columns="_merge")


# ============================================================
# Kd ONLY
# Kd에는 있지만 Ki에는 없는 interaction
# ============================================================

kd_only = pd.merge(
    kd_pairs,
    ki_pairs,
    on=["smiles", "uniprot_id"],
    how="left",
    indicator=True
)

kd_only = kd_only[
    kd_only["_merge"] == "left_only"
].drop(columns="_merge")


# ============================================================
# Result
# ============================================================

print("\n")
print("=" * 80)
print("Ki / Kd SMILES + UniProt overlap")
print("=" * 80)

print(f"Ki unique (SMILES + UniProt) : {len(ki_pairs):,}")
print(f"Kd unique (SMILES + UniProt) : {len(kd_pairs):,}")
print()

print(f"Ki + Kd BOTH                 : {len(common):,}")
print(f"Ki ONLY                      : {len(ki_only):,}")
print(f"Kd ONLY                      : {len(kd_only):,}")
print()

print(
    f"Ki overlap ratio             : "
    f"{len(common) / len(ki_pairs) * 100:.2f}%"
)

print(
    f"Kd overlap ratio             : "
    f"{len(common) / len(kd_pairs) * 100:.2f}%"
)

print("=" * 80)