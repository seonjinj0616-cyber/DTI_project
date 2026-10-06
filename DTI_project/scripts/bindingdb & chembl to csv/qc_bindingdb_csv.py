import pandas as pd

import os
import numpy as np

INPUT_CSV = "../data/output/bindingdb_dti.csv"

if not os.path.exists(INPUT_CSV):
    print(f"Error: 파일을 찾을 수 없습니다 -> {INPUT_CSV}")
    exit(1)

print(f"bindingdb_dti.csv 품질 검증(QC) 시작: {INPUT_CSV}\n")
df = pd.read_csv(INPUT_CSV, low_memory=False)

total_rows = len(df)
total_cols = len(df.columns)

# 1. FILE & SCHEMA QC
print("=" * 60)
print("1. FILE & SCHEMA QC")
print("=" * 60)
print(f"- 파일 경로: {INPUT_CSV}")
print(f"- 전체 Row 수: {total_rows:,} 개")
print(f"- 전체 Column 수: {total_cols} 개")
print(f"- 컬럼 목록: {list(df.columns)}")
print("- Source 출처 분포:")
for src, cnt in df['source'].value_counts(dropna=False).items():
    print(f"  ├─ {src}: {cnt:,} 건 ({(cnt / total_rows) * 100:.2f}%)")

# 2. DRUG QC
print("\n" + "=" * 60)
print("2. DRUG QC")
print("=" * 60)
chembl_ids = df['chembl_id'].dropna().unique()
bindingdb_ids = df['bindingdb_id'].dropna().unique()
smiles_vals = df['smiles'].dropna().unique()
smiles_missing = df['smiles'].isna().sum()

print(f"- Unique chembl_id 수: {len(chembl_ids):,} 개")
print(f"- Unique bindingdb_id 수: {len(bindingdb_ids):,} 개")
print(f"- Unique SMILES 수: {len(smiles_vals):,} 개")
print(f"- SMILES Missing: {smiles_missing:,} 건 ({(smiles_missing / total_rows) * 100:.2f}%)")

# 3. TARGET QC
print("\n" + "=" * 60)
print("3. TARGET QC")
print("=" * 60)
uniprot_ids = df['uniprot_id'].dropna().unique()
target_names = df['target_name'].dropna().unique()
uniprot_missing = df['uniprot_id'].isna().sum()
human_cnt = (df['organism'] == 'Homo sapiens').sum()

print(f"- Unique UniProt ID 수: {len(uniprot_ids):,} 개")
print(f"- Unique Target Name 수: {len(target_names):,} 개")
print(f"- UniProt ID Missing: {uniprot_missing:,} 건 ({(uniprot_missing / total_rows) * 100:.2f}%)")
print(f"- Human Target 비율: {(human_cnt / total_rows) * 100:.2f}% ({human_cnt:,} 건)")

# 4. AFFINITY QC
print("\n" + "=" * 60)
print("4. AFFINITY QC")
print("=" * 60)
print("- Affinity Type 분포:")
for aff, cnt in df['affinity_type'].value_counts(dropna=False).items():
    print(f"  ├─ {aff}: {cnt:,} 건 ({(cnt / total_rows) * 100:.2f}%)")

print("- Affinity Unit 분포:")
for unit, cnt in df['affinity_unit'].value_counts(dropna=False).items():
    print(f"  ├─ {unit}: {cnt:,} 건 ({(cnt / total_rows) * 100:.2f}%)")

print("- Affinity Relation 분포:")
for rel, cnt in df['affinity_relation'].value_counts(dropna=False).items():
    print(f"  ├─ {rel}: {cnt:,} 건 ({(cnt / total_rows) * 100:.2f}%)")

pchembl = df['pchembl_value'].dropna()
if len(pchembl) > 0:
    print(f"- pChEMBL 값 수치 요약: Min={pchembl.min():.2f}, Median={pchembl.median():.2f}, Max={pchembl.max():.2f}")

# 수치 정합성 검증 (affinity_value nM -> pchembl_value 수식과 실제 pChEMBL 일치 여부)
# relation (=) 데이터
valid_eq = df[
    (df['affinity_relation'] == '=') &
    (df['affinity_unit'] == 'nM') &
    df['affinity_value'].notna() &
    df['pchembl_value'].notna() &
    (df['affinity_value'] > 0)
].copy()

calc = -np.log10(valid_eq['affinity_value'] * 1e-9)
diff = np.abs(calc - valid_eq['pchembl_value'])

print(
    f"Exact (=) pChEMBL 불일치 "
    f"(|diff| > 0.01): {(diff > 0.01).sum():,} 건"
)

# relation (>) 데이터
valid_gt = df[
    (df['affinity_relation'] == '>') &
    (df['affinity_unit'] == 'nM') &
    df['affinity_value'].notna() &
    df['pchembl_value'].notna() &
    (df['affinity_value'] > 0)
].copy()

calc = -np.log10(valid_gt['affinity_value'] * 1e-9)

# 현재 pchembl_value가 boundary와 일치하는지 검사
diff = np.abs(calc - valid_gt['pchembl_value'])

print(
    f"> relation boundary pChEMBL 불일치 "
    f"(|diff| > 0.01): {(diff > 0.01).sum():,} 건"
)

# relation (<) 데이터
valid_lt = df[
    (df['affinity_relation'] == '<') &
    (df['affinity_unit'] == 'nM') &
    df['affinity_value'].notna() &
    df['pchembl_value'].notna() &
    (df['affinity_value'] > 0)
].copy()

calc = -np.log10(valid_lt['affinity_value'] * 1e-9)

diff = np.abs(calc - valid_lt['pchembl_value'])

print(
    f"< relation boundary pChEMBL 불일치 "
    f"(|diff| > 0.01): {(diff > 0.01).sum():,} 건"
)

# 5. DRUG-TARGET PAIR QC
print("\n" + "=" * 60)
print("5. DRUG-TARGET PAIR QC")
print("=" * 60)
valid_dt = df.dropna(subset=['smiles', 'uniprot_id'])
dt_counts = valid_dt.groupby(['smiles', 'uniprot_id']).size()
unique_dt = len(dt_counts)

if unique_dt > 0:
    avg_meas = dt_counts.mean()
    max_meas = dt_counts.max()
    multi_meas = (dt_counts > 1).sum()
    top_pair = dt_counts.idxmax()
    top_pair_cnt = dt_counts.max()

    print(f"- Unique Drug-Target Pairs (SMILES x UniProt): {unique_dt:,} 개")
    print(f"- Pair당 평균 Measurement 수: {avg_meas:.2f} 회")
    print(f"- Pair당 최대 Measurement 수: {max_meas:,} 회")
    print(f"- 다중 Measurement 보유 Pair 수: {multi_meas:,} 개 ({(multi_meas / unique_dt) * 100:.2f}%)")
    print(f"- 가장 많이 반복된 Pair: Target({top_pair[1]}) | 측정 {top_pair_cnt:,} 회")

# 6. PROVENANCE & DUPLICATE QC
print("\n" + "=" * 60)
print("6. PROVENANCE & DUPLICATE QC")
print("=" * 60)
pub_series = df['publication_id'].dropna().astype(str).str.strip()
pub_cnt = pub_series[~pub_series.isin(['nan', '', 'None'])].count()
chembl_has = df['chembl_id'].notna().sum()
uniq_source_rec = df['source_record_id'].nunique()
full_dup = df.duplicated().sum()

print(f"- publication_id (PMID) 보유 비율: {(pub_cnt / total_rows) * 100:.2f}% ({pub_cnt:,} 건)")
print(f"- chembl_id (Ligand) 보유 비율: {(chembl_has / total_rows) * 100:.2f}% ({chembl_has:,} 건)")
print(f"- Unique source_record_id (Reactant_set_id) 수: {uniq_source_rec:,} 개")
print(f"- Complete Full-row Duplicate 수: {full_dup:,} 건 ({(full_dup / total_rows) * 100:.2f}%)")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE: bindingdb_dti.csv Raw QC Finished.")
print("=" * 60)