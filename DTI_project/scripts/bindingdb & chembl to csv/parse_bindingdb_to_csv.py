import pandas as pd
import numpy as np
import os

INPUT_TSV = "../data/bindingdb/BindingDB_All.tsv"
OUTPUT_CSV = "../data/output/bindingdb_dti.csv"


# 1. 람다 함수로 동적 컬럼 매칭 (최소 컬럼 + UniProt 체인)
def match_columns(c):
    use_cols = [
        'BindingDB Reactant_set_id',
        'Ligand SMILES',
        'ChEMBL ID of Ligand',
        'BindingDB MonomerID',
        'Target Name',
        'Target Source Organism According to Curator or DataSource',
        'Number of Protein Chains in Target (>1 implies a multichain complex)',
        'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'PMID'
    ]
    return (c in use_cols) or str(c).startswith('UniProt (SwissProt) Primary ID of Target Chain')


chunk_size = 200000
first_chunk = True

# 기존 파일 존재 시 삭제 후 새로 쓰기
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

print("BindingDB 최소 컬럼 스트리밍 파싱 시작...")

for chunk_idx, chunk in enumerate(
        pd.read_csv(INPUT_TSV, sep="\t", usecols=match_columns, chunksize=chunk_size, on_bad_lines='skip',
                    low_memory=False)):

    # 2. 인간 단백질 필터링
    if 'Target Source Organism According to Curator or DataSource' in chunk.columns:
        chunk = chunk[chunk['Target Source Organism According to Curator or DataSource'] == 'Homo sapiens']

    # 3. 다중 체인 복합체 제외 (단일 단백질 타겟만 유지)
    # 복합체를 별개의 측정치로 잘못 쪼개는 것을 방지
    if 'Number of Protein Chains in Target (>1 implies a multichain complex)' in chunk.columns:
        chunk['Number of Protein Chains in Target (>1 implies a multichain complex)'] = pd.to_numeric(chunk['Number of Protein Chains in Target (>1 implies a multichain complex)'],
                                                                    errors='coerce')
        chunk = chunk[chunk['Number of Protein Chains in Target (>1 implies a multichain complex)'] == 1]

    chunk = chunk.dropna(subset=['Ligand SMILES'])
    if len(chunk) == 0: continue

    # 4. UniProt ID 추출 (N=1이므로 첫 번째 유효한 UniProt ID만 병합)
    uniprot_cols = [c for c in chunk.columns if str(c).startswith(
        'UniProt (SwissProt) Primary ID of Target Chain')]

    if len(uniprot_cols) == 0: continue

    # single-chain target이므로 첫 번째 chain ID 사용
    chunk['uniprot_id'] = chunk[uniprot_cols[0]]
    chunk = chunk.dropna(subset=['uniprot_id'])

    if len(chunk) == 0:
        continue

    # 5. Affinity 컬럼 Melt (Ki, IC50, Kd)
    affinity_cols = [c for c in ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)'] if c in chunk.columns]
    id_vars = [c for c in chunk.columns if c not in affinity_cols]

    melted = pd.melt(chunk, id_vars=id_vars, value_vars=affinity_cols, var_name='affinity_type_raw',
                     value_name='affinity_str')
    melted = melted.dropna(subset=['affinity_str'])
    melted = melted[melted['affinity_str'].astype(str).str.strip() != '']
    if len(melted) == 0: continue

    # 6. 부등호 및 수치 정제, pChEMBL 계산
    melted['affinity_str'] = melted['affinity_str'].astype(str).str.strip()
    melted['affinity_relation'] = '='
    melted.loc[melted['affinity_str'].str.startswith('>'), 'affinity_relation'] = '>'
    melted.loc[melted['affinity_str'].str.startswith('<'), 'affinity_relation'] = '<'

    clean_vals = melted['affinity_str'].str.replace('>', '', regex=False).str.replace('<', '', regex=False).str.strip()
    melted['affinity_value'] = pd.to_numeric(clean_vals, errors='coerce')
    melted = melted.dropna(subset=['affinity_value'])
    melted = melted[melted['affinity_value'] > 0]
    if len(melted) == 0: continue

    melted['affinity_type'] = melted['affinity_type_raw'].str.split(' ').str[0]
    # nM -> M로 변환 후, log scale 변환
    melted['pchembl_value'] = -np.log10(melted['affinity_value'] * 1e-9)

    # 7. 표준 스키마 매핑 및 ID 교정
    result_chunk = pd.DataFrame({
        'source': 'BindingDB',
        'source_record_id': melted['BindingDB Reactant_set_id'].astype(str),  # Reaction 고유 ID
        'chembl_id': melted.get('ChEMBL ID of Ligand', None),
        'bindingdb_id': melted['BindingDB MonomerID'].astype(str),  # Ligand 고유 ID
        'smiles': melted['Ligand SMILES'],
        'target_chembl_id': None,
        'uniprot_id': melted['uniprot_id'],
        'target_name': melted.get('Target Name', None),
        'target_type': 'SINGLE PROTEIN',  # ChEMBL과 일치
        'organism': 'Homo sapiens',
        'assay_id': None,
        'assay_type': 'B',
        'affinity_type': melted['affinity_type'],
        'affinity_relation': melted['affinity_relation'],
        'affinity_value': melted['affinity_value'],
        'affinity_unit': 'nM',
        'pchembl_value': melted['pchembl_value'],
        'publication_id': melted.get('PMID', None).astype(str)
    })

    # 8. CSV 파일에 직접 Append (메모리 최적화)
    mode = 'w' if first_chunk else 'a'
    result_chunk.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", mode=mode, header=first_chunk)
    first_chunk = False
    print(f"청크 {chunk_idx} 처리 완료...")

print("최종 저장 완료:", OUTPUT_CSV)