import os
import re
import pandas as pd

def inspect_11_duplicates(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 파일이 존재하지 않습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    wt_df = df[df['type'] == 'wildtype'].copy()
    mut_df = df[df['type'] == 'mutant'].copy()

    # ID 파싱
    wt_df['clean_id'] = wt_df['id'].astype(str).str.replace(r'^WT_', '', regex=True).str.strip()
    wt_df['base_id'] = wt_df['clean_id'].apply(lambda x: x.split('-')[0].split('.')[0])

    def parse_mut_base(mut_str):
        clean = re.sub(r'^WT_', '', str(mut_str)).strip()
        uid = re.split(r'[_:/\s]', clean)[0]
        return uid.split('-')[0].split('.')[0]

    mut_base_uids = set(mut_df['id'].apply(parse_mut_base))

    # WT 그룹 분리
    paired_wt_df = wt_df[wt_df['base_id'].isin(mut_base_uids)].copy()
    pure_wt_df = wt_df[~wt_df['base_id'].isin(mut_base_uids)].copy()

    # 서열 -> ID 맵핑 딕셔너리 생성
    paired_seq_to_ids = paired_wt_df.groupby('sequence')['id'].apply(list).to_dict()

    print("==========================================")
    print("🔍 [서열 100% 중복 Pure WT 11개 세부 조회]")
    print("==========================================")

    count = 0
    for idx, row in pure_wt_df.iterrows():
        seq = row['sequence']
        if seq in paired_seq_to_ids:
            count += 1
            matching_paired_ids = paired_seq_to_ids[seq]
            print(f"\n[{count}] Pure WT ID: {row['id']} (길이: {row['length']}aa)")
            print(f"    └─ 서열 100% 일치하는 Paired WT ID: {matching_paired_ids}")

    print("\n==========================================")

if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    target_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")
    inspect_11_duplicates(target_csv)