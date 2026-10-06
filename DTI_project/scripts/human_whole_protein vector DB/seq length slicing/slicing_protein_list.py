import os
import re
import pandas as pd


def audit_and_list_over_1024_sequences(input_csv, output_dir):
    if not os.path.exists(input_csv):
        print(f"❌ [오류] 파일이 존재하지 않습니다: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    print("==========================================")
    print(f"📖 데이터 로드 완료: 총 {len(df):,}개 서열")
    print("==========================================")

    # ID 파싱 및 그룹화
    def extract_canonical_id(raw_id, seq_type):
        clean_id = re.sub(r'^WT_', '', str(raw_id)).strip()
        if seq_type == 'wildtype':
            return clean_id.split('-')[0]
        else:
            return clean_id.split('_')[0].split('-')[0]

    df['canonical_id'] = df.apply(lambda r: extract_canonical_id(r['id'], r['type']), axis=1)

    # WT / Mutant 구분
    wt_df = df[df['type'] == 'wildtype'].copy()
    mutant_df = df[df['type'] == 'mutant'].copy()

    mutant_canonical_ids = set(mutant_df['canonical_id'])

    # 1024aa 초과 데이터 필터링
    over_1024_df = df[df['length'] > 1024].copy()

    # 1024aa 초과 그룹별 세부 분류
    over_1024_mutants = over_1024_df[over_1024_df['type'] == 'mutant']
    over_1024_wts = over_1024_df[over_1024_df['type'] == 'wildtype']

    # WT 중 Standalone(0~1024 절단) vs Paired(변이 위치 기준 절단) 구분
    over_1024_standalone_wts = over_1024_wts[~over_1024_wts['canonical_id'].isin(mutant_canonical_ids)].copy()
    over_1024_paired_wts = over_1024_wts[over_1024_wts['canonical_id'].isin(mutant_canonical_ids)].copy()

    # 절단 방식 태깅
    over_1024_df['crop_strategy'] = 'Unassigned'
    over_1024_df.loc[over_1024_df['type'] == 'mutant', 'crop_strategy'] = 'Mutant Window (Center on Mutation)'
    over_1024_df.loc[
        over_1024_df['id'].isin(over_1024_paired_wts['id']), 'crop_strategy'] = 'WT Paired Window (Match Mutant)'
    over_1024_df.loc[
        over_1024_df['id'].isin(over_1024_standalone_wts['id']), 'crop_strategy'] = 'WT Standalone (0 ~ 1024aa)'

    print("📊 [1,024aa 초과 절단 대상 종합 통계]")
    print(f"• 전체 1,024aa 초과 서열 개수: {len(over_1024_df):,}개 (전체 서열의 {len(over_1024_df) / len(df) * 100:.2f}%)")
    print("------------------------------------------")
    print(f"  1. Mutant 서열 (변이 위치 기준 절단): {len(over_1024_mutants):,}개")
    print(f"  2. Paired WT 서열 (Mutant 대응 구간 절단): {len(over_1024_paired_wts):,}개")
    print(f"  3. Standalone WT 서열 (앞쪽 0~1024aa 절단): {len(over_1024_standalone_wts):,}개")
    print("==========================================")

    # 출력 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # CSV 저장 1: 전체 1024aa 초과 목록
    total_list_path = os.path.join(output_dir, "all_over_1024_sequences_list.csv")
    over_1024_df.to_csv(total_list_path, index=False, encoding='utf-8-sig')

    # CSV 저장 2: Standalone 0~1024aa 절단 대상 WT 전용 목록
    standalone_list_path = os.path.join(output_dir, "standalone_wt_sliced_0_to_1024.csv")
    over_1024_standalone_wts.to_csv(standalone_list_path, index=False, encoding='utf-8-sig')

    print(f"✅ [전체 1024aa 초과 목록 저장 완료]: {total_list_path}")
    print(f"✅ [0~1024aa WT 전용 목록 저장 완료]: {standalone_list_path}")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    input_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")

    audit_and_list_over_1024_sequences(input_csv, base_dir)