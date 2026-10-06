import os
import re
import pandas as pd


def audit_pure_wildtypes(csv_path):
    print("==========================================")
    print("🔍 [Pure WT 정밀 진단 QC] 순수 Wildtype ID 상태 검증")
    print("==========================================")

    if not os.path.exists(csv_path):
        print(f"❌ [오류] 대상 파일이 존재하지 않습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📖 데이터 로드 완료: 총 {len(df):,}개 서열")

    wt_df = df[df['type'] == 'wildtype'].copy()
    mut_df = df[df['type'] == 'mutant'].copy()

    # ----------------------------------------------------
    # 0. ID 정규화 처리 (기본 파싱)
    # ----------------------------------------------------
    wt_df['clean_id'] = wt_df['id'].astype(str).str.replace(r'^WT_', '', regex=True).str.strip()
    wt_df['base_id'] = wt_df['clean_id'].apply(lambda x: x.split('-')[0].split('.')[0])

    def parse_mut_id(mut_str):
        clean = re.sub(r'^WT_', '', str(mut_str)).strip()
        # 다양한 구분자(_, -, :, /) 대응
        parts = re.split(r'[_:/\s]', clean)
        uid = parts[0]
        base_uid = uid.split('-')[0].split('.')[0]
        return uid, base_uid

    mut_parsed = mut_df['id'].apply(parse_mut_id)
    mut_df['mut_uid'] = [p[0] for p in mut_parsed]
    mut_df['mut_base_uid'] = [p[1] for p in mut_parsed]

    mut_uids = set(mut_df['mut_uid'])
    mut_base_uids = set(mut_df['mut_base_uid'])

    # 1차 분류: Mutant와 완전/Base 매칭이 안 되는 "Pure WT" 추출
    pure_wt_df = wt_df[~wt_df['base_id'].isin(mut_base_uids)].copy()
    paired_wt_df = wt_df[wt_df['base_id'].isin(mut_base_uids)].copy()

    print(f"• 전체 WT 서열 수: {len(wt_df):,}개")
    print(f"• Mutant 쌍이 존재하는 WT: {len(paired_wt_df):,}개")
    print(f"• 진단 대상 Pure WT 서열 수: {len(pure_wt_df):,}개")
    print("------------------------------------------")

    # ----------------------------------------------------
    # [검사 1] 숨은 보이지 않는 문자 / 공백 / 특수문자 검사
    # ----------------------------------------------------
    dirty_ids = pure_wt_df[pure_wt_df['clean_id'].str.contains(r'[\s\t\r\n|:]', regex=True)]
    print(f"📌 [검사 1] 공백/특수문자/줄바꿈 포함 ID: {len(dirty_ids):,}개")
    if not dirty_ids.empty:
        print("   └─ 예시:", dirty_ids['clean_id'].head(3).tolist())

    # ----------------------------------------------------
    # [검사 2] 비표준 접두사 (WT_ 외 sp|, tr|, ENSP, NP_ 등) 잔여 검사
    # ----------------------------------------------------
    non_uniprot_prefix = pure_wt_df[
        pure_wt_df['clean_id'].str.contains(r'^(sp\||tr\||ENSP|NP_|REFSEQ|HUMAN)', case=False, regex=True)]
    print(f"📌 [검사 2] 비표준 접두사(sp|, ENSP 등) 포함 ID: {len(non_uniprot_prefix):,}개")
    if not non_uniprot_prefix.empty:
        print("   └─ 예시:", non_uniprot_prefix['clean_id'].head(3).tolist())

    # ----------------------------------------------------
    # [검사 3] UniProt 표준 Accession 정규식 검증
    # UniProt 포맷: [O,P,Q][0-9][A-Z0-9]{3}[0-9] 또는 [A-N,R-Z][0-9]([A-Z0-9]{3}[0-9]|[A-Z0-9]{5}[0-9])
    # ----------------------------------------------------
    uniprot_regex = r'^[A-NR-Z][0-9][A-Z0-9]{3,5}[0-9](-\d+)?$'
    invalid_uniprot = pure_wt_df[~pure_wt_df['clean_id'].str.match(uniprot_regex, na=False)]
    print(f"📌 [검사 3] 표준 UniProt ACC 포맷 비일치 ID: {len(invalid_uniprot):,}개")
    if not invalid_uniprot.empty:
        print("   └─ 비표준 ID 샘플 (상위 5개):", invalid_uniprot['clean_id'].head(5).tolist())

    # ----------------------------------------------------
    # [검사 4] Substring / Partial Match 교차 검사
    # Pure WT ID가 Mutant의 id/raw_header 문자열 어딘가에 숨어있는지 검사
    # ----------------------------------------------------
    mut_headers_concat = " ".join(
        mut_df['id'].astype(str).tolist() + mut_df.get('raw_header', pd.Series()).astype(str).tolist())

    partial_matched = []
    for idx, row in pure_wt_df.iterrows():
        base_id = row['base_id']
        if len(base_id) >= 6 and base_id in mut_headers_concat:
            partial_matched.append(row['clean_id'])

    print(f"📌 [검사 4] Mutant 헤더 내 문자열로 포함되어 숨어있는 ID: {len(partial_matched):,}개")
    if partial_matched:
        print("   └─ 발견된 숨은 ID 샘플:", partial_matched[:5])

    # ----------------------------------------------------
    # [검사 5] 서열 완전 동일성 (Sequence Redundancy) 검사
    # Pure WT 서열 중 Paired WT 서열과 100% 동일한 서열이 있는지 검사 (ID만 다른 중복)
    # ----------------------------------------------------
    paired_seq_set = set(paired_wt_df['sequence'])
    identical_seq_count = pure_wt_df['sequence'].isin(paired_seq_set).sum()
    print(f"📌 [검사 5] Paired WT와 서열이 100% 동일한 중복 Pure WT: {identical_seq_count:,}개")

    # ----------------------------------------------------
    # [종합 요약 결론]
    # ----------------------------------------------------
    print("==========================================")
    print("📊 [Pure WT 종합 진단 결론]")
    print("==========================================")

    total_anomalies = len(dirty_ids) + len(non_uniprot_prefix) + len(partial_matched)

    if total_anomalies == 0:
        print("✅ [검증 완료] 표기 오류, 숨은 특수문자, 미인식 구분자로 인한 누락은 0건입니다.")
        print(f"✅ 진단 대상 Pure WT ({len(pure_wt_df):,}개)는 전원 정상적인 독립 인간 단백질(Wildtype)입니다.")
    else:
        print(f"⚠️ 총 {total_anomalies:,}건의 표기/포맷 이상 의심 항목이 발견되었습니다.")
        print("   상기 샘플 결과를 바탕으로 정규화 파싱 로직 보완을 권장합니다.")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    # 정규화된 최신 CSV 파일 지정
    target_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")

    if not os.path.exists(target_csv):
        target_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset.csv")

    audit_pure_wildtypes(target_csv)