import os
import re
import pandas as pd


def audit_and_normalize_ids(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"❌ [오류] 파일이 존재하지 않습니다: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    print("==========================================")
    print(f"🔍 [ID 정밀 진단 및 표준화 시작] 원본 서열: {len(df):,}개")
    print("==========================================")

    # 1. 정밀 ID 파싱 함수
    def parse_clean_id_info(row):
        raw_id = str(row['id']).strip()
        seq_type = str(row['type']).strip().lower()

        # 접두사(WT_) 제거
        clean_id = re.sub(r'^WT_', '', raw_id)

        if seq_type == 'wildtype':
            uniprot_id = clean_id
            # 아이소폼 접미사(-1, -2 등) 제거한 기본 UniProt ACC
            canonical_id = clean_id.split('-')[0]
            mut_info = None
        else:
            # Mutant ID 파싱 (예: P04217_H52R 또는 P04217-1_H52R)
            parts = clean_id.split('_')
            uniprot_id = parts[0]
            canonical_id = uniprot_id.split('-')[0]
            mut_info = parts[1] if len(parts) > 1 else None

        return pd.Series([clean_id, uniprot_id, canonical_id, mut_info])

    # ID 파싱 적용
    df[['clean_id', 'uniprot_id', 'canonical_id', 'mut_info']] = df.apply(parse_clean_id_info, axis=1)

    # 2. 아이소폼 감안 매칭 검증
    wt_df = df[df['type'] == 'wildtype']
    mut_df = df[df['type'] == 'mutant']

    exact_wt_uids = set(wt_df['uniprot_id'])
    exact_mut_uids = set(mut_df['uniprot_id'])

    canonical_wt_uids = set(wt_df['canonical_id'])
    canonical_mut_uids = set(mut_df['canonical_id'])

    # 결과 진단
    exact_standalone = exact_wt_uids - exact_mut_uids
    canonical_matched_standalone = {uid for uid in exact_standalone if uid.split('-')[0] in canonical_mut_uids}

    print("\n📊 [ID 매칭 정밀 진단 결과]")
    print(f"• 전체 WT 서열 수: {len(wt_df):,}개 (고유 UniProt ID: {len(exact_wt_uids):,}개)")
    print(f"• 전체 Mutant 서열 수: {len(mut_df):,}개 (고유 UniProt ID: {len(exact_mut_uids):,}개)")
    print(f"• 정확히 매칭된 WT-Mutant 쌍: {len(exact_wt_uids & exact_mut_uids):,}개")
    print(f"• 단독 WT 수: {len(exact_standalone):,}개")

    if canonical_matched_standalone:
        print(f"⚠️ [주의] 아이소폼 표기 차이(-1 등)로 매칭이 누락되었던 WT 발견: {len(canonical_matched_standalone):,}개")
        print(f"   예시: {list(canonical_matched_standalone)[:5]}")
    else:
        print("✅ 아이소폼 표기 오류로 인한 WT 매칭 누락은 존재하지 않습니다.")

    # 3. ID 컬럼 정돈 및 데이터셋 저장
    # id 컬럼의 WT_ 접두사를 깔끔하게 통일하거나 메타데이터 정리
    df['uniprot_id'] = df['uniprot_id']

    # 불필요 임시 컬럼 정리 후 저장
    save_df = df.drop(columns=['clean_id', 'canonical_id', 'mut_info'], errors='ignore')
    save_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n------------------------------------------")
    print(f"✅ ID 표준화 정제 완료 파일 저장: {output_csv}")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    input_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset.csv")
    output_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")

    audit_and_normalize_ids(input_csv, output_csv)