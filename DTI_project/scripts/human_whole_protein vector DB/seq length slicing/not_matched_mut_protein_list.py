import os
import re
import pandas as pd


def analyze_standalone_wt_lengths(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"❌ [오류] 파일이 존재하지 않습니다: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    # ID 파싱하여 Canonical ID 추출
    def extract_canonical_id(raw_id, seq_type):
        clean_id = re.sub(r'^WT_', '', str(raw_id)).strip()
        if seq_type == 'wildtype':
            return clean_id.split('-')[0]
        else:
            return clean_id.split('_')[0].split('-')[0]

    df['canonical_id'] = df.apply(lambda r: extract_canonical_id(r['id'], r['type']), axis=1)

    wt_df = df[df['type'] == 'wildtype']
    mutant_df = df[df['type'] == 'mutant']
    mutant_canonical_ids = set(mutant_df['canonical_id'])

    # Standalone WT (Mutant 없는 WT) 중 1,024aa 초과 데이터 필터링
    standalone_wt = wt_df[~wt_df['canonical_id'].isin(mutant_canonical_ids)]
    standalone_over_1024 = standalone_wt[standalone_wt['length'] > 1024].copy()

    # 길이를 기준으로 내림차순 정렬 (가장 긴 단백질부터)
    standalone_over_1024 = standalone_over_1024.sort_values(by='length', ascending=False)

    # 길이 통계 계산
    total_count = len(standalone_over_1024)
    max_len = standalone_over_1024['length'].max()
    min_len = standalone_over_1024['length'].min()
    avg_len = standalone_over_1024['length'].mean()

    print("==========================================")
    print(f"🔍 [Standalone WT (>1024aa) 570개 리스트 및 길이 분석]")
    print("==========================================")
    print(f"• 총 대상 개수: {total_count:,}개")
    print(f"• 평균 길이: {avg_len:,.1f} aa")
    print(f"• 최소 길이: {min_len:,} aa")
    print(f"• 최대 길이: {max_len:,} aa")
    print("------------------------------------------")

    print("🔝 [길이가 가장 긴 Top 10 Standalone WT]")
    print(standalone_over_1024[['id', 'length']].head(10).to_string(index=False))

    print("\n🔽 [길이가 가장 짧은 Bottom 5 Standalone WT (1024aa 근접)]")
    print(standalone_over_1024[['id', 'length']].tail(5).to_string(index=False))

    # 570개 전체 목록 파일 저장 (ID, 서열, 길이 포함)
    standalone_over_1024.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("==========================================")
    print(f"✅ 570개 전체 리스트 파일 저장 완료: \n   {output_csv}")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    input_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")
    output_csv = os.path.join(base_dir, "not_matched_mut_protein_details.csv")

    analyze_standalone_wt_lengths(input_csv, output_csv)

import time
import requests


def fetch_uniprot_features(uniprot_id):
    """
    UniProt API를 호출하여 단백질의 Active site와 Binding site 정보를 가져오는 함수
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])

            extracted_sites = []
            for feature in features:
                f_type = feature.get('type')
                # Active site 또는 Binding site만 추출
                if f_type in ['Active site', 'Binding site']:
                    desc = feature.get('description', '')
                    loc = feature.get('location', {})
                    start = loc.get('start', {}).get('value', '?')
                    end = loc.get('end', {}).get('value', '?')

                    # 정보 포맷팅 (예: Active site(120-120): Proton acceptor)
                    site_info = f"{f_type}({start}-{end})"
                    if desc:
                        site_info += f": {desc}"
                    extracted_sites.append(site_info)

            return " | ".join(extracted_sites) if extracted_sites else "None annotated"
        elif response.status_code == 404:
            return "ID Not Found in UniProt"
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return "Connection Failed"


def add_active_sites_to_csv(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"❌ [오류] 파일이 존재하지 않습니다: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    print("==========================================")
    print(f"🌐 [UniProt API 연동] Active Site 정보 조회 시작")
    print(f"• 대상 단백질 수: {len(df):,}개")
    print("• 예상 소요 시간: 약 1~2분 (서버 과부하 방지를 위해 천천히 조회합니다)")
    print("==========================================")

    active_sites_list = []

    for idx, row in df.iterrows():
        uid = row['canonical_id']
        sites = fetch_uniprot_features(uid)
        active_sites_list.append(sites)

        # 진행률 표시 (50개 단위)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"⏳ 진행 중... ({idx + 1}/{len(df)})")

        # UniProt 서버에 무리를 주지 않기 위한 딜레이
        time.sleep(0.1)

    # 데이터프레임에 새로운 열 추가
    df['active_sites'] = active_sites_list

    # 결과 요약
    has_site_count = len(
        df[~df['active_sites'].isin(["None annotated", "ID Not Found in UniProt", "Connection Failed"])])

    print("\n==========================================")
    print(f"📊 [조회 결과 요약]")
    print(f"• Active/Binding Site 정보가 존재하는 단백질: {has_site_count:,}개")
    print(f"• 정보가 없는 단백질: {len(df) - has_site_count:,}개")

    # 저장
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("------------------------------------------")
    print(f"✅ Active Site 정보가 포함된 파일이 저장되었습니다:")
    print(f"   {output_csv}")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    # 앞서 만든 570개 리스트 파일을 입력으로 사용
    input_csv = os.path.join(base_dir, "not_matched_mut_protein_details.csv")
    output_csv = os.path.join(base_dir, "standalone_wt_over_1024_with_active_sites.csv")

    add_active_sites_to_csv(input_csv, output_csv)