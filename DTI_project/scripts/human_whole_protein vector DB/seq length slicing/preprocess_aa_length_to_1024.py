import os
import re
import pandas as pd


def process_final_1024_cropping(input_csv, active_site_csv, output_csv, max_len=1024):
    if not os.path.exists(input_csv):
        print(f"❌ [오류] 파일이 존재하지 않습니다: {input_csv}")
        return

    print("==========================================")
    print("✂️ [최종 절단 파이프라인 시작] 1,024aa 맞춤형 Window Cropping")
    print("==========================================")

    df = pd.read_csv(input_csv)

    # 1. Active Site 데이터 로드 및 파싱 딕셔너리 구축
    active_site_map = {}
    if os.path.exists(active_site_csv):
        as_df = pd.read_csv(active_site_csv)
        for _, row in as_df.iterrows():
            uid = str(row['canonical_id'])
            sites_str = str(row.get('active_sites', ''))

            # 정규식으로 "(숫자-" 패턴 추출 (예: Active site(120-120) -> 120)
            matches = re.findall(r'\((\d+)-', sites_str)
            if matches:
                active_site_map[uid] = int(matches[0])  # 첫 번째 Active site 기준
    else:
        print(f"⚠️ Active site 정보 파일이 없습니다. Standalone WT는 모두 중앙을 기준으로 절단됩니다.")

    # 2. WT 검색 딕셔너리 구축 (Paired WT 대조군 추적용)
    wt_exact_dict = {}
    wt_canonical_dict = {}
    wt_df = df[df['type'] == 'wildtype']

    for _, row in wt_df.iterrows():
        raw_id = str(row['id']).replace('WT_', '').strip()
        canonical_id = raw_id.split('-')[0]
        wt_exact_dict[raw_id] = row.to_dict()
        if canonical_id not in wt_canonical_dict or '-' not in raw_id:
            wt_canonical_dict[canonical_id] = row.to_dict()

    mutant_canonical_ids = set()

    # 통계 기록용 딕셔너리
    stats = {
        'mutant_cropped': 0,
        'mutant_kept': 0,
        'paired_wt_cropped': 0,
        'paired_wt_kept': 0,
        'standalone_active_site': 0,
        'standalone_center': 0,
        'standalone_kept': 0
    }

    final_rows = []
    seen_ids = set()  # 중복 방지 세트

    # 공통 절단 함수
    def get_crop_bounds(center, seq_len):
        half = max_len // 2
        start = max(0, center - half)
        end = start + max_len
        if end > seq_len:
            end = seq_len
            start = max(0, seq_len - max_len)
        return int(start), int(end)

    def add_row(row_dict):
        if row_dict['id'] not in seen_ids:
            final_rows.append(row_dict)
            seen_ids.add(row_dict['id'])
            return True
        return False

    # 3. Mutant 기준 순회 (Mutant 및 Paired WT 처리)
    for _, row in df[df['type'] == 'mutant'].iterrows():
        seq = row['sequence']
        seq_len = row['length']

        # ID 파싱
        clean_id = str(row['id']).replace('WT_', '').strip()
        uid = clean_id.split('_')[0]
        canonical_uid = uid.split('-')[0]
        mutant_canonical_ids.add(canonical_uid)

        # 원본 WT 찾기
        wt_match = wt_exact_dict.get(uid) or wt_canonical_dict.get(canonical_uid)

        if seq_len <= max_len:
            # Mutant 1024 이하 -> 그대로 보존
            add_row(row.to_dict())
            stats['mutant_kept'] += 1

            # Paired WT도 그대로 보존
            if wt_match:
                if add_row(wt_match):
                    stats['paired_wt_kept'] += 1
        else:
            # Mutant 1024 초과 -> 변이 위치 기준 절단
            match = re.search(r'[A-Z](\d+)[A-Z]$', clean_id)
            mut_pos = int(match.group(1)) if match else (seq_len // 2)

            start, end = get_crop_bounds(mut_pos - 1, seq_len)

            # Mutant 저장
            mut_row = row.to_dict()
            mut_row['sequence'] = seq[start:end]
            mut_row['length'] = len(mut_row['sequence'])
            add_row(mut_row)
            stats['mutant_cropped'] += 1

            # Paired WT 저장 (동일 구간 절단)
            if wt_match:
                wt_seq = wt_match['sequence']
                wt_cropped_seq = wt_seq[start:end]
                wt_win_id = f"WT_{wt_match['id'].replace('WT_', '')}_win_{start}_{end}"

                wt_row = wt_match.copy()
                wt_row['id'] = wt_win_id
                wt_row['sequence'] = wt_cropped_seq
                wt_row['length'] = len(wt_cropped_seq)
                if add_row(wt_row):
                    stats['paired_wt_cropped'] += 1

    # 4. Standalone WT 순회 처리
    for _, row in wt_df.iterrows():
        raw_id = str(row['id']).replace('WT_', '').strip()
        canonical_uid = raw_id.split('-')[0]

        # 이미 처리된 Paired WT는 건너뜀
        if canonical_uid in mutant_canonical_ids:
            continue

        seq = row['sequence']
        seq_len = row['length']

        if seq_len <= max_len:
            if add_row(row.to_dict()):
                stats['standalone_kept'] += 1
        else:
            # 1024 초과 Standalone WT 절단 로직
            active_center = active_site_map.get(canonical_uid)

            if active_center is not None:
                start, end = get_crop_bounds(active_center - 1, seq_len)
                stats['standalone_active_site'] += 1
                crop_type = "AS"  # Active Site
            else:
                start, end = get_crop_bounds(seq_len // 2, seq_len)
                stats['standalone_center'] += 1
                crop_type = "MID"  # Center Middle

            st_row = row.to_dict()
            st_row['sequence'] = seq[start:end]
            st_row['length'] = len(st_row['sequence'])
            st_row['id'] = f"WT_{raw_id}_win_{crop_type}_{start}_{end}"
            add_row(st_row)

    # 5. 결과 저장
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("📊 [절단 처리 결과 요약 (1024aa 이상 절단 대상)]")
    print("------------------------------------------")
    print(f"📍 Mutant (변이 지점 기준 512aa 양방향): {stats['mutant_cropped']:,}개")
    print(f"📍 Paired WT (Mutant와 동일 구간 매칭): {stats['paired_wt_cropped']:,}개")
    print(f"📍 Standalone WT (Active Site 기준 절단): {stats['standalone_active_site']:,}개")
    print(f"📍 Standalone WT (서열 중앙부 기준 절단): {stats['standalone_center']:,}개")
    print("\n📊 [보존된 원본 서열 (1024aa 이하 보존)]")
    print("------------------------------------------")
    print(f"✔️ Mutant 원본 보존: {stats['mutant_kept']:,}개")
    print(f"✔️ Paired WT 원본 보존: {stats['paired_wt_kept']:,}개")
    print(f"✔️ Standalone WT 원본 보존: {stats['standalone_kept']:,}개")
    print("==========================================")
    print(f"✅ 최종 파일 저장 완료: {output_csv}")
    print(f"   (총 서열 수: {len(final_df):,}개)")
    print("==========================================")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    input_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset_normalized.csv")
    active_site_csv = os.path.join(base_dir, "standalone_wt_over_1024_with_active_sites.csv")
    output_csv = os.path.join(base_dir, "esm2_ready_dataset_cropped_final.csv")

    process_final_1024_cropping(input_csv, active_site_csv, output_csv)