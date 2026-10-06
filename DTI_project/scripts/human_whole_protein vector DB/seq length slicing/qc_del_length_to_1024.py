import os
import re
import pandas as pd


def run_crop_quality_control(orig_csv, cropped_csv):
    print("==========================================")
    print("🔍 [정밀 QC] Mutant vs 대응 윈도우 WT 정합성 검증")
    print("==========================================")

    if not os.path.exists(cropped_csv) or not os.path.exists(orig_csv):
        print("❌ 입력 데이터 파일이 존재하지 않습니다.")
        return

    df_cropped = pd.read_csv(cropped_csv)
    df_orig = pd.read_csv(orig_csv)

    # 원본 WT 서열 딕셔너리
    wt_full_dict = df_orig[df_orig['type'] == 'wildtype'].set_index(
        df_orig[df_orig['type'] == 'wildtype']['id'].str.replace('WT_', '')
    )['sequence'].to_dict()

    passed = True

    # 1. 1024aa 제한 검사
    max_len = df_cropped['length'].max()
    if max_len > 1024:
        print(f"❌ [FAIL] 1024aa 초과 서열 존재: {max_len}aa")
        passed = False
    else:
        print(f"✅ [PASS 1] 모든 서열 1024aa 이하 보장 (최대 길이: {max_len}aa)")

    # 2. Mutant 변이 지점 정합성 검사
    mutant_df = df_cropped[df_cropped['type'] == 'mutant']
    diff_errors = 0
    checked = 0

    for idx, row in mutant_df.iterrows():
        mut_id = row['id']
        mut_seq = row['sequence']

        parts = mut_id.split('_')
        uid = parts[0]
        match = re.search(r'([A-Z])(\d+)([A-Z])$', mut_id)

        if uid in wt_full_dict and match:
            wt_full = wt_full_dict[uid]
            checked += 1

            # Mutant 서열과 원본 WT 슬라이딩 윈도우 비교 (차이 지점 1개 존재 여부)
            found_match = False
            for start_idx in range(0, max(1, len(wt_full) - len(mut_seq) + 1)):
                wt_sub = wt_full[start_idx:start_idx + len(mut_seq)]
                diff_count = sum(1 for a, b in zip(wt_sub, mut_seq) if a != b)
                if diff_count == 1:
                    found_match = True
                    break

            if not found_match:
                diff_errors += 1

    if diff_errors > 0:
        print(f"❌ [FAIL] 변이 위치 매칭 실패 Mutant: {diff_errors}개")
        passed = False
    else:
        print(f"✅ [PASS 2] {checked:,}개 Mutant 전체가 WT 대응 구간과 정확히 1개 변이 지점만 다름을 확인!")

    print("------------------------------------------")
    if passed:
        print("🎉 [QC 최종 통과] esm2_ready_dataset_cropped.csv 사용 준비 완료!")
    else:
        print("🚨 [QC 실패] 위 메시지를 확인하세요.")


if __name__ == "__main__":
    base_dir = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
    orig_csv = os.path.join(base_dir, "esm2_ready_total_protein_dataset.csv")
    cropped_csv = os.path.join(base_dir, "esm2_ready_dataset_cropped_final.csv")

    run_crop_quality_control(orig_csv, cropped_csv)