import os
import pandas as pd

# 1. 경로 설정
target_dir = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"

# 대입할 대상 파일 세트 (원본 파일명, 상세 결과 파일명, 최종 대입 결과 파일명)
file_sets = {
    "IC50": {
        "orig": "IC50_esm2_final_qc.csv",
        "detail": "IC50_slicing_details_all.csv",
        "out": "IC50_esm2_final_updated.csv"
    },
    "Ki": {
        "orig": "Ki_esm2_final_qc.csv",
        "detail": "Ki_slicing_details_all.csv",
        "out": "Ki_esm2_final_updated.csv"
    },
    "Kd": {
        "orig": "Kd_esm2_final_qc.csv",
        "detail": "Kd_slicing_details_all.csv",
        "out": "Kd_esm2_final_updated.csv"
    }
}

print("=" * 60)
print("STARTING SEQUENCE MAPPING & REPLACEMENT PIPELINE")
print("=" * 60)

for name, paths in file_sets.items():
    orig_path = os.path.join(target_dir, paths["orig"])
    detail_path = os.path.join(target_dir, paths["detail"])
    out_path = os.path.join(target_dir, paths["out"])

    # 파일 존재 확인
    if not os.path.exists(orig_path):
        print(f"[Skip] 원본 파일을 찾을 수 없습니다: {orig_path}")
        continue
    if not os.path.exists(detail_path):
        print(f"[Skip] 상세 로그 파일을 찾을 수 없습니다: {detail_path}")
        continue

    print(f"\nProcessing [{name}]...")
    df_orig = pd.read_csv(orig_path, low_memory=False)
    df_detail = pd.read_csv(detail_path, low_memory=False)

    print(f"  - 원본 데이터 행 수 : {len(df_orig):,}개")
    print(f"  - 상세 로그 행 수   : {len(df_detail):,}개")

    # 1:1 행(Row) 매핑 검증 및 서열/QC 컬럼 대입
    if len(df_orig) == len(df_detail):
        # 1) 기존 원본 단백질 서열 컬럼을 슬라이싱된 원본 서열로 교체
        seq_col_name = 'protein_sequence' if 'protein_sequence' in df_orig.columns else 'sequence'
        df_orig[seq_col_name] = df_detail['sliced_sequence']

        # 2) 절단 QC 정보 컬럼 추측/추가
        df_orig['slicing_rule'] = df_detail['slicing_rule']
        df_orig['crop_start_idx'] = df_detail['crop_start_idx']
        df_orig['crop_end_idx'] = df_detail['crop_end_idx']
        df_orig['original_length'] = df_detail['original_length']
        df_orig['sliced_length'] = df_detail['sliced_length']

        print("  -> 슬라이싱 서열 및 QC 컬럼 대입 완료")
    else:
        print("  [경고] 원본과 상세 로그의 행 수가 일치하지 않습니다. 인덱스 기반으로 부분 대입을 시도합니다.")
        seq_col_name = 'protein_sequence' if 'protein_sequence' in df_orig.columns else 'sequence'
        df_orig.loc[:len(df_detail)-1, seq_col_name] = df_detail['sliced_sequence'].values

    # 검증: 대입 후 최대 서열 길이 확인
    max_len = df_orig[seq_col_name].astype(str).str.len().max()
    over_1024 = (df_orig[seq_col_name].astype(str).str.len() > 1024).sum()
    print(f"  [대입 후 QC 검증] 최대 길이: {max_len} aa | >1024aa 잔여: {over_1024}개")

    # 최종 CSV 저장
    df_orig.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  --> 최종 완료 파일 저장: {paths['out']}")

print("\n" + "=" * 60)
print("ALL DATASETS SUCCESSFULLY UPDATED AND SAVED!")
print("=" * 60)