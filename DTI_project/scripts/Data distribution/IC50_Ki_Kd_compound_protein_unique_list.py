import os
import pandas as pd

# =====================================================================
# 1. 경로 및 컬럼 설정
# =====================================================================
# 알려주신 기본 폴더 경로 (Windows 경로이므로 앞에 r을 붙여줍니다)
base_dir = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\final_final"

# 각 파일 경로 설정 (실제 파일 이름이 다를 경우 파일명을 수정해주세요)
file_paths = {
    "IC50": os.path.join(base_dir, "IC50_ems2_fi.csv"),
    "Kd": os.path.join(base_dir, "Kd_ems2_fi.csv"),
    "Ki": os.path.join(base_dir, "Ki_ems2_fi.csv")
}

# CSV 파일 내 실제 사용 중인 컬럼명 (필요시 변경)
smiles_col = "smiles"
uniprot_col = "uniprot_id"

# =====================================================================
# 2. 중복 제외 고유 개수 계산 (읽기 전용)
# =====================================================================
print("=" * 65)
print("IC50, Kd, Ki 데이터셋 고유 화합물 및 타겟 단백질 개수 분석")
print("=" * 65)

for dataset_name, path in file_paths.items():
    if not os.path.exists(path):
        print(f"[{dataset_name}] 파일을 찾을 수 없습니다: {path}\n")
        continue

    try:
        # read_csv는 메모리로만 로드하므로 원본 파일은 안전합니다.
        df = pd.read_csv(path, usecols=[smiles_col, uniprot_col])
        total_rows = len(df)

        # 각각 중복을 제거한 후 고유 개수 측정
        unique_compounds = len(df.drop_duplicates(subset=[smiles_col]))
        unique_proteins = len(df.drop_duplicates(subset=[uniprot_col]))

        # 결과 출력
        print(f"[{dataset_name} Dataset]")
        print(f" - 파일 경로                 : {path}")
        print(f" - 전체 데이터 행 수 (Rows)  : {total_rows:,} 개")
        print(f" - 중복 제외 화합물 수 (SMILES): {unique_compounds:,} 개")
        print(f" - 중복 제외 단백질 수 (UniProt): {unique_proteins:,} 개")
        print("-" * 65)

    except ValueError:
        print(f"[{dataset_name}] '{smiles_col}' 또는 '{uniprot_col}' 컬럼을 찾을 수 없습니다. CSV 파일의 컬럼명을 확인해주세요.")
    except Exception as e:
        print(f"[{dataset_name}] 오류 발생: {e}")