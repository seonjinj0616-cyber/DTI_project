import os
import pandas as pd

# =====================================================================
# 1. 경로 및 설정
# =====================================================================
base_dir = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\final_final"

file_paths = {
    "IC50": os.path.join(base_dir, "IC50_ems2_fi.csv"),
    "Kd": os.path.join(base_dir, "Kd_ems2_fi.csv"),
    "Ki": os.path.join(base_dir, "Ki_ems2_fi.csv")
}

affinity_col = "paffinity"

# =====================================================================
# 2. 구간(Bin) 설정 및 레이블
# =====================================================================
bins = [-float('inf'), 5.0, 6.0, 7.0, 8.0, 9.0, float('inf')]
labels = ['5 미만', '5 이상 ~ 6 미만', '6 이상 ~ 7 미만', '7 이상 ~ 8 미만', '8 이상 ~ 9 미만', '9 이상']

print("=" * 70)
print("paffinity 값의 구간별 분포 및 백분율 분석")
print("=" * 70)

for dataset_name, path in file_paths.items():
    if not os.path.exists(path):
        print(f"[{dataset_name}] 파일을 찾을 수 없습니다: {path}\n")
        continue

    try:
        # paffinity 컬럼만 읽어오기 (원본 파일은 안전함)
        df = pd.read_csv(path, usecols=[affinity_col])
        total_rows = len(df)

        if total_rows == 0:
            print(f"[{dataset_name}] 데이터가 비어 있습니다.\n")
            continue

        # 결측치(NaN) 제외
        valid_df = df.dropna(subset=[affinity_col]).copy()
        valid_count = len(valid_df)

        # pd.cut을 이용해 구간별로 데이터 분류
        valid_df['bin'] = pd.cut(valid_df[affinity_col], bins=bins, labels=labels, right=False)

        # 구간별 개수 및 백분율 계산
        counts = valid_df['bin'].value_counts().reindex(labels)
        percentages = (counts / valid_count) * 100

        # 결과 출력
        print(f"\n[{dataset_name} Dataset]")
        print(f" - 파일 경로     : {path}")
        print(f" - 전체 데이터 수 : {total_rows:,} 개 (유효 값: {valid_count:,} 개)")
        print(f" - 구간별 분포:")

        for label in labels:
            cnt = counts[label]
            pct = percentages[label]
            # 💡 수정된 부분: f-스트링 내부 정렬 구문 수정
            print(f"   * {label:<16} : {cnt:10,} 개 ({pct:5.2f}%)")

        print("-" * 70)

    except ValueError:
        print(f"[{dataset_name}] '{affinity_col}' 컬럼을 찾을 수 없습니다. 컬럼명을 확인해주세요.")
    except Exception as e:
        print(f"[{dataset_name}] 오류 발생: {e}")