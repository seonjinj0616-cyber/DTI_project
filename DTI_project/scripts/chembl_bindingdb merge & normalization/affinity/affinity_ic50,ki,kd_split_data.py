import os
import numpy as np
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\bindingdb_chembl_merging"

INPUT_FILE = "merged_chembl_binding_final.csv"

INPUT_PATH = os.path.join(
    DATA_DIR,
    INPUT_FILE
)


# 통합 최종 파일
OUTPUT_FILE = "merged_chembl_binding_final_paffinity.csv"

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    OUTPUT_FILE
)


# affinity_type별 분리 파일
IC50_OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "IC50.csv"
)

KI_OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "Ki.csv"
)

KD_OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "Kd.csv"
)


# ============================================================
# 2. 컬럼 설정
# ============================================================

MEAN_COL = "affinity_value_mean"

DELETE_COLS = [
    "measurement_count",
    "relation_type_count",
    "total_measurement_count",
    "affinity_relation"
]


# ============================================================
# 3. CSV LOAD
# ============================================================

print("=" * 100)
print("CSV LOAD")
print("=" * 100)

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(f"입력 파일 : {INPUT_PATH}")
print(f"전체 행 수 : {len(df):,}")
print()


# ============================================================
# 4. 필수 컬럼 확인
# ============================================================

required_cols = [
    "affinity_type",
    MEAN_COL
]

missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_cols}"
    )


# ============================================================
# 5. 삭제 대상 컬럼 확인
# ============================================================

print("=" * 100)
print("COLUMN DELETION")
print("=" * 100)

existing_delete_cols = [
    col
    for col in DELETE_COLS
    if col in df.columns
]

missing_delete_cols = [
    col
    for col in DELETE_COLS
    if col not in df.columns
]

print("삭제할 컬럼:")
for col in existing_delete_cols:
    print(f"  - {col}")

if missing_delete_cols:
    print()
    print("이미 존재하지 않는 컬럼:")
    for col in missing_delete_cols:
        print(f"  - {col}")

print()


# ============================================================
# 6. affinity_value_mean 숫자형 변환
# ============================================================

df[MEAN_COL] = pd.to_numeric(
    df[MEAN_COL],
    errors="coerce"
)


# ============================================================
# 7. pAffinity 계산
# ============================================================
#
# affinity_value_mean 단위 = nM
#
# pAffinity = -log10(M)
#
# nM → M
# nM × 10^-9
#
# 따라서
#
# pAffinity
# = -log10(nM × 10^-9)
# = 9 - log10(nM)
#
# ============================================================

print("=" * 100)
print("pAFFINITY CALCULATION")
print("=" * 100)


# 계산 가능한 값 확인
valid_paffinity_mask = (
    df[MEAN_COL].notna()
    & (df[MEAN_COL] > 0)
)

invalid_paffinity_count = (
    ~valid_paffinity_mask
).sum()

print(
    f"pAffinity 계산 가능 데이터 : "
    f"{valid_paffinity_mask.sum():,}"
)

print(
    f"pAffinity 계산 불가능 데이터 : "
    f"{invalid_paffinity_count:,}"
)


# pAffinity 컬럼 생성
df["paffinity"] = np.nan

df.loc[
    valid_paffinity_mask,
    "paffinity"
] = (
    9
    - np.log10(
        df.loc[
            valid_paffinity_mask,
            MEAN_COL
        ]
    )
)


# ============================================================
# 8. 삭제 대상 컬럼 제거
# ============================================================

df.drop(
    columns=existing_delete_cols,
    inplace=True
)


# ============================================================
# 9. pAffinity 통계
# ============================================================

print()
print("=" * 100)
print("pAFFINITY SUMMARY")
print("=" * 100)

print(
    f"pAffinity 최소값 : "
    f"{df['paffinity'].min():.6f}"
)

print(
    f"pAffinity 최대값 : "
    f"{df['paffinity'].max():.6f}"
)

print(
    f"pAffinity 평균값 : "
    f"{df['paffinity'].mean():.6f}"
)

print(
    f"pAffinity 결측값 : "
    f"{df['paffinity'].isna().sum():,}"
)

print()


# ============================================================
# 10. 컬럼 확인
# ============================================================

print("=" * 100)
print("FINAL COLUMNS")
print("=" * 100)

for i, col in enumerate(df.columns, start=1):
    print(
        f"{i:3d}. {col}"
    )

print()


# ============================================================
# 11. 전체 통합 파일 저장
# ============================================================

print("=" * 100)
print("SAVE MERGED DATA")
print("=" * 100)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"저장 완료 : {OUTPUT_PATH}"
)

print(
    f"행 수 : {len(df):,}"
)

print()


# ============================================================
# 12. affinity_type 정규화
# ============================================================

df["affinity_type"] = (
    df["affinity_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 13. affinity_type별 데이터 분리
# ============================================================

print("=" * 100)
print("AFFINITY TYPE SPLIT")
print("=" * 100)


# ------------------------------------------------------------
# IC50
# ------------------------------------------------------------

df_ic50 = df[
    df["affinity_type"] == "ic50"
].copy()

df_ic50.to_csv(
    IC50_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"IC50 : {len(df_ic50):,} rows"
)

print(
    f"저장 : {IC50_OUTPUT_PATH}"
)


# ------------------------------------------------------------
# Ki
# ------------------------------------------------------------

df_ki = df[
    df["affinity_type"] == "ki"
].copy()

df_ki.to_csv(
    KI_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Ki   : {len(df_ki):,} rows"
)

print(
    f"저장 : {KI_OUTPUT_PATH}"
)


# ------------------------------------------------------------
# Kd
# ------------------------------------------------------------

df_kd = df[
    df["affinity_type"] == "kd"
].copy()

df_kd.to_csv(
    KD_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Kd   : {len(df_kd):,} rows"
)

print(
    f"저장 : {KD_OUTPUT_PATH}"
)


# ============================================================
# 14. 최종 검증
# ============================================================

print()
print("=" * 100)
print("FINAL VALIDATION")
print("=" * 100)

print(
    f"전체 데이터 : {len(df):,}"
)

print(
    f"IC50        : {len(df_ic50):,}"
)

print(
    f"Ki          : {len(df_ki):,}"
)

print(
    f"Kd          : {len(df_kd):,}"
)

print(
    f"IC50 + Ki + Kd : "
    f"{len(df_ic50) + len(df_ki) + len(df_kd):,}"
)

print()

print("affinity_type 분포:")
print(
    df["affinity_type"]
    .value_counts()
)

print()


# ============================================================
# 15. 완료
# ============================================================

print("=" * 100)
print("DONE")
print("=" * 100)

print()
print("생성된 파일:")
print(f"1. {OUTPUT_PATH}")
print(f"2. {IC50_OUTPUT_PATH}")
print(f"3. {KI_OUTPUT_PATH}")
print(f"4. {KD_OUTPUT_PATH}")