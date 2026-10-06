import os
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\bindingdb_chembl_merging"


FILES = {
    "IC50": "IC50.csv",
    "Ki": "Ki.csv",
    "Kd": "Kd.csv"
}


# ============================================================
# 2. 기준 설정
# ============================================================

PAFFINITY_COL = "paffinity"

MIN_PAFFINITY = 0
MAX_PAFFINITY = 12


# ============================================================
# 3. IC50 / Ki / Kd 각각 독립적으로 처리
# ============================================================

for affinity_type, filename in FILES.items():

    input_path = os.path.join(
        DATA_DIR,
        filename
    )

    # --------------------------------------------------------
    # CSV LOAD
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(f"{affinity_type} 처리")
    print("=" * 100)

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    original_count = len(df)

    print(
        f"입력 파일 : {input_path}"
    )

    print(
        f"원본 행 수 : {original_count:,}"
    )

    # --------------------------------------------------------
    # 필수 컬럼 확인
    # --------------------------------------------------------

    if PAFFINITY_COL not in df.columns:
        raise ValueError(
            f"{filename}에 '{PAFFINITY_COL}' 컬럼이 없습니다."
        )

    # --------------------------------------------------------
    # paffinity 숫자형 변환
    # --------------------------------------------------------

    df[PAFFINITY_COL] = pd.to_numeric(
        df[PAFFINITY_COL],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 삭제 대상
    #
    # paffinity < 0
    # OR
    # paffinity > 12
    #
    # 0 <= paffinity <= 12 는 유지
    # --------------------------------------------------------

    invalid_mask = (
        (df[PAFFINITY_COL] < MIN_PAFFINITY)
        |
        (df[PAFFINITY_COL] > MAX_PAFFINITY)
    )

    invalid_df = df.loc[
        invalid_mask
    ].copy()

    invalid_count = len(invalid_df)

    # --------------------------------------------------------
    # 삭제 대상 조회
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print("삭제 대상 조회")
    print("-" * 100)

    print(
        f"paffinity < 0 : "
        f"{(df[PAFFINITY_COL] < 0).sum():,}"
    )

    print(
        f"paffinity > 12 : "
        f"{(df[PAFFINITY_COL] > 12).sum():,}"
    )

    print(
        f"전체 삭제 대상 : "
        f"{invalid_count:,}"
    )

    # --------------------------------------------------------
    # 삭제 대상 실제 데이터 출력
    # --------------------------------------------------------

    if invalid_count > 0:

        print()
        print("삭제 대상 데이터:")

        print(
            invalid_df.to_string(
                index=False
            )
        )

    else:

        print()
        print("삭제 대상 데이터가 없습니다.")

    # --------------------------------------------------------
    # 정상 범위 데이터만 유지
    # --------------------------------------------------------

    df = df.loc[
        ~invalid_mask
    ].copy()

    final_count = len(df)

    # --------------------------------------------------------
    # 기존 파일에 덮어쓰기
    # --------------------------------------------------------

    df.to_csv(
        input_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 최종 결과
    # --------------------------------------------------------

    print()
    print("-" * 100)
    print(f"{affinity_type} 최종 결과")
    print("-" * 100)

    print(
        f"원본 행 수 : {original_count:,}"
    )

    print(
        f"삭제 행 수 : {invalid_count:,}"
    )

    print(
        f"최종 행 수 : {final_count:,}"
    )

    if original_count > 0:

        print(
            f"삭제 비율 : "
            f"{invalid_count / original_count * 100:.4f}%"
        )

    print()
    print("최종 paffinity 범위:")

    if final_count > 0:

        print(
            f"최소값 : "
            f"{df[PAFFINITY_COL].min():.6f}"
        )

        print(
            f"최대값 : "
            f"{df[PAFFINITY_COL].max():.6f}"
        )

    print()
    print(
        f"저장 완료 : {input_path}"
    )


# ============================================================
# 4. 완료
# ============================================================

print()
print("=" * 100)
print("ALL DONE")
print("=" * 100)

print()
print("IC50 / Ki / Kd를 각각 독립적으로 처리했습니다.")
print()
print("최종 조건:")
print("0 <= paffinity <= 12")