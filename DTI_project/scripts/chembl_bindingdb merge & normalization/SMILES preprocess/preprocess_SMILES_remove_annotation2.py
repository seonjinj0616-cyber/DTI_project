import os
import re
import pandas as pd
from rdkit import Chem

# ============================================================
# 경로 설정
# ============================================================
input_path = "../../data/output/merged_dti_raw.csv"
output_path = "../../data/output/merged_dti_smiles_cleaned.csv"

# ============================================================
# 설정
# ============================================================
chunksize = 200_000

# 기존 결과 파일 삭제
if os.path.exists(output_path):
    os.remove(output_path)


# ============================================================
# SMILES 정제 함수
# ============================================================
def clean_smiles(smiles):
    """
    SMILES 뒤에 존재하는 |...| annotation을 제거하고
    앞뒤 공백을 제거한다.

    예:
        CCO |ABC:123|       -> CCO
        CCO|ABC:123|        -> CCO
        CCO |r,THB:...|     -> CCO
        CCO |wU:...,wD:...| -> CCO
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles)

    # --------------------------------------------------------
    # 1. | ... | 사이의 모든 문자 제거
    # --------------------------------------------------------
    # non-greedy로 |...| 형태를 모두 제거
    cleaned = re.sub(r"\|.*?\|", "", smiles)

    # --------------------------------------------------------
    # 2. 앞뒤 공백 제거
    # --------------------------------------------------------
    cleaned = cleaned.strip()

    return cleaned


# ============================================================
# RDKit validation
# ============================================================
def validate_smiles(smiles):
    """
    RDKit으로 SMILES 검증

    return:
        VALID
        MISSING
        PARSE_FAILED
        VALENCE_ERROR
        KEKULIZE_ERROR
        OTHER_ERROR
    """

    if smiles is None or smiles == "":
        return "MISSING"

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return "PARSE_FAILED"

        return "VALID"

    except Exception as e:

        error_msg = str(e).lower()

        if "valence" in error_msg:
            return "VALENCE_ERROR"

        elif "kekul" in error_msg:
            return "KEKULIZE_ERROR"

        else:
            return "OTHER_ERROR"


# ============================================================
# 처리 시작
# ============================================================
total_rows = 0

valid_rows = 0
missing_rows = 0
parse_failed_rows = 0
valence_error_rows = 0
kekulize_error_rows = 0
other_error_rows = 0

annotation_removed_rows = 0
deleted_rows = 0

first_chunk = True


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        input_path,
        chunksize=chunksize,
        low_memory=False
    )
):

    print(f"\nChunk {chunk_idx + 1}")

    total_rows += len(chunk)

    # --------------------------------------------------------
    # 원본 SMILES 저장
    # --------------------------------------------------------
    original_smiles = chunk["smiles"].copy()

    # --------------------------------------------------------
    # SMILES 정제
    # --------------------------------------------------------
    chunk["smiles"] = chunk["smiles"].apply(clean_smiles)

    # 실제 변경 여부 확인
    changed = (
        original_smiles.fillna("").astype(str).str.strip()
        != chunk["smiles"].fillna("").astype(str)
    )

    annotation_removed_rows += int(changed.sum())

    # --------------------------------------------------------
    # RDKit 검증
    # --------------------------------------------------------
    validation_result = chunk["smiles"].apply(validate_smiles)

    # --------------------------------------------------------
    # 오류 통계
    # --------------------------------------------------------
    valid_count = int((validation_result == "VALID").sum())
    missing_count = int((validation_result == "MISSING").sum())
    parse_failed_count = int(
        (validation_result == "PARSE_FAILED").sum()
    )
    valence_error_count = int(
        (validation_result == "VALENCE_ERROR").sum()
    )
    kekulize_error_count = int(
        (validation_result == "KEKULIZE_ERROR").sum()
    )
    other_error_count = int(
        (validation_result == "OTHER_ERROR").sum()
    )

    valid_rows += valid_count
    missing_rows += missing_count
    parse_failed_rows += parse_failed_count
    valence_error_rows += valence_error_count
    kekulize_error_rows += kekulize_error_count
    other_error_rows += other_error_count

    # --------------------------------------------------------
    # VALENCE_ERROR / KEKULIZE_ERROR 삭제
    # --------------------------------------------------------
    delete_mask = validation_result.isin([
        "VALENCE_ERROR",
        "KEKULIZE_ERROR"
    ])

    deleted_rows += int(delete_mask.sum())

    chunk = chunk.loc[~delete_mask].copy()

    # --------------------------------------------------------
    # validation 결과 컬럼은 최종 데이터에 추가하지 않음
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------
    chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    print(f"  원본 행 수              : {len(validation_result):,}")
    print(f"  VALID                   : {valid_count:,}")
    print(f"  MISSING                 : {missing_count:,}")
    print(f"  PARSE_FAILED            : {parse_failed_count:,}")
    print(f"  VALENCE_ERROR           : {valence_error_count:,}")
    print(f"  KEKULIZE_ERROR          : {kekulize_error_count:,}")
    print(f"  OTHER_ERROR             : {other_error_count:,}")
    print(f"  삭제                    : {int(delete_mask.sum()):,}")
    print(f"  저장                    : {len(chunk):,}")


# ============================================================
# 최종 결과
# ============================================================
print("\n")
print("=" * 80)
print("SMILES CLEANING + VALIDATION 완료")
print("=" * 80)

print(f"전체 원본 행              : {total_rows:,}")
print(f"annotation 제거된 행      : {annotation_removed_rows:,}")
print()
print(f"VALID                     : {valid_rows:,}")
print(f"MISSING                   : {missing_rows:,}")
print(f"PARSE_FAILED              : {parse_failed_rows:,}")
print(f"VALENCE_ERROR             : {valence_error_rows:,}")
print(f"KEKULIZE_ERROR            : {kekulize_error_rows:,}")
print(f"OTHER_ERROR               : {other_error_rows:,}")
print()
print(f"VALENCE/KEKULIZE 삭제     : {deleted_rows:,}")
print(f"최종 저장 행 수           : {total_rows - deleted_rows:,}")
print()
print(f"최종 파일                 : {output_path}")
print("=" * 80)