import pandas as pd
from rdkit import Chem
import os

# ============================================================
# 경로
# ============================================================

input_path = "../../../data/output/merged_dti_DEDUP6.csv"
output_path = "../../../data/output/merged_dti_CANONICAL_SMILES1.csv"

chunksize = 200_000


# ============================================================
# 통계
# ============================================================

total_rows = 0
total_valid = 0
total_missing = 0
total_parse_failed = 0
total_changed = 0


first_chunk = True


# ============================================================
# Chunk 처리
# ============================================================

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        input_path,
        chunksize=chunksize,
        low_memory=False
    )
):

    print(f"\n{'='*70}")
    print(f"Chunk {chunk_idx + 1}")
    print(f"{'='*70}")

    canonical_smiles_list = []

    chunk_changed = 0
    chunk_parse_failed = 0
    chunk_missing = 0

    for smiles in chunk["smiles"]:

        # ----------------------------------------------------
        # 결측
        # ----------------------------------------------------

        if pd.isna(smiles) or str(smiles).strip() == "":
            chunk_missing += 1
            total_missing += 1

            canonical_smiles_list.append(smiles)
            continue

        smiles = str(smiles).strip()

        # ----------------------------------------------------
        # RDKit parsing
        # ----------------------------------------------------

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:

            chunk_parse_failed += 1
            total_parse_failed += 1

            # 원래 값 유지
            canonical_smiles_list.append(smiles)

            continue

        # ----------------------------------------------------
        # Canonical SMILES 생성
        # ----------------------------------------------------

        canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True
        )

        canonical_smiles_list.append(canonical_smiles)

        total_valid += 1

        # ----------------------------------------------------
        # 기존 SMILES와 canonical SMILES가 다른 경우
        # ----------------------------------------------------

        if canonical_smiles != smiles:
            chunk_changed += 1
            total_changed += 1

    # ========================================================
    # smiles 컬럼 교체
    # ========================================================

    chunk["smiles"] = canonical_smiles_list

    # ========================================================
    # 저장
    # ========================================================

    chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    total_rows += len(chunk)

    print(f"전체 행              : {len(chunk):,}")
    print(f"VALID                 : {len(chunk) - chunk_parse_failed - chunk_missing:,}")
    print(f"결측                  : {chunk_missing:,}")
    print(f"PARSE_FAILED          : {chunk_parse_failed:,}")
    print(f"Canonical 변경        : {chunk_changed:,}")


# ============================================================
# 최종 결과
# ============================================================

print("\n" + "="*70)
print("CANONICAL SMILES 변환 완료")
print("="*70)

print(f"전체 행                  : {total_rows:,}")
print(f"VALID                    : {total_valid:,}")
print(f"MISSING                  : {total_missing:,}")
print(f"PARSE_FAILED             : {total_parse_failed:,}")
print(f"Canonical 변경            : {total_changed:,}")

print(f"\n저장 위치:")
print(output_path)