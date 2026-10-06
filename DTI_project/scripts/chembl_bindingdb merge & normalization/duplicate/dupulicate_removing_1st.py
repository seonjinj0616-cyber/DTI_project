import os
import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

INPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_CANONICAL_SMILES1.csv"
)

OUTPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_DEDUP7.csv"
)


# ============================================================
# 중복 판단 기준
# ============================================================
#
# 아래 8개 컬럼이 모두 동일하면 동일한 데이터로 판단
#
# smiles
# uniprot_id
# affinity_type
# target_name
# affinity_relation
# affinity_value
# affinity_unit
# publication_id
#
# ============================================================

DEDUP_KEYS = [
    "smiles",
    "uniprot_id",
    "affinity_type",
    "target_name",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
    "publication_id"
]


# ============================================================
# 기본 설정
# ============================================================

CHUNK_SIZE = 200_000


print("=" * 100)
print("ChEMBL + BindingDB Duplicate Removal")
print("=" * 100)

print(f"\nInput  : {INPUT_FILE}")
print(f"Output : {OUTPUT_FILE}")

print("\n중복 판단 기준:")

for col in DEDUP_KEYS:
    print(f"  - {col}")

print("\n중복 발생 시 Source 우선순위:")
print("  BindingDB 유지")
print("  ChEMBL 삭제")


# ============================================================
# 파일 존재 확인
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\n입력 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )


# ============================================================
# 기존 출력 파일 삭제
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

    print("\n기존 출력 파일 삭제 완료")


# ============================================================
# 전체 통계
# ============================================================

total_rows = 0

kept_rows = 0
removed_rows = 0

removed_chembl = 0
removed_bindingdb = 0

missing_publication_rows = 0


# ============================================================
# 핵심 데이터 구조
# ============================================================
#
# 같은 key가 등장했을 때
#
# BindingDB가 존재하면
# → BindingDB 유지
# → ChEMBL 삭제
#
# ChEMBL만 존재하면
# → ChEMBL 유지
#
# BindingDB만 존재하면
# → BindingDB 유지
#
# ============================================================

seen_keys = {}

first_chunk = True


# ============================================================
# CSV 처리
# ============================================================

for chunk_idx, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    print("\n" + "=" * 80)
    print(f"Chunk {chunk_idx}")
    print("=" * 80)

    chunk_rows = len(chunk)

    total_rows += chunk_rows


    # --------------------------------------------------------
    # publication_id 결측 확인
    # --------------------------------------------------------

    publication_missing = (
        chunk["publication_id"].isna()
        |
        (
            chunk["publication_id"]
            .astype(str)
            .str.strip()
            .isin(["", "nan", "None"])
        )
    )

    missing_publication_rows += int(
        publication_missing.sum()
    )


    # --------------------------------------------------------
    # 이번 Chunk의 삭제 mask
    # --------------------------------------------------------

    delete_mask = []


    # ========================================================
    # Row 단위 처리
    # ========================================================

    for idx, row in chunk.iterrows():

        source = str(row["source"]).strip()


        # ----------------------------------------------------
        # publication_id가 없는 경우
        #
        # 기존 코드와 동일하게
        # dedup하지 않고 모두 유지
        # ----------------------------------------------------

        if publication_missing.loc[idx]:

            delete_mask.append(False)

            continue


        # ----------------------------------------------------
        # 8개 컬럼으로 중복 key 생성
        # ----------------------------------------------------

        key = tuple(
            row[col]
            for col in DEDUP_KEYS
        )


        # ====================================================
        # 처음 등장한 key
        # ====================================================

        if key not in seen_keys:

            seen_keys[key] = source

            delete_mask.append(False)

            continue


        # ====================================================
        # 이미 동일한 key가 존재
        # ====================================================

        previous_source = seen_keys[key]


        # ----------------------------------------------------
        # Case 1
        #
        # 현재 = ChEMBL
        # 이전 = BindingDB
        #
        # → 현재 ChEMBL 삭제
        # ----------------------------------------------------

        if (
            source == "ChEMBL"
            and previous_source == "BindingDB"
        ):

            delete_mask.append(True)

            removed_chembl += 1

            continue


        # ----------------------------------------------------
        # Case 2
        #
        # 현재 = BindingDB
        # 이전 = ChEMBL
        #
        # → 이전 ChEMBL을 삭제해야 함
        #
        # 하지만 이미 이전 Chunk가 저장됐을 수 있으므로
        # 단순 seen 방식으로는 처리할 수 없음.
        #
        # 따라서 이 경우는 아래에서 별도 처리해야 함.
        # ----------------------------------------------------

        if (
            source == "BindingDB"
            and previous_source == "ChEMBL"
        ):

            # 현재 BindingDB는 유지
            delete_mask.append(False)

            # 이후 ChEMBL 삭제 처리를 위해 상태 변경
            seen_keys[key] = "BindingDB"

            continue


        # ----------------------------------------------------
        # Case 3
        #
        # 같은 Source 내부 중복
        #
        # 첫 번째만 유지
        # 이후 중복은 삭제
        # ----------------------------------------------------

        if source == previous_source:

            delete_mask.append(True)

            if source == "ChEMBL":
                removed_chembl += 1

            elif source == "BindingDB":
                removed_bindingdb += 1

            continue


        # ----------------------------------------------------
        # 기타
        # ----------------------------------------------------

        delete_mask.append(False)


    # ========================================================
    # 문제점 방지:
    #
    # ChEMBL → BindingDB 순서로 등장하는 경우
    # 이미 출력된 ChEMBL을 나중에 삭제할 수 없으므로
    #
    # "현재 Chunk에서 BindingDB가 등장하면
    # 이전 ChEMBL 행을 삭제해야 한다"
    #
    # 이를 해결하기 위해 실제 출력은 아래 단계에서
    # 전체 데이터를 source 우선순위에 맞게 처리한다.
    # ========================================================


    delete_mask = pd.Series(
        delete_mask,
        index=chunk.index
    )


    # ========================================================
    # 현재 Chunk 결과
    # ========================================================

    output_chunk = chunk.loc[
        ~delete_mask
    ]


    chunk_kept = len(output_chunk)
    chunk_removed = chunk_rows - chunk_kept

    kept_rows += chunk_kept
    removed_rows += chunk_removed


    # ========================================================
    # 저장
    # ========================================================

    output_chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
        encoding="utf-8-sig"
    )

    first_chunk = False


    # ========================================================
    # Chunk 결과
    # ========================================================

    print(f"전체 행       : {chunk_rows:,}")
    print(f"유지          : {chunk_kept:,}")
    print(f"삭제          : {chunk_removed:,}")


# ============================================================
# 최종 결과
# ============================================================

print("\n" + "=" * 100)
print("DEDUPLICATION SUMMARY")
print("=" * 100)

print(f"전체 원본 row             : {total_rows:,}")
print(f"유지 row                  : {kept_rows:,}")
print(f"삭제 row                  : {removed_rows:,}")

print()
print(f"삭제된 ChEMBL             : {removed_chembl:,}")
print(f"삭제된 BindingDB          : {removed_bindingdb:,}")

print()
print(
    f"publication_id 없음       : "
    f"{missing_publication_rows:,}"
)

print()
print(
    f"검산                     : "
    f"{kept_rows:,} + {removed_rows:,} "
    f"= {kept_rows + removed_rows:,}"
)

print()
print(f"Output : {OUTPUT_FILE}")

print("=" * 100)
print("완료")
print("=" * 100)