import pandas as pd
from pathlib import Path


# ============================================================
# 1. 경로 설정
# ============================================================

INPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_DEDUP3.csv"
)

OUTPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_DEDUP4.csv"
)

QC_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\dedup4_qc.csv"
)

CHUNK_SIZE = 200_000


# ============================================================
# 2. 중복 판단 기준
#
# DEDUP3에서 target_name만 제외
#
# 포함:
#   chembl_id
#   uniprot_id
#   affinity_type
#   affinity_relation
#   affinity_value → round(2)
#   affinity_unit
#
# 제외:
#   target_name
#   assay_id
#   smiles
# ============================================================

MATCH_KEYS = [
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "affinity_relation",
    "affinity_unit",
]


REQUIRED_COLUMNS = [
    "source",
    "smiles",
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "target_name",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
]


# ============================================================
# 3. 입력 파일 확인
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. 컬럼 확인
# ============================================================

header = pd.read_csv(
    INPUT_FILE,
    nrows=0,
    dtype=str,
    keep_default_na=False
)

missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in header.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )


# ============================================================
# 5. 시작
# ============================================================

print("=" * 100)
print("DEDUP4 : ChEMBL ↔ BindingDB 중복 제거")
print("=" * 100)

print()
print("DEDUP4 중복 판단 기준:")
print("  chembl_id")
print("  uniprot_id")
print("  affinity_type")
print("  affinity_relation")
print("  affinity_value  ← 소수점 둘째 자리까지 반올림하여 비교")
print("  affinity_unit")

print()
print("이번 단계에서 제외:")
print("  target_name")
print("  assay_id")
print("  smiles")

print()
print("동일 조건이면:")
print("  ChEMBL    → 삭제")
print("  BindingDB → 유지")
print("  기타 Source → 유지")

print()
print("※ 원본 affinity_value 값은 변경하지 않습니다.")
print("=" * 100)


# ============================================================
# 6. PASS 1
#
# BindingDB의 비교용 key를 전부 수집
#
# 중요:
# chunk 순서와 관계없이 BindingDB 전체를 먼저 수집한다.
# ============================================================

print()
print("=" * 100)
print("PASS 1 : BindingDB 비교용 key 수집")
print("=" * 100)


bindingdb_keys = set()

total_rows_pass1 = 0
bindingdb_rows_pass1 = 0
invalid_affinity_bindingdb = 0


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    )
):

    total_rows_pass1 += len(chunk)

    # --------------------------------------------------------
    # BindingDB만 추출
    # --------------------------------------------------------

    bindingdb = chunk[
        chunk["source"].eq("BindingDB")
    ].copy()

    bindingdb_rows_pass1 += len(bindingdb)

    if not bindingdb.empty:

        # ----------------------------------------------------
        # affinity_value 숫자 변환 + round(2)
        # ----------------------------------------------------

        bindingdb[
            "affinity_value_compare"
        ] = pd.to_numeric(
            bindingdb["affinity_value"],
            errors="coerce"
        ).round(2)

        # ----------------------------------------------------
        # 숫자로 변환 가능한 행만 비교
        # ----------------------------------------------------

        valid_mask = (
            bindingdb[
                "affinity_value_compare"
            ].notna()
        )

        invalid_affinity_bindingdb += int(
            (~valid_mask).sum()
        )

        bindingdb_valid = bindingdb.loc[
            valid_mask
        ]

        if not bindingdb_valid.empty:

            # ------------------------------------------------
            # MATCH_KEYS 복사
            # ------------------------------------------------

            temp_keys = bindingdb_valid[
                MATCH_KEYS
            ].copy()

            # ------------------------------------------------
            # 반올림된 affinity_value를 key에 추가
            #
            # DEDUP3 코드와 동일한 방식
            # ------------------------------------------------

            temp_keys[
                "affinity_value"
            ] = bindingdb_valid[
                "affinity_value_compare"
            ]

            # ------------------------------------------------
            # tuple 생성
            # ------------------------------------------------

            keys = temp_keys.itertuples(
                index=False,
                name=None
            )

            bindingdb_keys.update(keys)

    print(
        f"Chunk {chunk_idx:>3} | "
        f"전체 {len(chunk):>9,} | "
        f"BindingDB {len(bindingdb):>9,} | "
        f"누적 unique key {len(bindingdb_keys):>12,}"
    )


print()
print(f"전체 입력 행 수             : {total_rows_pass1:,}")
print(f"BindingDB 행 수              : {bindingdb_rows_pass1:,}")
print(f"BindingDB unique 비교 key    : {len(bindingdb_keys):,}")
print(
    f"비교 불가 BindingDB 행 수     : "
    f"{invalid_affinity_bindingdb:,}"
)


# ============================================================
# 7. PASS 2
#
# ChEMBL:
#   BindingDB와 동일 key → 삭제
#
# BindingDB:
#   전부 유지
#
# 기타 source:
#   전부 유지
# ============================================================

print()
print("=" * 100)
print("PASS 2 : ChEMBL 중복 행 삭제")
print("=" * 100)


first_write = True

total_rows = 0

deleted_chembl = 0
kept_chembl = 0
kept_bindingdb = 0
kept_other = 0

invalid_affinity_chembl = 0

qc_records = []


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    )
):

    input_chunk_rows = len(chunk)

    total_rows += input_chunk_rows

    # --------------------------------------------------------
    # Source mask
    # --------------------------------------------------------

    chembl_mask = chunk[
        "source"
    ].eq("ChEMBL")

    bindingdb_mask = chunk[
        "source"
    ].eq("BindingDB")

    other_mask = (
        ~chembl_mask
        & ~bindingdb_mask
    )


    # ========================================================
    # ChEMBL 검사
    # ========================================================

    chembl = chunk.loc[
        chembl_mask
    ].copy()

    deleted_count = 0
    kept_count = 0


    if not chembl.empty:

        # ----------------------------------------------------
        # affinity_value 숫자 변환 + round(2)
        # ----------------------------------------------------

        chembl[
            "affinity_value_compare"
        ] = pd.to_numeric(
            chembl["affinity_value"],
            errors="coerce"
        ).round(2)


        # ----------------------------------------------------
        # 숫자로 변환되지 않는 affinity_value
        #
        # 이런 행은 비교하지 않고 유지
        # ----------------------------------------------------

        invalid_mask = (
            chembl[
                "affinity_value_compare"
            ].isna()
        )

        invalid_affinity_chembl += int(
            invalid_mask.sum()
        )


        # ----------------------------------------------------
        # 정상적인 affinity_value만 비교
        # ----------------------------------------------------

        valid_chembl = chembl.loc[
            ~invalid_mask
        ]


        if not valid_chembl.empty:

            # ------------------------------------------------
            # 비교용 key 생성
            #
            # target_name은 여기서 사용하지 않음
            # ------------------------------------------------

            temp_keys = valid_chembl[
                MATCH_KEYS
            ].copy()

            temp_keys[
                "affinity_value"
            ] = valid_chembl[
                "affinity_value_compare"
            ]


            # ------------------------------------------------
            # tuple 생성
            # ------------------------------------------------

            chembl_keys = temp_keys.itertuples(
                index=False,
                name=None
            )


            # ------------------------------------------------
            # BindingDB에 동일 key가 있는지 확인
            # ------------------------------------------------

            delete_mask_valid = pd.Series(
                [
                    key in bindingdb_keys
                    for key in chembl_keys
                ],
                index=valid_chembl.index
            )


            # ------------------------------------------------
            # 삭제 대상 ChEMBL index
            # ------------------------------------------------

            delete_indices = valid_chembl.index[
                delete_mask_valid
            ]


            deleted_count = len(
                delete_indices
            )

            kept_count = (
                len(chembl)
                - deleted_count
            )


            # ------------------------------------------------
            # ChEMBL만 삭제
            # ------------------------------------------------

            chunk = chunk.drop(
                index=delete_indices
            )


    # ========================================================
    # 통계
    # ========================================================

    deleted_chembl += deleted_count

    kept_chembl += kept_count

    kept_bindingdb += int(
        bindingdb_mask.sum()
    )

    kept_other += int(
        other_mask.sum()
    )


    # ========================================================
    # QC 기록
    # ========================================================

    if deleted_count > 0:

        qc_records.append({
            "chunk": chunk_idx,
            "deleted_chembl_rows": deleted_count,
            "reason": (
                "chembl_id + "
                "uniprot_id + "
                "affinity_type + "
                "affinity_relation + "
                "rounded_affinity_value(2dp) + "
                "affinity_unit "
                "matched with BindingDB; "
                "target_name excluded; "
                "assay_id excluded; "
                "smiles excluded"
            )
        })


    # ========================================================
    # 결과 저장
    # ========================================================

    if not chunk.empty:

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig"
        )

        first_write = False


    print(
        f"Chunk {chunk_idx:>3} | "
        f"입력 {input_chunk_rows:>9,} | "
        f"ChEMBL 삭제 {deleted_count:>9,} | "
        f"누적 삭제 {deleted_chembl:>12,}"
    )


# ============================================================
# 8. 최종 결과
# ============================================================

remaining_rows = (
    total_rows
    - deleted_chembl
)


print()
print("=" * 100)
print("DEDUP4 완료")
print("=" * 100)

print(f"입력 행 수                 : {total_rows:,}")
print(f"ChEMBL 삭제 행 수          : {deleted_chembl:,}")
print(f"ChEMBL 유지 행 수          : {kept_chembl:,}")
print(f"BindingDB 유지 행 수       : {kept_bindingdb:,}")
print(f"기타 Source 유지 행 수     : {kept_other:,}")
print(f"비교 불가 ChEMBL 행 수     : {invalid_affinity_chembl:,}")
print(f"최종 행 수                 : {remaining_rows:,}")


# ============================================================
# 9. QC 저장
# ============================================================

qc_df = pd.DataFrame(
    qc_records,
    columns=[
        "chunk",
        "deleted_chembl_rows",
        "reason"
    ]
)

qc_df.to_csv(
    QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. Source 최종 검증
# ============================================================

print()
print("=" * 100)
print("최종 Source 검증")
print("=" * 100)


source_counts = {}


for chunk in pd.read_csv(
    OUTPUT_FILE,
    dtype=str,
    keep_default_na=False,
    usecols=["source"],
    chunksize=CHUNK_SIZE
):

    counts = chunk[
        "source"
    ].value_counts()

    for source, count in counts.items():

        source_counts[source] = (
            source_counts.get(source, 0)
            + int(count)
        )


for source, count in source_counts.items():

    print(
        f"{source:15s}: {count:,}"
    )


print()
print(f"결과 파일 : {OUTPUT_FILE}")
print(f"QC 파일    : {QC_FILE}")
print("=" * 100)