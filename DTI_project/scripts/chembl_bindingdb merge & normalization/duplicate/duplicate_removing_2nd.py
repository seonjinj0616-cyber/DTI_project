import pandas as pd
from pathlib import Path


# ============================================================
# 1. 경로 설정
# ============================================================

INPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\output\merged_dti_DEDUP1.csv"
)

OUTPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\output\merged_dti_DEDUP2.csv"
)

QC_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\output\dedup2_qc.csv"
)

CHUNK_SIZE = 200_000


# ============================================================
# 2. 중복 판단 기준
# ============================================================
#
# 중요:
# smiles는 비교하지 않는다.
#
# chembl_id가 같으면서 아래 6개 조건까지 모두 같아야
# ChEMBL ↔ BindingDB 중복으로 판단한다.
#
# ============================================================

MATCH_KEYS = [
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "target_name",
    "affinity_relation",
    "affinity_value",
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
    col for col in REQUIRED_COLUMNS
    if col not in header.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )


print("=" * 100)
print("DEDUP2 : ChEMBL ↔ BindingDB 중복 제거")
print("=" * 100)

print()
print("중복 판단 기준:")
print("  chembl_id")
print("  uniprot_id")
print("  affinity_type")
print("  target_name")
print("  affinity_relation")
print("  affinity_value")
print("  affinity_unit")

print()
print("smiles는 중복 판단 기준에서 제외합니다.")
print("동일 조건이면 ChEMBL 삭제 / BindingDB 유지")
print("=" * 100)


# ============================================================
# 5. PASS 1
#
# BindingDB에서 존재하는 7-key를 모두 수집한다.
#
# 이 정보를 기준으로 PASS 2에서
# ChEMBL의 동일 key를 삭제한다.
#
# 파일의 행 순서와 관계없이 항상
# BindingDB를 우선할 수 있다.
# ============================================================

print()
print("=" * 100)
print("PASS 1 : BindingDB key 수집")
print("=" * 100)

bindingdb_keys = set()

total_rows_pass1 = 0
bindingdb_rows_pass1 = 0


for chunk_idx, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    )
):

    total_rows_pass1 += len(chunk)

    bindingdb = chunk[
        chunk["source"].eq("BindingDB")
    ]

    bindingdb_rows_pass1 += len(bindingdb)

    if not bindingdb.empty:

        keys = bindingdb[
            MATCH_KEYS
        ].itertuples(
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
print(f"전체 입력 행 수       : {total_rows_pass1:,}")
print(f"BindingDB 행 수        : {bindingdb_rows_pass1:,}")
print(f"BindingDB unique 7-key : {len(bindingdb_keys):,}")


# ============================================================
# 6. PASS 2
#
# ChEMBL:
#   BindingDB에 동일한 7-key가 존재하면 삭제
#
# BindingDB:
#   절대 삭제하지 않는다.
#
# 기타 Source:
#   그대로 유지한다.
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

    chembl_mask = chunk["source"].eq("ChEMBL")
    bindingdb_mask = chunk["source"].eq("BindingDB")

    other_mask = (
        ~chembl_mask
        & ~bindingdb_mask
    )


    # --------------------------------------------------------
    # ChEMBL 검사
    # --------------------------------------------------------

    chembl = chunk.loc[
        chembl_mask
    ].copy()

    deleted_count = 0
    kept_count = 0


    if not chembl.empty:

        chembl_keys = chembl[
            MATCH_KEYS
        ].itertuples(
            index=False,
            name=None
        )

        delete_mask = [
            key in bindingdb_keys
            for key in chembl_keys
        ]

        delete_mask = pd.Series(
            delete_mask,
            index=chembl.index
        )

        delete_indices = chembl.index[
            delete_mask
        ]

        deleted_count = len(
            delete_indices
        )

        kept_count = len(chembl) - deleted_count

        # ChEMBL만 삭제
        chunk = chunk.drop(
            index=delete_indices
        )


    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    deleted_chembl += deleted_count
    kept_chembl += kept_count

    kept_bindingdb += int(
        bindingdb_mask.sum()
    )

    kept_other += int(
        other_mask.sum()
    )


    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    if deleted_count > 0:

        qc_records.append({
            "chunk": chunk_idx,
            "deleted_chembl_rows": deleted_count,
            "reason": (
                "chembl_id + "
                "uniprot_id + "
                "affinity_type + "
                "target_name + "
                "affinity_relation + "
                "affinity_value + "
                "affinity_unit "
                "matched with BindingDB"
            )
        })


    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

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
# 7. 최종 결과
# ============================================================

remaining_rows = (
    total_rows - deleted_chembl
)


print()
print("=" * 100)
print("DEDUP2 완료")
print("=" * 100)

print(f"입력 행 수                 : {total_rows:,}")
print(f"ChEMBL 삭제 행 수          : {deleted_chembl:,}")
print(f"ChEMBL 유지 행 수          : {kept_chembl:,}")
print(f"BindingDB 유지 행 수       : {kept_bindingdb:,}")
print(f"기타 Source 유지 행 수     : {kept_other:,}")
print(f"최종 행 수                 : {remaining_rows:,}")


# ============================================================
# 8. QC 파일 저장
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
# 9. 결과 Source 검증
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