import pandas as pd
import os


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\output"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "merged_dti_DEDUP5.csv"
)

RELATION_FILE = os.path.join(
    BASE_DIR,
    "chembl_bindingdb_ALL_OVERLAPPING_ROWS_HEURISTIC_RELATION_CLEAN_FINAL_EQUALIZED.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "merged_dti_DEDUP6.csv"
)

QC_FILE = os.path.join(
    BASE_DIR,
    "merged_dti_DEDUP6_QC.csv"
)

CHUNK_SIZE = 200_000


# ============================================================
# DEDUP2 기준
# ============================================================

DEDUP2_KEYS = [
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "target_name",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_source(x):

    if pd.isna(x):
        return ""

    return str(x).strip().lower()


def normalize_record_id(x):

    if pd.isna(x):
        return ""

    s = str(x).strip()

    try:
        f = float(s)

        if f.is_integer():
            return str(int(f))

        return format(f, ".15g")

    except Exception:
        return s


def make_key(row):

    return tuple(
        "<NA>" if pd.isna(x)
        else str(x).strip()
        for x in row
    )


# ============================================================
# 1. RELATION CLEAN FILE LOAD
# ============================================================

print("=" * 100)
print("1. Relation Clean 파일 로드")
print("=" * 100)

relation_df = pd.read_csv(
    RELATION_FILE,
    low_memory=False
)

print(
    f"Relation clean rows: {len(relation_df):,}"
)


required_relation_cols = [
    "source",
    "source_record_id",
    "affinity_relation"
]

missing = [
    c for c in required_relation_cols
    if c not in relation_df.columns
]

if missing:
    raise ValueError(
        f"Relation 파일에 필요한 컬럼이 없습니다: {missing}"
    )


# ============================================================
# 2. RELATION MAP
# ============================================================

relation_df["_source_norm"] = (
    relation_df["source"]
    .apply(normalize_source)
)

relation_df["_record_id_norm"] = (
    relation_df["source_record_id"]
    .apply(normalize_record_id)
)

relation_df["_match_key"] = (
    relation_df["_source_norm"]
    + "||"
    + relation_df["_record_id_norm"]
)

relation_df["clean_relation"] = (
    relation_df["affinity_relation"]
    .astype(str)
    .str.strip()
)


# 동일 source + source_record_id 검사
duplicate_mapping = relation_df[
    relation_df["_match_key"].duplicated(
        keep=False
    )
]

if len(duplicate_mapping) > 0:

    relation_conflict = (
        duplicate_mapping
        .groupby("_match_key")["clean_relation"]
        .nunique()
    )

    conflict_groups = (
        relation_conflict > 1
    ).sum()

    if conflict_groups > 0:

        raise ValueError(
            "동일 source + source_record_id에 "
            f"서로 다른 relation이 존재합니다: "
            f"{conflict_groups:,} groups"
        )

    relation_df = relation_df.drop_duplicates(
        subset=["_match_key"],
        keep="first"
    )


relation_map = dict(
    zip(
        relation_df["_match_key"],
        relation_df["clean_relation"]
    )
)

print(
    f"Unique relation mappings: "
    f"{len(relation_map):,}"
)


# ============================================================
# 3. INPUT CHECK
# ============================================================

header = pd.read_csv(
    INPUT_FILE,
    nrows=0
)

required_input_cols = [
    "source",
    "source_record_id",
    "affinity_relation"
] + DEDUP2_KEYS

missing = [
    c for c in required_input_cols
    if c not in header.columns
]

if missing:
    raise ValueError(
        f"DEDUP5에 필요한 컬럼이 없습니다: {missing}"
    )


# ============================================================
# 4. PASS 1
#
# Relation 변경 후 BindingDB의 DEDUP2 key만 수집
#
# !!! 중요 !!!
# BindingDB 내부 중복을 삭제하기 위한 것이 아님.
#
# 목적:
# ChEMBL row의 key가 BindingDB에 존재하는지만 확인하기 위함.
# ============================================================

print("\n" + "=" * 100)
print("2. PASS 1 - Relation 적용 후 BindingDB key 수집")
print("=" * 100)

bindingdb_keys = set()

total_pass1 = 0
relation_matched = 0
relation_changed = 0
bindingdb_rows = 0

for chunk_no, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    total_pass1 += len(chunk)

    # --------------------------------------------------------
    # source + source_record_id
    # --------------------------------------------------------

    source_norm = (
        chunk["source"]
        .apply(normalize_source)
    )

    record_id_norm = (
        chunk["source_record_id"]
        .apply(normalize_record_id)
    )

    match_key = (
        source_norm
        + "||"
        + record_id_norm
    )

    # --------------------------------------------------------
    # relation 적용
    # --------------------------------------------------------

    original_relation = (
        chunk["affinity_relation"]
        .astype(str)
        .str.strip()
    )

    mapped_relation = match_key.map(
        relation_map
    )

    matched_mask = mapped_relation.notna()

    changed_mask = (
        matched_mask
        &
        (
            original_relation
            != mapped_relation
        )
    )

    relation_matched += int(
        matched_mask.sum()
    )

    relation_changed += int(
        changed_mask.sum()
    )

    chunk.loc[
        matched_mask,
        "affinity_relation"
    ] = mapped_relation[
        matched_mask
    ]

    # --------------------------------------------------------
    # BindingDB만 key 저장
    # --------------------------------------------------------

    bindingdb_mask = (
        source_norm == "bindingdb"
    )

    bdb = chunk.loc[
        bindingdb_mask,
        DEDUP2_KEYS
    ]

    bindingdb_rows += len(bdb)

    for row in bdb.itertuples(
        index=False,
        name=None
    ):

        bindingdb_keys.add(
            make_key(row)
        )

    print(
        f"Chunk {chunk_no}: "
        f"rows={len(chunk):,} | "
        f"relation changed={changed_mask.sum():,} | "
        f"BindingDB rows={len(bdb):,} | "
        f"unique BDB keys={len(bindingdb_keys):,}"
    )


print("\nPASS 1 완료")

print(
    f"전체 rows:              {total_pass1:,}"
)

print(
    f"Relation matched:       {relation_matched:,}"
)

print(
    f"Relation changed:       {relation_changed:,}"
)

print(
    f"BindingDB rows:         {bindingdb_rows:,}"
)

print(
    f"Unique BindingDB keys:  {len(bindingdb_keys):,}"
)


# ============================================================
# 5. PASS 2
#
# 실제 삭제
#
# ONLY:
#
# ChEMBL key == BindingDB key
#       ↓
# ChEMBL DELETE
#
# BindingDB:
#       절대 삭제하지 않음
#
# ============================================================

print("\n" + "=" * 100)
print("3. PASS 2 - ChEMBL ↔ BindingDB 중복 제거")
print("=" * 100)


if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

if os.path.exists(QC_FILE):
    os.remove(QC_FILE)


total_rows = 0
kept_rows = 0
deleted_rows = 0

deleted_chembl = 0

relation_matched_pass2 = 0
relation_changed_pass2 = 0

qc_list = []

first_chunk = True


for chunk_no, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    total_rows += len(chunk)

    # --------------------------------------------------------
    # Relation 적용
    # --------------------------------------------------------

    source_norm = (
        chunk["source"]
        .apply(normalize_source)
    )

    record_id_norm = (
        chunk["source_record_id"]
        .apply(normalize_record_id)
    )

    match_key = (
        source_norm
        + "||"
        + record_id_norm
    )

    original_relation = (
        chunk["affinity_relation"]
        .astype(str)
        .str.strip()
    )

    mapped_relation = match_key.map(
        relation_map
    )

    matched_mask = mapped_relation.notna()

    changed_mask = (
        matched_mask
        &
        (
            original_relation
            != mapped_relation
        )
    )

    relation_matched_pass2 += int(
        matched_mask.sum()
    )

    relation_changed_pass2 += int(
        changed_mask.sum()
    )

    chunk.loc[
        matched_mask,
        "affinity_relation"
    ] = mapped_relation[
        matched_mask
    ]

    # --------------------------------------------------------
    # 기본적으로 모든 행 KEEP
    # --------------------------------------------------------

    keep_mask = pd.Series(
        True,
        index=chunk.index
    )

    delete_reason = pd.Series(
        "",
        index=chunk.index,
        dtype="object"
    )

    # --------------------------------------------------------
    # ChEMBL만 검사
    # --------------------------------------------------------

    chembl_mask = (
        source_norm == "chembl"
    )

    chembl_indices = chunk.index[
        chembl_mask
    ]

    # --------------------------------------------------------
    # ChEMBL key가 BindingDB에 있으면 삭제
    # --------------------------------------------------------

    for idx in chembl_indices:

        row = chunk.loc[
            idx,
            DEDUP2_KEYS
        ]

        key = make_key(
            row.tolist()
        )

        if key in bindingdb_keys:

            keep_mask.loc[idx] = False

            delete_reason.loc[idx] = (
                "BINDINGDB_PRIORITY"
            )

            deleted_chembl += 1

    # --------------------------------------------------------
    # KEEP / DELETE
    # --------------------------------------------------------

    chunk_keep = chunk.loc[
        keep_mask
    ].copy()

    chunk_delete = chunk.loc[
        ~keep_mask
    ].copy()

    chunk_kept = len(chunk_keep)
    chunk_deleted = len(chunk_delete)

    kept_rows += chunk_kept
    deleted_rows += chunk_deleted

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    if chunk_deleted > 0:

        delete_qc = chunk_delete[
            [
                "source",
                "source_record_id",
                "chembl_id",
                "uniprot_id",
                "affinity_type",
                "target_name",
                "affinity_relation",
                "affinity_value",
                "affinity_unit"
            ]
        ].copy()

        delete_qc["delete_reason"] = (
            delete_reason.loc[
                ~keep_mask
            ].values
        )

        qc_list.append(
            delete_qc
        )

    # --------------------------------------------------------
    # KEEP rows 저장
    # --------------------------------------------------------

    chunk_keep.to_csv(
        OUTPUT_FILE,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
        encoding="utf-8-sig"
    )

    first_chunk = False

    print(
        f"Chunk {chunk_no}: "
        f"input={len(chunk):,} | "
        f"keep={chunk_kept:,} | "
        f"delete={chunk_deleted:,}"
    )


# ============================================================
# 6. QC 저장
# ============================================================

if qc_list:

    qc_df = pd.concat(
        qc_list,
        ignore_index=True
    )

else:

    qc_df = pd.DataFrame(
        columns=[
            "source",
            "source_record_id",
            "chembl_id",
            "uniprot_id",
            "affinity_type",
            "target_name",
            "affinity_relation",
            "affinity_value",
            "affinity_unit",
            "delete_reason"
        ]
    )


qc_df.to_csv(
    QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 최종 결과
# ============================================================

print("\n" + "=" * 100)
print("DEDUP6 완료")
print("=" * 100)

print(
    f"전체 입력:              {total_rows:,}"
)

print(
    f"최종 KEEP:              {kept_rows:,}"
)

print(
    f"최종 DELETE:            {deleted_rows:,}"
)

print(
    f"ChEMBL 삭제:            {deleted_chembl:,}"
)

print(
    f"BindingDB 삭제:         0"
)

print(
    f"Relation 변경:          {relation_changed_pass2:,}"
)

print("\n출력:")
print(OUTPUT_FILE)

print("\nQC:")
print(QC_FILE)

print("\n" + "=" * 100)
print("삭제 규칙")
print("=" * 100)

print("""
DEDUP2 기준:

chembl_id
uniprot_id
affinity_type
target_name
affinity_relation
affinity_value
affinity_unit

동일 key가

ChEMBL + BindingDB

양쪽에 존재할 경우:

    BindingDB → KEEP
    ChEMBL    → DELETE

BindingDB 내부 데이터:
    DELETE 하지 않음

ChEMBL 내부 데이터:
    DELETE 하지 않음

SMILES:
    중복 판정에 사용하지 않음

publication_id:
    중복 판정에 사용하지 않음

assay_id:
    중복 판정에 사용하지 않음
""")