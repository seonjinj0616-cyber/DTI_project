import os
import time
import requests
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================

DATA_DIR = r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"

FILES = [
    "IC50.csv",
    "Ki.csv",
    "Kd.csv"
]

# API batch 크기
BATCH_SIZE = 100

# API 요청 간 대기시간
SLEEP_TIME = 0.2

# UniProt REST API
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


# ============================================================
# 2. CSV에서 Unique UniProt ID 수집
# ============================================================

print("=" * 80)
print("1. Unique UniProt ID 수집")
print("=" * 80)

all_ids = set()

for file in FILES:

    path = os.path.join(DATA_DIR, file)

    df = pd.read_csv(
        path,
        usecols=["uniprot_id"],
        low_memory=False
    )

    ids = (
        df["uniprot_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    ids = ids[ids != ""]

    print(
        f"{file:<10} : "
        f"{len(ids):>10,} rows / "
        f"{ids.nunique():>6,} unique UniProt"
    )

    all_ids.update(ids.tolist())


all_ids = sorted(all_ids)

print()
print("-" * 80)
print(f"전체 Unique UniProt ID : {len(all_ids):,}")
print("-" * 80)


# ============================================================
# 3. UniProt 정보 조회 함수
# ============================================================

def fetch_uniprot_batch(uniprot_ids):

    query = " OR ".join(
        f"accession:{uid}"
        for uid in uniprot_ids
    )

    params = {
        "query": query,
        "format": "tsv",
        "fields": (
            "accession,"
            "protein_name,"
            "gene_names,"
            "organism_name,"
            "reviewed,"
            "sequence,"
            "length"
        ),
        "size": len(uniprot_ids)
    }

    response = requests.get(
        UNIPROT_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    return response.text


# ============================================================
# 4. UniProt 4,767개 조회
# ============================================================

print()
print("=" * 80)
print("2. UniProt protein sequence 조회")
print("=" * 80)

protein_records = []

total = len(all_ids)

for start in range(0, total, BATCH_SIZE):

    batch_ids = all_ids[start:start + BATCH_SIZE]

    batch_number = start // BATCH_SIZE + 1
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(
        f"[Batch {batch_number}/{total_batches}] "
        f"{start + 1:,} ~ "
        f"{min(start + BATCH_SIZE, total):,} / "
        f"{total:,}"
    )

    try:

        text = fetch_uniprot_batch(batch_ids)

        lines = text.strip().splitlines()

        if len(lines) <= 1:
            print("  -> 결과 없음")
            continue

        header = lines[0].split("\t")

        for line in lines[1:]:

            values = line.split("\t")

            if len(values) != len(header):
                continue

            record = dict(zip(header, values))

            protein_records.append(record)

        print(
            f"  -> UniProt 결과 "
            f"{len(lines) - 1:,}개"
        )

    except Exception as e:

        print(
            f"  !! Batch 오류: {e}"
        )

    time.sleep(SLEEP_TIME)


# ============================================================
# 5. Protein DataFrame 생성
# ============================================================

protein_df = pd.DataFrame(protein_records)


print()
print("=" * 80)
print("3. UniProt 조회 결과")
print("=" * 80)

print(
    f"조회된 protein record : "
    f"{len(protein_df):,}"
)

if not protein_df.empty:
    print(
        f"조회된 unique UniProt : "
        f"{protein_df['Entry'].nunique():,}"
    )


# ============================================================
# 6. 컬럼명 정리
# ============================================================

rename_dict = {
    "Entry": "uniprot_id",
    "Protein names": "protein_name",
    "Gene Names": "gene_name",
    "Organism": "organism",
    "Reviewed": "reviewed",
    "Sequence": "protein_sequence",
    "Length": "sequence_length"
}

protein_df = protein_df.rename(
    columns=rename_dict
)


# ============================================================
# 7. 필요한 컬럼만 유지
# ============================================================

protein_columns = [
    "uniprot_id",
    "protein_name",
    "gene_name",
    "organism",
    "reviewed",
    "protein_sequence",
    "sequence_length"
]

protein_df = protein_df[
    [
        col
        for col in protein_columns
        if col in protein_df.columns
    ]
].copy()


# ============================================================
# 8. 문자열 정리
# ============================================================

protein_df["uniprot_id"] = (
    protein_df["uniprot_id"]
    .astype(str)
    .str.strip()
)

protein_df["protein_sequence"] = (
    protein_df["protein_sequence"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 9. 중복 UniProt 제거
# ============================================================

protein_df = (
    protein_df
    .drop_duplicates(
        subset=["uniprot_id"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. 조회 실패한 UniProt 확인
# ============================================================

found_ids = set(
    protein_df["uniprot_id"]
)

missing_ids = sorted(
    set(all_ids) - found_ids
)


print()
print("=" * 80)
print("4. UniProt 매칭 QC")
print("=" * 80)

print(
    f"전체 요청 ID       : {len(all_ids):,}"
)

print(
    f"정상 매칭 ID       : {len(found_ids):,}"
)

print(
    f"매칭 실패 ID       : {len(missing_ids):,}"
)


if missing_ids:

    print()
    print("매칭 실패 UniProt ID:")

    for uid in missing_ids:
        print(uid)


# ============================================================
# 11. Protein reference 저장
#
# 참고:
# 최종 IC50/Ki/Kd에는 직접 sequence를 넣을 것이지만
# QC 및 재사용을 위해 reference도 하나 저장
# ============================================================

REFERENCE_FILE = os.path.join(
    DATA_DIR,
    "uniprot_protein_reference.csv"
)

protein_df.to_csv(
    REFERENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    f"Protein reference 저장 완료:"
)
print(REFERENCE_FILE)


# ============================================================
# 12. Dictionary 생성
# ============================================================

protein_mapping = (
    protein_df
    .set_index("uniprot_id")
    .to_dict("index")
)


# ============================================================
# 13. IC50 / Ki / Kd 각각에 Protein 정보 추가
# ============================================================

print()
print("=" * 80)
print("5. IC50 / Ki / Kd Protein Sequence 매칭")
print("=" * 80)


for file in FILES:

    input_path = os.path.join(
        DATA_DIR,
        file
    )

    print()
    print(f"[처리 시작] {file}")

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    # --------------------------------------------------------
    # UniProt ID 정리
    # --------------------------------------------------------

    df["uniprot_id"] = (
        df["uniprot_id"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Protein 정보 각각 매칭
    # --------------------------------------------------------

    df["protein_name"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "protein_name"
            )
            if pd.notna(x)
            else None
        )
    )

    df["gene_name"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "gene_name"
            )
            if pd.notna(x)
            else None
        )
    )

    df["organism"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "organism"
            )
            if pd.notna(x)
            else None
        )
    )

    df["reviewed"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "reviewed"
            )
            if pd.notna(x)
            else None
        )
    )

    df["protein_sequence"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "protein_sequence"
            )
            if pd.notna(x)
            else None
        )
    )

    df["sequence_length"] = (
        df["uniprot_id"]
        .map(
            lambda x:
            protein_mapping.get(x, {}).get(
                "sequence_length"
            )
            if pd.notna(x)
            else None
        )
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    matched = df["protein_sequence"].notna().sum()
    unmatched = df["protein_sequence"].isna().sum()

    print(
        f"전체 rows       : {len(df):,}"
    )

    print(
        f"Sequence 매칭   : {matched:,}"
    )

    print(
        f"Sequence 실패   : {unmatched:,}"
    )

    if len(df) > 0:

        print(
            f"매칭률          : "
            f"{matched / len(df) * 100:.4f}%"
        )

    # --------------------------------------------------------
    # 원본 파일 덮어쓰기
    # --------------------------------------------------------

    df.to_csv(
        input_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"저장 완료        : {input_path}"
    )


# ============================================================
# 14. 최종 완료
# ============================================================

print()
print("=" * 80)
print("전체 UniProt Protein Sequence 매칭 완료")
print("=" * 80)

print(
    f"Unique UniProt : {len(all_ids):,}"
)

print(
    f"Reference      : {REFERENCE_FILE}"
)

print()
print("IC50.csv / Ki.csv / Kd.csv 각각에")
print("protein_sequence 및 Protein metadata가 추가되었습니다.")