import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\uniprot_protein_reference_sequence_fixed.csv"
)

OUTPUT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\uniprot_protein_reference_sequence_fixed_v2.csv"
)

REPORT_FILE = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB\uniprot_sequence_recovery_report_v2.csv"
)

MIN_SEQUENCE_LENGTH = 80


# ============================================================
# AMINO ACID VALIDATION
# ============================================================
#
# Standard amino acids:
# A C D E F G H I K L M N P Q R S T V W Y
#
# Additional valid symbols:
# X = unknown amino acid
# U = selenocysteine
# O = pyrrolysine
#
# whitespace / line break는 제거한 뒤 검사
#

VALID_AA = set(
    "ACDEFGHIKLMNPQRSTVWYOUX"
)


def clean_sequence(value):
    """
    문자열에서 whitespace를 제거.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    # 모든 whitespace 제거
    value = re.sub(r"\s+", "", value)

    return value.upper()


def is_valid_amino_acid_sequence(value, min_length=MIN_SEQUENCE_LENGTH):
    """
    실제 protein sequence인지 검사.

    조건:
    1. 문자열
    2. 길이 >= 80
    3. 모든 문자가 허용 amino-acid 문자
    """

    seq = clean_sequence(value)

    if len(seq) < min_length:
        return False

    return all(aa in VALID_AA for aa in seq)


def is_short_amino_acid_sequence(value):
    """
    80 aa 미만이지만 amino-acid 문자만으로 구성된 값인지 확인.
    """

    seq = clean_sequence(value)

    if not seq:
        return False

    if len(seq) >= MIN_SEQUENCE_LENGTH:
        return False

    return all(aa in VALID_AA for aa in seq)


def excel_col(index):
    """
    0-based index -> Excel column
    예:
    0 -> A
    1 -> B
    25 -> Z
    26 -> AA
    """

    result = ""

    index += 1

    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result

    return result


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("Protein Sequence Recovery - V2")
print("=" * 100)

print(f"Input : {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

original_df = df.copy(deep=True)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print()
print("Current columns:")

for i, col in enumerate(df.columns):
    print(
        f"  {excel_col(i)} "
        f"(index={i:02d}) : {col}"
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

if "uniprot_id" not in df.columns:
    raise ValueError(
        "'uniprot_id' column이 없습니다."
    )

if "protein_sequence" not in df.columns:
    raise ValueError(
        "'protein_sequence' column이 없습니다."
    )

if df.columns.tolist().count("uniprot_id") != 1:
    raise ValueError(
        "'uniprot_id' 컬럼이 중복되어 있습니다. "
        "자동 삭제하지 않습니다."
    )

if df.columns.tolist().count("protein_sequence") != 1:
    raise ValueError(
        "'protein_sequence' 컬럼이 중복되어 있습니다. "
        "자동 삭제하지 않습니다."
    )


# ============================================================
# ORIGINAL STATE
# ============================================================

original_ids = (
    df["uniprot_id"]
    .astype("string")
    .tolist()
)

original_sequence = (
    df["protein_sequence"]
    .copy()
)


# ============================================================
# SEARCH COLUMNS
# ============================================================

# uniprot_id는 검색하지 않음
search_columns = [
    col
    for col in df.columns
    if col != "uniprot_id"
]

print()
print("=" * 100)
print("SEARCH ORDER")
print("=" * 100)

for i, col in enumerate(search_columns, start=1):
    col_index = df.columns.get_loc(col)

    print(
        f"{i:02d}. "
        f"{excel_col(col_index)} "
        f"(index={col_index:02d}) : {col}"
    )


# ============================================================
# BEFORE QC
# ============================================================

before_valid = 0
before_short = 0
before_invalid = 0
before_missing = 0

for value in df["protein_sequence"]:

    if pd.isna(value) or str(value).strip() == "":
        before_missing += 1

    elif is_valid_amino_acid_sequence(value):
        before_valid += 1

    elif is_short_amino_acid_sequence(value):
        before_short += 1

    else:
        before_invalid += 1


print()
print("=" * 100)
print("BEFORE QC")
print("=" * 100)

print(f"Valid sequence >= {MIN_SEQUENCE_LENGTH} aa : {before_valid:,}")
print(f"Short sequence < {MIN_SEQUENCE_LENGTH} aa  : {before_short:,}")
print(f"Invalid sequence                           : {before_invalid:,}")
print(f"Missing                                    : {before_missing:,}")


# ============================================================
# RECOVERY
# ============================================================

recovery_records = []

recovered_count = 0
existing_valid_count = 0
still_missing_count = 0

short_existing_count = 0
invalid_existing_count = 0


for row_idx in range(len(df)):

    row = df.iloc[row_idx]

    uniprot_id = row["uniprot_id"]

    old_value = row["protein_sequence"]

    old_clean = clean_sequence(old_value)

    # --------------------------------------------------------
    # 1. F열이 정상 sequence면 유지
    # --------------------------------------------------------

    if is_valid_amino_acid_sequence(old_value):

        existing_valid_count += 1

        recovery_records.append({
            "row_index": row_idx,
            "uniprot_id": uniprot_id,
            "old_sequence_length": len(old_clean),
            "new_sequence_length": len(old_clean),
            "old_source": "protein_sequence",
            "new_source": "protein_sequence",
            "source_column": "protein_sequence",
            "source_excel_column": excel_col(
                df.columns.get_loc("protein_sequence")
            ),
            "action": "preserved",
            "status": "valid"
        })

        continue

    # --------------------------------------------------------
    # 2. 기존 F열 상태 기록
    # --------------------------------------------------------

    if not old_clean:
        old_status = "missing"

    elif is_short_amino_acid_sequence(old_value):
        old_status = "short_sequence"
        short_existing_count += 1

    else:
        old_status = "invalid_sequence"
        invalid_existing_count += 1

    # --------------------------------------------------------
    # 3. 같은 행의 모든 column을 왼쪽 -> 오른쪽 탐색
    # --------------------------------------------------------

    found_sequence = None
    found_column = None
    found_column_index = None

    for col in search_columns:

        # 기존 protein_sequence도 검사하지만
        # 위에서 이미 invalid/short로 판정되었으므로
        # 다시 선택될 가능성은 없음
        value = row[col]

        if is_valid_amino_acid_sequence(value):

            candidate = clean_sequence(value)

            found_sequence = candidate
            found_column = col
            found_column_index = df.columns.get_loc(col)

            break

    # --------------------------------------------------------
    # 4. sequence 발견
    # --------------------------------------------------------

    if found_sequence is not None:

        df.at[row_idx, "protein_sequence"] = found_sequence

        # sequence_length도 실제 sequence 길이로 수정
        if "sequence_length" in df.columns:
            df.at[
                row_idx,
                "sequence_length"
            ] = len(found_sequence)

        recovered_count += 1

        recovery_records.append({
            "row_index": row_idx,
            "uniprot_id": uniprot_id,
            "old_sequence_length": len(old_clean),
            "new_sequence_length": len(found_sequence),
            "old_source": "protein_sequence",
            "new_source": found_column,
            "source_column": found_column,
            "source_excel_column": excel_col(
                found_column_index
            ),
            "action": "recovered_from_row",
            "status": "recovered"
        })

    # --------------------------------------------------------
    # 5. 같은 행에서도 못 찾음
    # --------------------------------------------------------

    else:

        still_missing_count += 1

        recovery_records.append({
            "row_index": row_idx,
            "uniprot_id": uniprot_id,
            "old_sequence_length": len(old_clean),
            "new_sequence_length": len(old_clean),
            "old_source": "protein_sequence",
            "new_source": "",
            "source_column": "",
            "source_excel_column": "",
            "action": "not_recovered",
            "status": old_status
        })


# ============================================================
# AFTER QC
# ============================================================

after_valid = 0
after_short = 0
after_invalid = 0
after_missing = 0

remaining_problem_rows = []

for row_idx, value in enumerate(df["protein_sequence"]):

    if pd.isna(value) or str(value).strip() == "":

        after_missing += 1

        remaining_problem_rows.append({
            "row_index": row_idx,
            "uniprot_id": df.iloc[row_idx]["uniprot_id"],
            "problem": "missing",
            "sequence_length": 0,
            "protein_sequence": ""
        })

    elif is_valid_amino_acid_sequence(value):

        after_valid += 1

    elif is_short_amino_acid_sequence(value):

        after_short += 1

        remaining_problem_rows.append({
            "row_index": row_idx,
            "uniprot_id": df.iloc[row_idx]["uniprot_id"],
            "problem": "short_sequence",
            "sequence_length": len(
                clean_sequence(value)
            ),
            "protein_sequence": value
        })

    else:

        after_invalid += 1

        remaining_problem_rows.append({
            "row_index": row_idx,
            "uniprot_id": df.iloc[row_idx]["uniprot_id"],
            "problem": "invalid_sequence",
            "sequence_length": len(
                clean_sequence(value)
            ),
            "protein_sequence": value
        })


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 100)
print("RECOVERY RESULT")
print("=" * 100)

print(
    f"Existing valid sequence : "
    f"{existing_valid_count:,}"
)

print(
    f"Recovered from row      : "
    f"{recovered_count:,}"
)

print(
    f"Still missing/problem   : "
    f"{still_missing_count:,}"
)


print()
print("=" * 100)
print("AFTER QC")
print("=" * 100)

print(
    f"Valid sequence >= {MIN_SEQUENCE_LENGTH} aa : "
    f"{after_valid:,}"
)

print(
    f"Short sequence < {MIN_SEQUENCE_LENGTH} aa  : "
    f"{after_short:,}"
)

print(
    f"Invalid sequence                           : "
    f"{after_invalid:,}"
)

print(
    f"Missing                                    : "
    f"{after_missing:,}"
)


# ============================================================
# REMAINING PROBLEM IDS
# ============================================================

if remaining_problem_rows:

    print()
    print("=" * 100)
    print("REMAINING PROBLEM SEQUENCES")
    print("=" * 100)

    print(
        f"Count : {len(remaining_problem_rows):,}"
    )

    print()

    for item in remaining_problem_rows[:100]:

        print(
            f"{item['uniprot_id']:10s} | "
            f"{item['problem']:18s} | "
            f"length={item['sequence_length']:4d} | "
            f"{str(item['protein_sequence'])[:80]}"
        )

else:

    print()
    print("=" * 100)
    print("REMAINING PROBLEM SEQUENCES")
    print("=" * 100)

    print("Count : 0")


# ============================================================
# VALIDATE ORIGINAL VALID SEQUENCES WERE NOT CHANGED
# ============================================================

changed_original_valid = []

for row_idx in range(len(original_df)):

    old_value = original_sequence.iloc[row_idx]

    if is_valid_amino_acid_sequence(old_value):

        new_value = df.iloc[row_idx]["protein_sequence"]

        old_seq = clean_sequence(old_value)
        new_seq = clean_sequence(new_value)

        if old_seq != new_seq:

            changed_original_valid.append({
                "row_index": row_idx,
                "uniprot_id": original_ids[row_idx],
                "old_sequence": old_seq,
                "new_sequence": new_seq
            })


if changed_original_valid:

    print()
    print(
        "[ERROR] Existing valid sequences were changed!"
    )

    for item in changed_original_valid[:10]:

        print(
            item["row_index"],
            item["uniprot_id"]
        )

    raise RuntimeError(
        "기존 정상 sequence가 변경되었습니다."
    )

else:

    print()
    print(
        "[PASS] Existing valid sequences preserved."
    )


# ============================================================
# VALIDATE RECOVERED SEQUENCES
# ============================================================

invalid_recovered = []

for item in recovery_records:

    if item["action"] != "recovered_from_row":
        continue

    row_idx = item["row_index"]

    value = df.iloc[row_idx]["protein_sequence"]

    if not is_valid_amino_acid_sequence(value):

        invalid_recovered.append({
            "row_index": row_idx,
            "uniprot_id": item["uniprot_id"],
            "sequence": value
        })


if invalid_recovered:

    print()
    print(
        "[ERROR] Recovered sequence validation failed."
    )

    raise RuntimeError(
        f"Invalid recovered sequences: "
        f"{len(invalid_recovered)}"
    )

else:

    print(
        "[PASS] Recovered sequences validated."
    )


# ============================================================
# VALIDATE ROW COUNT / ID ORDER
# ============================================================

if len(df) != len(original_df):

    raise RuntimeError(
        "Row count changed."
    )

new_ids = (
    df["uniprot_id"]
    .astype("string")
    .tolist()
)

if original_ids != new_ids:

    raise RuntimeError(
        "UniProt ID/order changed."
    )

print(
    "[PASS] Row count preserved."
)

print(
    "[PASS] UniProt ID/order preserved."
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

report_df = pd.DataFrame(
    recovery_records
)

report_df.to_csv(
    REPORT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# POST-SAVE VERIFICATION
# ============================================================

print()
print("=" * 100)
print("POST-SAVE VERIFICATION")
print("=" * 100)

check_df = pd.read_csv(
    OUTPUT_FILE,
    low_memory=False
)

if len(check_df) != len(df):

    raise RuntimeError(
        "Saved file row count mismatch."
    )

check_ids = (
    check_df["uniprot_id"]
    .astype("string")
    .tolist()
)

if check_ids != original_ids:

    raise RuntimeError(
        "Saved file UniProt ID/order mismatch."
    )


check_valid = sum(
    is_valid_amino_acid_sequence(x)
    for x in check_df["protein_sequence"]
)

check_short = sum(
    is_short_amino_acid_sequence(x)
    for x in check_df["protein_sequence"]
)

check_missing = sum(
    pd.isna(x) or str(x).strip() == ""
    for x in check_df["protein_sequence"]
)

print(
    "[PASS] Saved row count preserved."
)

print(
    "[PASS] Saved UniProt ID/order preserved."
)

print(
    f"Saved valid sequence >= 80 aa : "
    f"{check_valid:,}"
)

print(
    f"Saved short sequence < 80 aa  : "
    f"{check_short:,}"
)

print(
    f"Saved missing                  : "
    f"{check_missing:,}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    f"Total UniProt IDs       : {len(df):,}"
)

print(
    f"Valid sequence >=80 aa  : {check_valid:,}"
)

print(
    f"Short sequence <80 aa   : {check_short:,}"
)

print(
    f"Missing                  : {check_missing:,}"
)

print(
    f"Recovered from row       : {recovered_count:,}"
)

print()
print(
    f"Output:"
)

print(
    OUTPUT_FILE
)

print()
print(
    f"Report:"
)

print(
    REPORT_FILE
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)