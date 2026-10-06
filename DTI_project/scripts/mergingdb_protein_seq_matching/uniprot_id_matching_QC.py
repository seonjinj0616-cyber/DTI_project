import pandas as pd
import re
from pathlib import Path


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    r"C:\workspace\python\project_personal\DTI_project\data\esm2_bindingdb_chembl vectorDB"
)

FILES = {
    "IC50": BASE_DIR / "IC50_protein_matched_final.csv",
    "Ki": BASE_DIR / "Ki_protein_matched_final.csv",
    "Kd": BASE_DIR / "Kd_protein_matched_final.csv",
}


# ============================================================
# OUTPUT
# ============================================================

QC_SUMMARY_FILE = (
    BASE_DIR / "protein_sequence_length_qc_summary.csv"
)

PROBLEM_DETAIL_FILE = (
    BASE_DIR / "protein_sequence_problem_detail.csv"
)

MISSING_SEQUENCE_FILE = (
    BASE_DIR / "protein_sequence_missing.csv"
)

INVALID_SEQUENCE_FILE = (
    BASE_DIR / "protein_sequence_invalid_characters.csv"
)

LENGTH_MISMATCH_FILE = (
    BASE_DIR / "protein_sequence_length_mismatch.csv"
)

LONG_1024_FILE = (
    BASE_DIR / "esm2_long_protein_over_1024.csv"
)

LONG_1022_FILE = (
    BASE_DIR / "esm2_long_protein_over_1022.csv"
)

ESM2_PREPROCESS_ID_FILE = (
    BASE_DIR / "esm2_preprocessing_required_uniprot_ids.csv"
)


# ============================================================
# 설정
# ============================================================

USER_LENGTH_LIMIT = 1024

# ESM-2 standard 모델에서 residue 기준
ESM2_RESIDUE_LIMIT = 1022

# 표준 20개 amino acid
VALID_AA = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


# ============================================================
# 함수
# ============================================================

def normalize_uniprot_id(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if not value:
        return pd.NA

    # 예:
    # P24462 P24462 -> P24462
    # Q86V67 P27815 -> Q86V67
    return value.split()[0]


def clean_sequence(value):

    if value is None:
        return None

    # float NaN
    if isinstance(value, float) and pd.isna(value):
        return None

    # pandas NA / 기타 missing
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # 문자열이 아닌 경우 문자열로 변환
    seq = str(value).strip().upper()

    if not seq:
        return None

    # 공백 / 개행 제거
    seq = re.sub(r"\s+", "", seq)

    return seq


def check_invalid_aa(seq):

    if seq is None:
        return None

    if pd.isna(seq):
        return None

    seq = str(seq)

    invalid = sorted(set(seq) - VALID_AA)

    if invalid:
        return "".join(invalid)

    return None


# ============================================================
# 시작
# ============================================================

print("=" * 100)
print("PROTEIN SEQUENCE / LENGTH QC")
print("=" * 100)


summary_records = []

all_problem_records = []

all_missing_records = []

all_invalid_records = []

all_mismatch_records = []

all_long_1024_records = []

all_long_1022_records = []


# ============================================================
# Dataset별 검사
# ============================================================

for dataset_name, file_path in FILES.items():

    print()
    print("=" * 100)
    print(f"[{dataset_name}]")
    print("=" * 100)

    if not file_path.exists():

        print(
            f"[ERROR] 파일이 존재하지 않습니다:"
        )

        print(file_path)

        continue

    # --------------------------------------------------------
    # CSV 로드
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    print(
        f"파일: {file_path.name}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # 필수 컬럼 확인
    # --------------------------------------------------------

    required_columns = [
        "uniprot_id",
        "protein_sequence"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name}에 필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # 원본 UniProt ID 보존
    # --------------------------------------------------------

    df["_original_uniprot_id"] = (
        df["uniprot_id"]
    )

    df["uniprot_id"] = (
        df["uniprot_id"]
        .apply(normalize_uniprot_id)
    )


    # --------------------------------------------------------
    # Sequence 정리
    # --------------------------------------------------------

    df["protein_sequence"] = (
        df["protein_sequence"]
        .apply(clean_sequence)
    )


    # ========================================================
    # 1. Sequence missing
    # ========================================================

    missing_seq_mask = (
        df["protein_sequence"].isna()
    )

    missing_seq_count = int(
        missing_seq_mask.sum()
    )

    missing_unique_ids = (
        df.loc[
            missing_seq_mask,
            "uniprot_id"
        ]
        .dropna()
        .nunique()
    )


    print()
    print(
        f"Protein sequence missing: "
        f"{missing_seq_count:,}"
    )

    print(
        f"Missing sequence unique UniProt ID: "
        f"{missing_unique_ids:,}"
    )


    if missing_seq_count > 0:

        missing_df = df.loc[
            missing_seq_mask,
            [
                "uniprot_id",
                "_original_uniprot_id",
                "protein_sequence"
            ]
        ].copy()

        missing_df.insert(
            0,
            "dataset",
            dataset_name
        )

        missing_df.insert(
            1,
            "problem_type",
            "SEQUENCE_MISSING"
        )

        all_missing_records.append(
            missing_df
        )

        all_problem_records.append(
            missing_df
        )

        print()
        print(
            ">>> Missing sequence UniProt IDs"
        )

        print(
            missing_df[
                [
                    "uniprot_id",
                    "_original_uniprot_id"
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )


    # ========================================================
    # 2. 실제 sequence length
    # ========================================================

    df["calculated_protein_length"] = (
        df["protein_sequence"]
        .apply(
            lambda x: len(x) if isinstance(x, str) else pd.NA
                if x is not None
                else pd.NA
        )
    )

    df["calculated_protein_length"] = (
        pd.to_numeric(
            df["calculated_protein_length"],
            errors="coerce"
        )
        .astype("Int64")
    )


    # --------------------------------------------------------
    # Length statistics
    # --------------------------------------------------------

    lengths = (
        df.loc[
            ~missing_seq_mask,
            "calculated_protein_length"
        ]
        .dropna()
    )


    if len(lengths) > 0:

        min_length = int(
            lengths.min()
        )

        max_length = int(
            lengths.max()
        )

        mean_length = float(
            lengths.mean()
        )

        median_length = float(
            lengths.median()
        )

    else:

        min_length = pd.NA
        max_length = pd.NA
        mean_length = pd.NA
        median_length = pd.NA


    print(
        f"Protein length min: "
        f"{min_length}"
    )

    print(
        f"Protein length max: "
        f"{max_length}"
    )

    print(
        f"Protein length mean: "
        f"{mean_length:.2f}"
        if pd.notna(mean_length)
        else "Protein length mean: NA"
    )

    print(
        f"Protein length median: "
        f"{median_length}"
        if pd.notna(median_length)
        else "Protein length median: NA"
    )


    # ========================================================
    # 3. Sequence length mismatch
    # ========================================================

    mismatch_count = 0
    mismatch_unique_ids = 0

    if "sequence_length" in df.columns:

        df["reference_sequence_length"] = (
            pd.to_numeric(
                df["sequence_length"],
                errors="coerce"
            )
        )

        mismatch_mask = (
            df["calculated_protein_length"].notna()
            &
            df["reference_sequence_length"].notna()
            &
            (
                df["calculated_protein_length"]
                !=
                df["reference_sequence_length"]
            )
        )

        mismatch_count = int(
            mismatch_mask.sum()
        )

        mismatch_unique_ids = (
            df.loc[
                mismatch_mask,
                "uniprot_id"
            ]
            .dropna()
            .nunique()
        )


        print(
            f"Sequence length mismatch: "
            f"{mismatch_count:,}"
        )

        print(
            f"Length mismatch unique UniProt ID: "
            f"{mismatch_unique_ids:,}"
        )


        if mismatch_count > 0:

            mismatch_df = df.loc[
                mismatch_mask,
                [
                    "uniprot_id",
                    "_original_uniprot_id",
                    "reference_sequence_length",
                    "calculated_protein_length",
                    "protein_sequence"
                ]
            ].copy()

            mismatch_df.insert(
                0,
                "dataset",
                dataset_name
            )

            mismatch_df.insert(
                1,
                "problem_type",
                "LENGTH_MISMATCH"
            )

            all_mismatch_records.append(
                mismatch_df
            )

            all_problem_records.append(
                mismatch_df
            )

            print()
            print(
                ">>> Length mismatch UniProt IDs"
            )

            print(
                mismatch_df[
                    [
                        "uniprot_id",
                        "reference_sequence_length",
                        "calculated_protein_length"
                    ]
                ]
                .drop_duplicates()
                .to_string(index=False)
            )

    else:

        print(
            "Sequence length mismatch: "
            "sequence_length 컬럼 없음"
        )


    # ========================================================
    # 4. Invalid amino acid
    # ========================================================

    df["invalid_aa"] = (
        df["protein_sequence"]
        .apply(check_invalid_aa)
    )

    invalid_mask = (
        df["invalid_aa"].notna()
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    invalid_unique_ids = (
        df.loc[
            invalid_mask,
            "uniprot_id"
        ]
        .dropna()
        .nunique()
    )


    print(
        f"Invalid amino-acid sequence: "
        f"{invalid_count:,}"
    )

    print(
        f"Invalid sequence unique UniProt ID: "
        f"{invalid_unique_ids:,}"
    )


    if invalid_count > 0:

        invalid_df = df.loc[
            invalid_mask,
            [
                "uniprot_id",
                "_original_uniprot_id",
                "calculated_protein_length",
                "invalid_aa",
                "protein_sequence"
            ]
        ].copy()

        invalid_df.insert(
            0,
            "dataset",
            dataset_name
        )

        invalid_df.insert(
            1,
            "problem_type",
            "INVALID_AMINO_ACID"
        )

        all_invalid_records.append(
            invalid_df
        )

        all_problem_records.append(
            invalid_df
        )

        # ----------------------------------------------------
        # invalid character distribution
        # ----------------------------------------------------

        invalid_char_counts = {}

        for value in invalid_df["invalid_aa"]:

            if pd.isna(value):
                continue

            for char in str(value):

                invalid_char_counts[char] = (
                    invalid_char_counts.get(
                        char,
                        0
                    ) + 1
                )


        print()
        print(
            "Invalid character distribution:"
        )

        for char, count in sorted(
            invalid_char_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(
                f"  '{char}' : "
                f"{count:,}"
            )

        print()
        print(
            ">>> Invalid sequence UniProt IDs"
        )

        print(
            invalid_df[
                [
                    "uniprot_id",
                    "calculated_protein_length",
                    "invalid_aa"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "uniprot_id"
            )
            .to_string(index=False)
        )


    # ========================================================
    # 5. >1024 aa
    # ========================================================

    long_1024_mask = (
        df["calculated_protein_length"].notna()
        &
        (
            df["calculated_protein_length"]
            > USER_LENGTH_LIMIT
        )
    )

    long_1024_rows = int(
        long_1024_mask.sum()
    )

    long_1024_unique_ids = (
        df.loc[
            long_1024_mask,
            "uniprot_id"
        ]
        .dropna()
        .nunique()
    )


    print()
    print(
        f"> {USER_LENGTH_LIMIT} aa:"
    )

    print(
        f"  rows = "
        f"{long_1024_rows:,}"
    )

    print(
        f"  unique ID = "
        f"{long_1024_unique_ids:,}"
    )


    if long_1024_rows > 0:

        long_df = df.loc[
            long_1024_mask,
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "calculated_protein_length",
                "protein_sequence"
            ]
        ].copy()

        long_df.insert(
            0,
            "dataset",
            dataset_name
        )

        long_df.insert(
            1,
            "problem_type",
            "PROTEIN_LENGTH_OVER_1024"
        )

        all_long_1024_records.append(
            long_df
        )

        print()
        print(
            ">>> >1024 aa UniProt IDs"
        )

        print(
            long_df[
                [
                    "uniprot_id",
                    "calculated_protein_length"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "calculated_protein_length",
                ascending=False
            )
            .to_string(index=False)
        )


    # ========================================================
    # 6. >1022 aa
    # ========================================================

    long_1022_mask = (
        df["calculated_protein_length"].notna()
        &
        (
            df["calculated_protein_length"]
            > ESM2_RESIDUE_LIMIT
        )
    )

    long_1022_rows = int(
        long_1022_mask.sum()
    )

    long_1022_unique_ids = (
        df.loc[
            long_1022_mask,
            "uniprot_id"
        ]
        .dropna()
        .nunique()
    )


    print()
    print(
        f"> {ESM2_RESIDUE_LIMIT} aa "
        f"(ESM-2 residue limit):"
    )

    print(
        f"  rows = "
        f"{long_1022_rows:,}"
    )

    print(
        f"  unique ID = "
        f"{long_1022_unique_ids:,}"
    )


    if long_1022_rows > 0:

        long_1022_df = df.loc[
            long_1022_mask,
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "calculated_protein_length",
                "protein_sequence"
            ]
        ].copy()

        long_1022_df.insert(
            0,
            "dataset",
            dataset_name
        )

        long_1022_df.insert(
            1,
            "problem_type",
            "PROTEIN_LENGTH_OVER_1022"
        )

        all_long_1022_records.append(
            long_1022_df
        )


    # ========================================================
    # Summary
    # ========================================================

    summary_records.append({

        "dataset":
            dataset_name,

        "rows":
            len(df),

        "unique_uniprot_id":
            df["uniprot_id"].nunique(),

        "sequence_missing_rows":
            missing_seq_count,

        "sequence_missing_unique_ids":
            missing_unique_ids,

        "sequence_available_rows":
            int((~missing_seq_mask).sum()),

        "length_mismatch_rows":
            mismatch_count,

        "length_mismatch_unique_ids":
            mismatch_unique_ids,

        "invalid_sequence_rows":
            invalid_count,

        "invalid_sequence_unique_ids":
            invalid_unique_ids,

        "min_protein_length":
            min_length,

        "max_protein_length":
            max_length,

        "mean_protein_length":
            (
                round(mean_length, 2)
                if pd.notna(mean_length)
                else pd.NA
            ),

        "median_protein_length":
            median_length,

        "over_1024_rows":
            long_1024_rows,

        "over_1024_unique_ids":
            long_1024_unique_ids,

        "over_1022_rows":
            long_1022_rows,

        "over_1022_unique_ids":
            long_1022_unique_ids
    })


# ============================================================
# Summary 저장
# ============================================================

summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    QC_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Problem Detail 통합
# ============================================================

if all_problem_records:

    problem_detail = pd.concat(
        all_problem_records,
        ignore_index=True
    )

else:

    problem_detail = pd.DataFrame(
        columns=[
            "dataset",
            "problem_type",
            "uniprot_id",
            "_original_uniprot_id"
        ]
    )


# 문제 유형 + ID + dataset 기준 중복 제거
problem_detail = (
    problem_detail
    .drop_duplicates(
        subset=[
            "dataset",
            "problem_type",
            "uniprot_id"
        ]
    )
)


problem_detail.to_csv(
    PROBLEM_DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Missing sequence 저장
# ============================================================

if all_missing_records:

    missing_all = pd.concat(
        all_missing_records,
        ignore_index=True
    )

else:

    missing_all = pd.DataFrame()


missing_all.to_csv(
    MISSING_SEQUENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Invalid sequence 저장
# ============================================================

if all_invalid_records:

    invalid_all = pd.concat(
        all_invalid_records,
        ignore_index=True
    )

else:

    invalid_all = pd.DataFrame()


invalid_all.to_csv(
    INVALID_SEQUENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Length mismatch 저장
# ============================================================

if all_mismatch_records:

    mismatch_all = pd.concat(
        all_mismatch_records,
        ignore_index=True
    )

else:

    mismatch_all = pd.DataFrame()


mismatch_all.to_csv(
    LENGTH_MISMATCH_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# >1024 aa 통합
# ============================================================

if all_long_1024_records:

    long_1024_all = pd.concat(
        all_long_1024_records,
        ignore_index=True
    )

    long_1024_unique = (
        long_1024_all
        .groupby(
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "calculated_protein_length",
                "protein_sequence"
            ],
            dropna=False
        )["dataset"]
        .apply(
            lambda x:
                ",".join(
                    sorted(set(x))
                )
        )
        .reset_index()
    )

    long_1024_unique = (
        long_1024_unique
        .rename(
            columns={
                "calculated_protein_length":
                    "protein_length",
                "dataset":
                    "datasets"
            }
        )
    )

    long_1024_unique = (
        long_1024_unique
        .sort_values(
            "protein_length",
            ascending=False
        )
    )

else:

    long_1024_unique = pd.DataFrame(
        columns=[
            "uniprot_id",
            "protein_name",
            "gene_name",
            "protein_length",
            "protein_sequence",
            "datasets"
        ]
    )


long_1024_unique.to_csv(
    LONG_1024_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# >1022 aa 통합
# ============================================================

if all_long_1022_records:

    long_1022_all = pd.concat(
        all_long_1022_records,
        ignore_index=True
    )

    long_1022_unique = (
        long_1022_all
        .groupby(
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "calculated_protein_length",
                "protein_sequence"
            ],
            dropna=False
        )["dataset"]
        .apply(
            lambda x:
                ",".join(
                    sorted(set(x))
                )
        )
        .reset_index()
    )

    long_1022_unique = (
        long_1022_unique
        .rename(
            columns={
                "calculated_protein_length":
                    "protein_length",
                "dataset":
                    "datasets"
            }
        )
    )

    long_1022_unique = (
        long_1022_unique
        .sort_values(
            "protein_length",
            ascending=False
        )
    )

else:

    long_1022_unique = pd.DataFrame(
        columns=[
            "uniprot_id",
            "protein_name",
            "gene_name",
            "protein_length",
            "protein_sequence",
            "datasets"
        ]
    )


long_1022_unique.to_csv(
    LONG_1022_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# ESM-2 preprocessing 대상
# ============================================================

if len(long_1024_unique) > 0:

    esm2_preprocess = long_1024_unique.copy()

    esm2_preprocess[
        "esm2_preprocessing"
    ] = "REQUIRED"

    esm2_preprocess[
        "reason"
    ] = "protein_length > 1024 aa"

    esm2_preprocess[
        "esm2_residue_limit"
    ] = ESM2_RESIDUE_LIMIT

    esm2_preprocess = (
        esm2_preprocess[
            [
                "uniprot_id",
                "protein_name",
                "gene_name",
                "protein_length",
                "datasets",
                "esm2_residue_limit",
                "esm2_preprocessing",
                "reason",
                "protein_sequence"
            ]
        ]
    )

else:

    esm2_preprocess = pd.DataFrame(
        columns=[
            "uniprot_id",
            "protein_name",
            "gene_name",
            "protein_length",
            "datasets",
            "esm2_residue_limit",
            "esm2_preprocessing",
            "reason",
            "protein_sequence"
        ]
    )


esm2_preprocess.to_csv(
    ESM2_PREPROCESS_ID_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 최종 QC 출력
# ============================================================

print()
print("=" * 100)
print("FINAL QC SUMMARY")
print("=" * 100)

print()

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 문제 파일 / ID 요약
# ============================================================

print()
print("=" * 100)
print("PROBLEM DATASET / UNIPROT SUMMARY")
print("=" * 100)


if len(problem_detail) == 0:

    print(
        "Sequence 관련 문제가 발견되지 않았습니다."
    )

else:

    problem_summary = (
        problem_detail
        .groupby(
            [
                "dataset",
                "problem_type"
            ]
        )
        .agg(
            problem_unique_uniprot_ids=(
                "uniprot_id",
                "nunique"
            )
        )
        .reset_index()
    )

    print(
        problem_summary.to_string(
            index=False
        )
    )


# ============================================================
# Missing sequence 상세
# ============================================================

print()
print("=" * 100)
print("MISSING SEQUENCE")
print("=" * 100)

if len(missing_all) > 0:

    print(
        missing_all[
            [
                "dataset",
                "uniprot_id",
                "_original_uniprot_id"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

else:

    print("없음")


# ============================================================
# Invalid AA 상세
# ============================================================

print()
print("=" * 100)
print("INVALID AMINO ACID")
print("=" * 100)

if len(invalid_all) > 0:

    print(
        invalid_all[
            [
                "dataset",
                "uniprot_id",
                "calculated_protein_length",
                "invalid_aa"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "dataset",
                "uniprot_id"
            ]
        )
        .to_string(index=False)
    )

else:

    print("없음")


# ============================================================
# Length mismatch 상세
# ============================================================

print()
print("=" * 100)
print("LENGTH MISMATCH")
print("=" * 100)

if len(mismatch_all) > 0:

    print(
        mismatch_all[
            [
                "dataset",
                "uniprot_id",
                "reference_sequence_length",
                "calculated_protein_length"
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

else:

    print("없음")


# ============================================================
# >1024 aa 최종 ID
# ============================================================

print()
print("=" * 100)
print("ESM-2 PREPROCESSING TARGET : >1024 aa")
print("=" * 100)

print(
    f"Unique UniProt IDs: "
    f"{len(long_1024_unique):,}"
)

if len(long_1024_unique) > 0:

    print()

    print(
        long_1024_unique[
            [
                "uniprot_id",
                "protein_length",
                "datasets"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# 저장 파일
# ============================================================

print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print(
    f"1. QC Summary:"
    f"\n   {QC_SUMMARY_FILE}"
)

print(
    f"\n2. 전체 문제 상세:"
    f"\n   {PROBLEM_DETAIL_FILE}"
)

print(
    f"\n3. Sequence missing:"
    f"\n   {MISSING_SEQUENCE_FILE}"
)

print(
    f"\n4. Invalid AA:"
    f"\n   {INVALID_SEQUENCE_FILE}"
)

print(
    f"\n5. Length mismatch:"
    f"\n   {LENGTH_MISMATCH_FILE}"
)

print(
    f"\n6. >1024 aa:"
    f"\n   {LONG_1024_FILE}"
)

print(
    f"\n7. >1022 aa:"
    f"\n   {LONG_1022_FILE}"
)

print(
    f"\n8. ESM-2 preprocessing IDs:"
    f"\n   {ESM2_PREPROCESS_ID_FILE}"
)

print()
print("=" * 100)
print("QC COMPLETE")
print("=" * 100)