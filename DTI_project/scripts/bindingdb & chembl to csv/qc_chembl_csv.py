import os
import numpy as np
import pandas as pd

# ============================================================
# 0. PATH
# ============================================================

INPUT_CSV = "../data/output/chembl_dti.csv"


# ============================================================
# 1. 기존 CSV에서 반드시 존재해야 하는 컬럼
# ============================================================

EXPECTED_OLD_COLUMNS = [
    'source',
    'source_record_id',
    'chembl_id',
    'bindingdb_id',
    'smiles',
    'target_chembl_id',
    'uniprot_id',
    'target_name',
    'target_type',
    'organism',
    'assay_id',
    'assay_type',
    'affinity_type',
    'affinity_relation',
    'affinity_value',
    'affinity_unit',
    'pchembl_value',
    'publication_id'
]


# ============================================================
# 2. 정상적으로 변환을 허용한 단위
# ============================================================

UNIT_MULTIPLIERS = {
    'nM': 1.0,

    # uM → nM
    '10^2 uM': 1e5,

    # 10^3 nM
    '10^3nM': 1e3,

    # mol/L → nM
    '10^-5 mol/L': 1e4,
    '10^-7mol/L': 1e2,
    '10^-8mol/L': 1e1,

    # microM → nM
    '10^-4microM': 1e-1,
    '10^-2microM': 1e1,
}


# ============================================================
# 3. Relation normalization
# ============================================================

RELATION_MAP = {
    '=': '=',
    '>': '>',
    '<': '<',
    '<=': '<',
    '>=': '>',
    '~': '=',
    '>>': '>'
}


# ============================================================
# 4. 파일 존재 확인
# ============================================================

if not os.path.exists(INPUT_CSV):
    print(f"[ERROR] 파일을 찾을 수 없습니다: {INPUT_CSV}")
    raise SystemExit(1)


print("=" * 80)
print("ChEMBL NORMALIZED CSV QC START")
print("=" * 80)
print(f"파일: {INPUT_CSV}\n")


# ============================================================
# 5. CSV LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV, low_memory=False)

print("[1] FILE INFO")
print("-" * 80)
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print()


# ============================================================
# 6. 기존 18개 컬럼 포함 여부
# ============================================================

print("[2] OLD COLUMN COMPATIBILITY QC")
print("-" * 80)

current_columns = list(df.columns)

missing_columns = [
    col for col in EXPECTED_OLD_COLUMNS
    if col not in current_columns
]

existing_old_columns = [
    col for col in EXPECTED_OLD_COLUMNS
    if col in current_columns
]

new_columns = [
    col for col in current_columns
    if col not in EXPECTED_OLD_COLUMNS
]


print(f"기존 컬럼 수           : {len(EXPECTED_OLD_COLUMNS)}")
print(f"새 CSV에서 발견된 수    : {len(existing_old_columns)}")

if missing_columns:
    print("\n[FAIL] 기존 컬럼 중 누락된 컬럼:")
    for col in missing_columns:
        print(f"  - {col}")
else:
    print("\n[PASS] 기존 18개 컬럼이 모두 존재합니다.")

print("\n새로 추가된 컬럼:")
if new_columns:
    for col in new_columns:
        print(f"  + {col}")
else:
    print("  없음")

print()


# ============================================================
# 7. 컬럼 순서 확인
# ============================================================

print("[3] OLD COLUMN ORDER QC")
print("-" * 80)

old_columns_in_current = [
    col for col in current_columns
    if col in EXPECTED_OLD_COLUMNS
]

if old_columns_in_current == EXPECTED_OLD_COLUMNS:
    print("[PASS] 기존 18개 컬럼의 순서도 동일합니다.")
else:
    print("[CHECK] 기존 컬럼은 모두 존재하지만 순서가 달라졌습니다.")
    print("\n기존 순서:")
    for i, col in enumerate(EXPECTED_OLD_COLUMNS, 1):
        print(f"{i:2d}. {col}")

    print("\n새 CSV 순서:")
    for i, col in enumerate(old_columns_in_current, 1):
        print(f"{i:2d}. {col}")

print()


# ============================================================
# 8. source / affinity_type QC
# ============================================================

print("[4] BASIC DATA QC")
print("-" * 80)

print("source:")
print(df['source'].value_counts(dropna=False))

print("\naffinity_type:")
print(df['affinity_type'].value_counts(dropna=False))

print("\naffinity_unit:")
print(df['affinity_unit'].value_counts(dropna=False).head(30))

print()


# ============================================================
# 9. 허용되지 않은 단위가 남아있는지 확인
# ============================================================

print("[5] UNIT QC")
print("-" * 80)

allowed_units = set(UNIT_MULTIPLIERS.keys()) | {'ug.mL-1'}

actual_units = set(df['affinity_unit'].dropna().astype(str).unique())

unsupported_units = sorted(actual_units - allowed_units)

print(f"전체 실제 unit 종류      : {len(actual_units)}")
print(f"허용 unit 종류           : {len(allowed_units)}")
print(f"지원하지 않는 unit 종류  : {len(unsupported_units)}")

if unsupported_units:
    print("\n[FAIL] 허용하지 않은 unit이 남아있습니다:")
    for unit in unsupported_units:
        count = (df['affinity_unit'] == unit).sum()
        print(f"  - {unit}: {count:,}")
else:
    print("[PASS] 지원하지 않는 unit이 남아있지 않습니다.")

print()


# ============================================================
# 10. /uM 삭제 확인
# ============================================================

print("[6] /uM REMOVAL QC")
print("-" * 80)

slash_um_count = (
    df['affinity_unit']
    .astype(str)
    .str.strip()
    .eq('/uM')
    .sum()
)

if slash_um_count == 0:
    print("[PASS] /uM 데이터가 모두 제거되었습니다.")
else:
    print(f"[FAIL] /uM 데이터가 {slash_um_count:,}건 남아있습니다.")

print()


# ============================================================
# 11. rate / activity unit 제거 확인
# ============================================================

print("[7] NON-CONCENTRATION UNIT QC")
print("-" * 80)

RATE_LIKE_UNITS = {
    '/s',
    '10\'-3/s',
    '10\'-1/s',
    '10\'-2/s',
    'mM/min',
    '%',
    '10^-1/s',
    '10\'3/M/min',
    '/min',
    's-1',
    'min-1',
    '10\'-4/s',
    '1/s',
    'um2/s',
    '10^-3/s',
    '10^-2/s',
    'uM tube-1',
    '1/ks',
    'ug nM-1'
}

remaining_rate_units = sorted(
    actual_units.intersection(RATE_LIKE_UNITS)
)

if remaining_rate_units:
    print("[FAIL] 농도/결합 affinity가 아닌 unit이 남아있습니다:")
    for unit in remaining_rate_units:
        count = (df['affinity_unit'] == unit).sum()
        print(f"  - {unit}: {count:,}")
else:
    print("[PASS] 명시적으로 제외한 비농도/rate 계열 unit이 남아있지 않습니다.")

print()


# ============================================================
# 12. data_validity_comment QC
# ============================================================

print("[8] DATA VALIDITY QC")
print("-" * 80)

if 'data_validity_comment' in df.columns:

    validity_counts = df['data_validity_comment'].value_counts(dropna=False)

    print(validity_counts)

    invalid_flags = df[
        df['data_validity_comment'].notna()
        & (df['data_validity_comment'] != 'Manually validated')
    ]

    if len(invalid_flags) == 0:
        print("\n[PASS] NULL 또는 Manually validated만 존재합니다.")
    else:
        print(
            f"\n[FAIL] 허용하지 않은 data_validity_comment "
            f"{len(invalid_flags):,}건 존재"
        )

else:
    print("[WARNING] data_validity_comment 컬럼이 없습니다.")

print()


# ============================================================
# 13. affinity_value 기본 QC
# ============================================================

print("[9] AFFINITY VALUE QC")
print("-" * 80)

bad_affinity = df[
    df['affinity_value'].isna()
    | ~np.isfinite(pd.to_numeric(df['affinity_value'], errors='coerce'))
    | (pd.to_numeric(df['affinity_value'], errors='coerce') <= 0)
]

print(f"잘못된 affinity_value : {len(bad_affinity):,}")

if len(bad_affinity) == 0:
    print("[PASS] affinity_value는 모두 양수이고 유효합니다.")
else:
    print("[FAIL] 잘못된 affinity_value가 존재합니다.")

print()


# ============================================================
# 14. affinity_value_nM 존재 여부
# ============================================================

print("[10] NORMALIZED nM QC")
print("-" * 80)

if 'affinity_value_nM' not in df.columns:
    print("[FAIL] affinity_value_nM 컬럼이 없습니다.")
else:

    nM = pd.to_numeric(df['affinity_value_nM'], errors='coerce')

    print(f"NULL nM                 : {nM.isna().sum():,}")
    print(f"<= 0 nM                 : {(nM <= 0).sum():,}")
    print(f"Infinity                 : {np.isinf(nM).sum():,}")

    bad_nM = (
        nM.isna()
        | ~np.isfinite(nM)
        | (nM <= 0)
    )

    if bad_nM.sum() == 0:
        print("[PASS] 모든 affinity_value_nM이 유효합니다.")
    else:
        print(f"[FAIL] 잘못된 affinity_value_nM: {bad_nM.sum():,}")

print()


# ============================================================
# 15. 고정 단위 변환 검증
# ============================================================

print("[11] FIXED UNIT CONVERSION QC")
print("-" * 80)

if 'affinity_value_nM' in df.columns:

    check_rows = []

    for unit, multiplier in UNIT_MULTIPLIERS.items():

        mask = df['affinity_unit'].astype(str).eq(unit)

        if mask.sum() == 0:
            continue

        original = pd.to_numeric(
            df.loc[mask, 'affinity_value'],
            errors='coerce'
        )

        converted = pd.to_numeric(
            df.loc[mask, 'affinity_value_nM'],
            errors='coerce'
        )

        expected = original * multiplier

        diff = (converted - expected).abs()

        # floating point 오차 고려
        tolerance = np.maximum(
            expected.abs() * 1e-8,
            1e-12
        )

        fail = diff > tolerance

        print(
            f"{unit:20s} | "
            f"rows={mask.sum():6,} | "
            f"expected multiplier={multiplier:g} | "
            f"FAIL={fail.sum():,}"
        )

        if fail.sum() > 0:
            check_rows.append(
                pd.DataFrame({
                    'unit': unit,
                    'original': original[fail],
                    'expected_nM': expected[fail],
                    'actual_nM': converted[fail],
                    'difference': diff[fail]
                })
            )

    if check_rows:
        print("\n[FAIL] 단위 변환 오류 발견")
        conversion_errors = pd.concat(check_rows)
    else:
        print("\n[PASS] 고정 단위 변환이 모두 정확합니다.")

print()


# ============================================================
# 16. ug.mL-1 RDKit 변환 QC
# ============================================================

print("[12] ug.mL-1 RDKit CONVERSION QC")
print("-" * 80)

if 'unit_conversion_status' in df.columns:

    ug_mask = df['affinity_unit'].astype(str).eq('ug.mL-1')

    print(f"ug.mL-1 rows: {ug_mask.sum():,}")

    if ug_mask.sum() > 0:

        print("\nunit_conversion_status:")
        print(
            df.loc[
                ug_mask,
                'unit_conversion_status'
            ].value_counts(dropna=False)
        )

        if 'molecular_weight' in df.columns:

            mw = pd.to_numeric(
                df.loc[ug_mask, 'molecular_weight'],
                errors='coerce'
            )

            value = pd.to_numeric(
                df.loc[ug_mask, 'affinity_value'],
                errors='coerce'
            )

            converted = pd.to_numeric(
                df.loc[ug_mask, 'affinity_value_nM'],
                errors='coerce'
            )

            expected = value * 1e6 / mw

            valid = (
                mw.notna()
                & np.isfinite(mw)
                & (mw > 0)
                & value.notna()
                & converted.notna()
            )

            diff = (
                converted[valid] - expected[valid]
            ).abs()

            tolerance = np.maximum(
                expected[valid].abs() * 1e-8,
                1e-12
            )

            fail = diff > tolerance

            print(f"\n검증 가능 row : {valid.sum():,}")
            print(f"변환 오류 row : {fail.sum():,}")

            if fail.sum() == 0:
                print("[PASS] ug.mL-1 → nM 변환이 정확합니다.")
            else:
                print("[FAIL] ug.mL-1 변환 오류가 존재합니다.")

        else:
            print(
                "[WARNING] molecular_weight 컬럼이 없어 "
                "RDKit 변환을 직접 검증할 수 없습니다."
            )

print()


# ============================================================
# 17. paffinity 공식 검증
# ============================================================

print("[13] PAFFINITY FORMULA QC")
print("-" * 80)

if 'paffinity' in df.columns:

    nM = pd.to_numeric(
        df['affinity_value_nM'],
        errors='coerce'
    )

    paffinity = pd.to_numeric(
        df['paffinity'],
        errors='coerce'
    )

    valid = (
        nM.notna()
        & np.isfinite(nM)
        & (nM > 0)
        & paffinity.notna()
    )

    expected = -np.log10(nM[valid] * 1e-9)

    diff = (
        paffinity[valid] - expected
    ).abs()

    print(f"검증 대상 : {valid.sum():,}")
    print(f"최대 차이 : {diff.max() if len(diff) else np.nan}")
    print(f"평균 차이 : {diff.mean() if len(diff) else np.nan}")

    fail = diff > 1e-8

    print(f"FAIL      : {fail.sum():,}")

    if fail.sum() == 0:
        print("[PASS] paffinity 계산이 정확합니다.")
    else:
        print("[FAIL] paffinity 계산 오류가 존재합니다.")

print()


# ============================================================
# 18. pChEMBL과 paffinity 비교
# ============================================================

print("[14] ORIGINAL pChEMBL vs PAFFINITY QC")
print("-" * 80)

if (
    'pchembl_value' in df.columns
    and 'paffinity' in df.columns
):

    original_pchembl = pd.to_numeric(
        df['pchembl_value'],
        errors='coerce'
    )

    paffinity = pd.to_numeric(
        df['paffinity'],
        errors='coerce'
    )

    valid = (
        original_pchembl.notna()
        & paffinity.notna()
    )

    diff = (
        original_pchembl[valid]
        - paffinity[valid]
    ).abs()

    print(f"비교 가능한 row : {valid.sum():,}")
    print(f"평균 |difference|: {diff.mean():.6f}")
    print(f"median           : {diff.median():.6f}")
    print(f"max              : {diff.max():.6f}")

    print(
        "\n주의: pChEMBL과 paffinity가 반드시 같아야 하는 것은 아닙니다."
    )
    print(
        "pchembl_value = ChEMBL 원본 값"
    )
    print(
        "paffinity     = 정규화된 affinity_value_nM에서 새로 계산한 값"
    )

print()


# ============================================================
# 19. Relation normalization QC
# ============================================================

print("[15] RELATION NORMALIZATION QC")
print("-" * 80)

if 'affinity_relation_normalized' in df.columns:

    original_relations = set(
        df['affinity_relation']
        .dropna()
        .astype(str)
        .unique()
    )

    normalized_relations = set(
        df['affinity_relation_normalized']
        .dropna()
        .astype(str)
        .unique()
    )

    print("Original relation:")
    print(
        df['affinity_relation']
        .value_counts(dropna=False)
    )

    print("\nNormalized relation:")
    print(
        df['affinity_relation_normalized']
        .value_counts(dropna=False)
    )

    expected_relations = set(RELATION_MAP.values())

    unexpected = normalized_relations - expected_relations

    if unexpected:
        print("\n[FAIL] 예상하지 않은 normalized relation:")
        print(unexpected)
    else:
        print(
            "\n[PASS] normalized relation이 "
            "허용된 값만 사용합니다."
        )

print()


# ============================================================
# 20. 핵심 ID NULL QC
# ============================================================

print("[16] KEY COLUMN NULL QC")
print("-" * 80)

KEY_COLUMNS = [
    'source_record_id',
    'chembl_id',
    'smiles',
    'uniprot_id',
    'target_chembl_id',
    'affinity_type',
    'affinity_value',
    'affinity_unit'
]

for col in KEY_COLUMNS:

    if col not in df.columns:
        print(f"{col:30s}: COLUMN MISSING")
        continue

    null_count = df[col].isna().sum()

    print(
        f"{col:30s}: "
        f"NULL={null_count:,}"
    )

print()


# ============================================================
# 21. 중복 QC
# ============================================================

print("[17] DUPLICATE QC")
print("-" * 80)

full_duplicates = df.duplicated().sum()

print(f"Full-row duplicates : {full_duplicates:,}")

if full_duplicates == 0:
    print("[PASS] 완전 동일 row 중복 없음")
else:
    print("[CHECK] 완전 동일 row 중복 존재")


pair_duplicates = (
    df.groupby(
        ['chembl_id', 'uniprot_id'],
        dropna=False
    )
    .size()
)

multi_measurement_pairs = (
    pair_duplicates > 1
).sum()

print(
    f"Drug-Target pair 수              : "
    f"{len(pair_duplicates):,}"
)

print(
    f"2개 이상 measurement pair        : "
    f"{multi_measurement_pairs:,}"
)

print()


# ============================================================
# 22. Extreme paffinity QC
# ============================================================

print("[18] EXTREME PAFFINITY QC")
print("-" * 80)

if 'paffinity' in df.columns:

    paffinity = pd.to_numeric(
        df['paffinity'],
        errors='coerce'
    ).dropna()

    print("pAffinity statistics:")
    print(paffinity.describe())

    print("\nLowest 20:")
    print(
        df.loc[
            df['paffinity']
            .nsmallest(20)
            .index,
            [
                'chembl_id',
                'uniprot_id',
                'affinity_type',
                'affinity_relation',
                'affinity_value',
                'affinity_unit',
                'affinity_value_nM',
                'paffinity'
            ]
        ].to_string(index=False)
    )

    print("\nHighest 20:")
    print(
        df.loc[
            df['paffinity']
            .nlargest(20)
            .index,
            [
                'chembl_id',
                'uniprot_id',
                'affinity_type',
                'affinity_relation',
                'affinity_value',
                'affinity_unit',
                'affinity_value_nM',
                'paffinity'
            ]
        ].to_string(index=False)
    )

print()


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("=" * 80)
print("FINAL QC SUMMARY")
print("=" * 80)

checks = {}

checks["old_18_columns_all_present"] = (
    len(missing_columns) == 0
)

checks["no_unsupported_units"] = (
    len(unsupported_units) == 0
)

checks["no_slash_uM"] = (
    slash_um_count == 0
)

checks["valid_affinity_value"] = (
    len(bad_affinity) == 0
)

if 'affinity_value_nM' in df.columns:
    checks["valid_affinity_value_nM"] = (
        bad_nM.sum() == 0
    )

if 'paffinity' in df.columns:
    checks["paffinity_formula"] = (
        fail.sum() == 0
    )

if 'data_validity_comment' in df.columns:
    checks["validity_filter"] = (
        len(invalid_flags) == 0
    )

for name, result in checks.items():
    print(
        f"{'PASS' if result else 'FAIL':6s} | {name}"
    )

print("=" * 80)
print("QC COMPLETE")
print("=" * 80)