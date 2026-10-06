import sqlite3
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import Descriptors


# ============================================================
# 1. 파일 경로 설정
# ============================================================

DB_PATH = r"C:\workspace\python\project_personal\DTI_project\data\chembl\chembl_37_sqlite\chembl_37\chembl_37_sqlite\chembl_37.db"
OUTPUT_CSV = "../data/output/chembl_dti_norm.csv"
os.makedirs("../data/output", exist_ok=True)


# ============================================================
# 2. 허용되는 affinity 단위
#
# multiplier:
# 원래 affinity_value × multiplier = nM
# ============================================================

UNIT_MULTIPLIERS = {

    # --------------------------------------------------------
    # 이미 nM
    # --------------------------------------------------------
    'nM': 1.0,

    # --------------------------------------------------------
    # uM
    #
    # 10^2 uM
    # = 100 uM
    # = 100,000 nM
    # --------------------------------------------------------
    '10^2 uM': 1e5,

    # --------------------------------------------------------
    # nM
    #
    # 10^3 nM
    # = 1,000 nM
    # --------------------------------------------------------
    '10^3nM': 1e3,

    # --------------------------------------------------------
    # mol/L
    #
    # 1 mol/L = 1e9 nM
    # --------------------------------------------------------
    '10^-5 mol/L': 1e4,
    '10^-7mol/L': 1e2,
    '10^-8mol/L': 1e1,

    # --------------------------------------------------------
    # microM
    #
    # 1 microM = 1,000 nM
    # --------------------------------------------------------
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
# 4. ug.mL-1 → nM
#
# concentration:
#
#     ug/mL
#
# 을 molecular weight를 이용하여 nM로 변환
#
# nM = (ug/mL × 1e6) / MW
#
# MW는 RDKit에서 SMILES로 계산
# ============================================================

def ug_ml_to_nM(value, smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        mw = Descriptors.MolWt(mol)

        if mw <= 0:
            return np.nan
        return (
            float(value)
            * 1e6
            / mw
        )
    except Exception:
        return np.nan

# ============================================================
# 5. SQL Query
#
# 여기서는 ChEMBL DB에서 필요한 원본 데이터를 가져온다.
#
# data_validity_comment
# potential_duplicate
# standard_flag
# activity_comment
#
# 등을 반드시 같이 가져온다.
# ============================================================

QUERY = """
WITH single_component_targets AS (
    SELECT tid
    FROM target_components
    GROUP BY tid
    HAVING COUNT(DISTINCT component_id) = 1
)

SELECT
    'ChEMBL' AS source,
    CAST(a.activity_id AS TEXT) AS source_record_id,
    CAST(a.activity_id AS TEXT) AS activity_id,
    md.chembl_id AS chembl_id,
    md.pref_name AS compound_name,
    NULL AS bindingdb_id,
    cs_struct.canonical_smiles AS smiles,
    td.chembl_id AS target_chembl_id,
    cs.accession AS uniprot_id,
    td.pref_name AS target_name,
    td.target_type AS target_type,
    td.organism AS organism,
    ass.chembl_id AS assay_id,
    ass.assay_type AS assay_type,
    a.standard_type AS affinity_type,
    COALESCE(a.standard_relation, '=') AS affinity_relation,
    a.standard_value AS affinity_value,
    a.standard_units AS affinity_unit,
    a.pchembl_value AS pchembl_value,
    a.data_validity_comment AS data_validity_comment,
    a.potential_duplicate AS potential_duplicate,
    a.standard_flag AS standard_flag,
    a.activity_comment AS activity_comment,
    a.action_type AS action_type,
    CAST(d.pubmed_id AS TEXT) AS publication_id,
    a.doc_id AS document_id,
    a.assay_id AS source_assay_id,
    a.molregno AS molregno
FROM activities a

JOIN assays ass
    ON a.assay_id = ass.assay_id

JOIN target_dictionary td
    ON ass.tid = td.tid

JOIN single_component_targets sct
    ON td.tid = sct.tid

JOIN target_components tc
    ON td.tid = tc.tid

JOIN component_sequences cs
    ON tc.component_id = cs.component_id

JOIN molecule_dictionary md
    ON a.molregno = md.molregno

JOIN compound_structures cs_struct
    ON md.molregno = cs_struct.molregno

LEFT JOIN docs d
    ON a.doc_id = d.doc_id

WHERE td.target_type = 'SINGLE PROTEIN'

  AND td.organism = 'Homo sapiens'

  AND ass.assay_type = 'B'

  AND a.standard_type IN (
      'Ki',
      'IC50',
      'Kd'
  )

  AND cs_struct.canonical_smiles IS NOT NULL

  AND a.standard_value IS NOT NULL

  AND a.standard_value > 0
"""


# ============================================================
# 6. 초기화
# ============================================================

chunk_size = 200000

first_chunk = True

total_before = 0
total_after_validity = 0
total_after_unit = 0

unit_counts_before = {}
unit_counts_after = {}

conversion_failed = 0


if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)


print("=" * 70)
print("ChEMBL 37 → chembl_dti.csv parsing 시작")
print("=" * 70)

print(f"DB     : {DB_PATH}")
print(f"OUTPUT : {OUTPUT_CSV}")
print()


# ============================================================
# 7. SQLite 조회 + Chunk processing
# ============================================================

with sqlite3.connect(DB_PATH) as conn:

    print("SQLite DB 연결 완료")
    print("SQL Query 실행 중...")
    print()


    for chunk_idx, chunk in enumerate(
        pd.read_sql_query(
            QUERY,
            conn,
            chunksize=chunk_size
        )
    ):

        total_before += len(chunk)

        print(
            f"[Chunk {chunk_idx}] "
            f"입력: {len(chunk):,}"
        )


        # ====================================================
        # 7-1. SMILES / UniProt 확인
        # ====================================================

        chunk = chunk.dropna(
            subset=[
                'smiles',
                'uniprot_id'
            ]
        )

        if len(chunk) == 0:
            continue


        # ====================================================
        # 7-2. data_validity_comment 처리
        #
        # KEEP:
        #   NULL
        #   Manually validated
        #
        # REMOVE:
        #   Outside typical range
        #   Potential transcription error
        #   Potential missing data
        #   Potential author error
        #   Author confirmed error
        #   기타 모든 non-NULL
        # ====================================================

        validity_mask = (
            chunk[
                'data_validity_comment'
            ].isna()
            |
            (
                chunk[
                    'data_validity_comment'
                ]
                == 'Manually validated'
            )
        )

        removed_validity = (
            ~validity_mask
        ).sum()

        chunk = chunk[
            validity_mask
        ].copy()

        total_after_validity += len(chunk)

        if removed_validity > 0:

            print(
                f"  validity 제거: "
                f"{removed_validity:,}"
            )

        if len(chunk) == 0:
            continue


        # ====================================================
        # 7-3. Relation normalization
        # ====================================================

        chunk[
            'affinity_relation_normalized'
        ] = (
            chunk[
                'affinity_relation'
            ]
            .map(RELATION_MAP)
        )


        # 매핑 불가능한 relation 제거

        chunk = chunk[
            chunk[
                'affinity_relation_normalized'
            ].notna()
        ].copy()

        if len(chunk) == 0:
            continue


        # ====================================================
        # 7-4. 현재 chunk의 원본 unit 분포 기록
        # ====================================================

        current_unit_counts = (
            chunk[
                'affinity_unit'
            ]
            .value_counts(dropna=False)
        )

        for unit, count in current_unit_counts.items():

            unit_counts_before[unit] = (
                unit_counts_before.get(unit, 0)
                + int(count)
            )


        # ====================================================
        # 7-5. affinity_value_nM 초기화
        # ====================================================

        chunk[
            'affinity_value_nM'
        ] = np.nan

        chunk[
            'unit_conversion_status'
        ] = 'UNSUPPORTED_UNIT'


        # ====================================================
        # 7-6. 고정 multiplier를 이용한 단위 변환
        # ====================================================

        for unit, multiplier in UNIT_MULTIPLIERS.items():

            mask = (
                chunk[
                    'affinity_unit'
                ]
                == unit
            )

            if not mask.any():
                continue

            chunk.loc[
                mask,
                'affinity_value_nM'
            ] = (
                chunk.loc[
                    mask,
                    'affinity_value'
                ]
                * multiplier
            )

            chunk.loc[
                mask,
                'unit_conversion_status'
            ] = 'CONVERTED'


        # ====================================================
        # 7-7. ug.mL-1
        #
        # RDKit MW 기반 변환
        # ====================================================

        ug_ml_mask = (
            chunk[
                'affinity_unit'
            ]
            == 'ug.mL-1'
        )

        if ug_ml_mask.any():

            print(
                f"  ug.mL-1 변환: "
                f"{ug_ml_mask.sum():,}"
            )

            converted_values = chunk.loc[
                ug_ml_mask
            ].apply(
                lambda row:
                    ug_ml_to_nM(
                        row[
                            'affinity_value'
                        ],
                        row[
                            'smiles'
                        ]
                    ),
                axis=1
            )

            chunk.loc[
                ug_ml_mask,
                'affinity_value_nM'
            ] = converted_values

            success_mask = (
                ug_ml_mask
                &
                chunk[
                    'affinity_value_nM'
                ].notna()
            )

            failed_mask = (
                ug_ml_mask
                &
                chunk[
                    'affinity_value_nM'
                ].isna()
            )

            chunk.loc[
                success_mask,
                'unit_conversion_status'
            ] = (
                'CONVERTED_USING_RDKIT_MW'
            )

            chunk.loc[
                failed_mask,
                'unit_conversion_status'
            ] = (
                'MW_CONVERSION_FAILED'
            )

            conversion_failed += (
                failed_mask.sum()
            )


        # ====================================================
        # 7-8. 변환 가능한 데이터만 유지
        #
        # /uM
        # /s
        # %
        # min-1
        # 등은 여기서 자동 제거
        # ====================================================

        chunk = chunk[
            chunk[
                'affinity_value_nM'
            ].notna()
        ].copy()


        if len(chunk) == 0:
            continue


        total_after_unit += len(chunk)


        # ====================================================
        # 7-9. 최종 unit 분포
        # ====================================================

        current_after_counts = (
            chunk[
                'affinity_unit'
            ]
            .value_counts(dropna=False)
        )

        for unit, count in current_after_counts.items():

            unit_counts_after[unit] = (
                unit_counts_after.get(unit, 0)
                + int(count)
            )


        # ====================================================
        # 7-10. pAffinity 계산
        #
        # affinity_value_nM → mol/L → -log10
        #
        # pAffinity = -log10(nM × 1e-9)
        # ====================================================

        chunk[
            'paffinity'
        ] = (
            -np.log10(
                chunk[
                    'affinity_value_nM'
                ]
                * 1e-9
            )
        )


        # ====================================================
        # 7-11. 최종 스키마
        # ====================================================

        result_chunk = chunk[[
            'source',
            'source_record_id',
            'activity_id',

            'chembl_id',
            'compound_name',

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

            # 원본 relation
            'affinity_relation',

            # 정규화 relation
            'affinity_relation_normalized',

            # 원본 affinity
            'affinity_value',
            'affinity_unit',

            # nM 정규화 affinity
            'affinity_value_nM',

            # ChEMBL 원본 pChEMBL
            'pchembl_value',

            # 우리가 계산한 pAffinity
            'paffinity',

            # 단위 처리 결과
            'unit_conversion_status',

            # ChEMBL QC metadata
            'data_validity_comment',
            'potential_duplicate',
            'standard_flag',
            'activity_comment',

            'action_type',

            'publication_id',
            'document_id',
            'source_assay_id',
            'molregno'
        ]]


        # ====================================================
        # 7-12. CSV Append
        # ====================================================

        mode = (
            'w'
            if first_chunk
            else 'a'
        )

        result_chunk.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding='utf-8-sig',
            mode=mode,
            header=first_chunk
        )

        first_chunk = False


        print(
            f"  최종 저장: "
            f"{len(result_chunk):,}"
        )


# ============================================================
# 8. 최종 결과 출력
# ============================================================

print()
print("=" * 70)
print("ChEMBL parsing 완료")
print("=" * 70)

print(
    f"SQL 추출 rows           : "
    f"{total_before:,}"
)

print(
    f"Validity filter 후      : "
    f"{total_after_validity:,}"
)

print(
    f"Unit normalization 후   : "
    f"{total_after_unit:,}"
)

print(
    f"MW 변환 실패            : "
    f"{conversion_failed:,}"
)

print()
print("원본 unit 분포")
print("-" * 50)

for unit, count in sorted(
    unit_counts_before.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{str(unit):25s} "
        f"{count:>12,}"
    )


print()
print("최종 유지된 unit 분포")
print("-" * 50)

for unit, count in sorted(
    unit_counts_after.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{str(unit):25s} "
        f"{count:>12,}"
    )


print()
print(
    f"최종 CSV: {OUTPUT_CSV}"
)