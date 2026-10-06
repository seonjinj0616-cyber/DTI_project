import os
import duckdb


def _find_project_root(marker_name="DTI_project"):
    """
    이 스크립트 파일 위치에서 위로 올라가며 이름이 marker_name 인 폴더를 찾는다.
    DTI_project 폴더 전체를 다른 컴퓨터/다른 드라이브/다른 사용자 계정으로 옮겨도,
    폴더 이름만 같으면 경로를 자동으로 다시 찾는다.
    """

    path = os.path.dirname(os.path.abspath(__file__))

    while True:

        if os.path.basename(path) == marker_name:
            return path

        parent = os.path.dirname(path)

        if parent == path:  # 드라이브/파일시스템 루트까지 올라갔는데 못 찾음
            raise RuntimeError(
                "'%s' 폴더를 찾을 수 없습니다. 이 스크립트가 %s 폴더 안(하위 폴더 포함)에 있는지 확인하세요."
                % (marker_name, marker_name)
            )

        path = parent


PROJECT_DIR = _find_project_root()
OT_DB_DIR = os.path.join(PROJECT_DIR, "data", "open_target_dataDB")

DEFAULT_PARQUET_DIR = os.path.join(OT_DB_DIR, "opentargets")
DEFAULT_SQLITE_DB_PATH = os.path.join(OT_DB_DIR, "cdss_integrated.db")


def load_open_targets_to_sqlite(
        parquet_base_dir: str = DEFAULT_PARQUET_DIR,
        sqlite_db_path: str = DEFAULT_SQLITE_DB_PATH
):
    if not os.path.exists(sqlite_db_path):
        print(f"❌ Error: '{sqlite_db_path}' 파일이 존재하지 않습니다.")
        return

    print("🚀 DuckDB 세션 생성 및 SQLite 데이터베이스 연결...")
    con = duckdb.connect()

    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_db_path}' AS sqlite_db (TYPE SQLITE);")

    targets_path = os.path.join(parquet_base_dir, "target", "**", "*.parquet")
    moa_path = os.path.join(parquet_base_dir, "drug_mechanism_of_action", "**", "*.parquet")
    evidence_path = os.path.join(parquet_base_dir, "evidence", "**", "*.parquet")
    diseases_path = os.path.join(parquet_base_dir, "disease", "**", "*.parquet")
    openfda_target_path = os.path.join(parquet_base_dir, "openfda_significant_adverse_target_reactions", "**", "*.parquet")

    # ------------------------------------------------------------------

    # Step 1: Pathways 적재
    # ------------------------------------------------------------------
    print(f"\n1/5. Processing Target Pathways ({targets_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_pathway (target_id, gene_symbol, pathway_id, pathway_name)
            WITH new_data AS (
                SELECT DISTINCT
                    id AS target_id,
                    approvedSymbol AS gene_symbol,
                    pw.pathwayId AS pathway_id,
                    pw.pathway AS pathway_name
                FROM read_parquet('{targets_path}', hive_partitioning=false)
                CROSS JOIN UNNEST(pathways) AS t(pw)
                WHERE pathways IS NOT NULL AND pw.pathwayId IS NOT NULL
            )
            SELECT n.* 
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_pathway e 
                   ON n.target_id = e.target_id AND n.pathway_id = e.pathway_id
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_pathway 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_pathway 적재 실패: {e}")

    # ------------------------------------------------------------------
    # Step 1.5: 타깃 안전성(Safety Liabilities) 적재
    #   target parquet 의 safetyLiabilities 필드 (event, eventId, datasource, effects 등)
    #   -> ref_target_phenotypes (target_id, phenotype_id, phenotype_label, evidence_source)
    # ------------------------------------------------------------------
    print(f"\n2/5. Processing Target Safety Liabilities (curated, from target.safetyLiabilities) ({targets_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_phenotypes (target_id, phenotype_id, phenotype_label, evidence_source)
            WITH new_data AS (
                SELECT DISTINCT
                    id AS target_id,
                    sl.eventId AS phenotype_id,
                    sl.event AS phenotype_label,
                    'safetyLiabilities:' || COALESCE(sl.datasource, 'unknown') AS evidence_source
                FROM read_parquet('{targets_path}', hive_partitioning=false)
                CROSS JOIN UNNEST(safetyLiabilities) AS t(sl)
                WHERE safetyLiabilities IS NOT NULL AND sl.event IS NOT NULL
            )
            SELECT n.*
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_phenotypes e
                   ON n.target_id = e.target_id
                  AND COALESCE(n.phenotype_id, '') = COALESCE(e.phenotype_id, '')
                  AND n.phenotype_label = e.phenotype_label
                  AND COALESCE(n.evidence_source, '') = COALESCE(e.evidence_source, '')
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_phenotypes 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_phenotypes 적재 실패: {e}")

    # ------------------------------------------------------------------
    # Step 2: MoA 적재
    # ------------------------------------------------------------------
    print(f"\n3/5. Processing Mechanisms of Action ({moa_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_moa (target_id, chembl_id, action_type, mechanism_of_action)
            WITH new_data AS (
                SELECT DISTINCT
                    t_id AS target_id,
                    chemblIds[1] AS chembl_id,
                    actionType AS action_type,
                    mechanismOfAction AS mechanism_of_action
                FROM read_parquet('{moa_path}', hive_partitioning=false)
                CROSS JOIN UNNEST(targets) AS t(t_id)
                WHERE targets IS NOT NULL AND t_id IS NOT NULL AND chemblIds IS NOT NULL AND len(chemblIds) > 0
            )
            SELECT n.* 
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_moa e 
                   ON n.target_id = e.target_id 
                  AND COALESCE(n.chembl_id, '') = COALESCE(e.chembl_id, '')
                  AND COALESCE(n.mechanism_of_action, '') = COALESCE(e.mechanism_of_action, '')
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_moa 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_moa 적재 실패: {e}")

    # ------------------------------------------------------------------
    # Step 4: 타깃 안전성(openFDA 통계적 신호) 적재
    #   openfda_significant_adverse_target_reactions: 이 타깃을 표적으로 하는 약물들 전체에서,
    #   FDA 부작용 보고(FAERS) 상 우연보다 유의미하게 많이 나타나는 부작용 (llr > critval).
    #   -> ref_target_phenotypes 에 safetyLiabilities 와는 다른 evidence_source 로 함께 저장
    #      (evidence_source 앞부분으로 두 출처를 구분: 'safetyLiabilities:...' vs 'openFDA_significant_adverse_target_reactions')
    # ------------------------------------------------------------------
    print(f"\n4/5. Processing openFDA Significant Adverse Target Reactions ({openfda_target_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_phenotypes (target_id, phenotype_id, phenotype_label, evidence_source)
            WITH new_data AS (
                SELECT DISTINCT
                    targetId AS target_id,
                    meddraCode AS phenotype_id,
                    event AS phenotype_label,
                    'openFDA_significant_adverse_target_reactions' AS evidence_source
                FROM read_parquet('{openfda_target_path}', hive_partitioning=false)
                WHERE event IS NOT NULL AND llr > critval
            )
            SELECT n.*
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_phenotypes e
                   ON n.target_id = e.target_id
                  AND COALESCE(n.phenotype_id, '') = COALESCE(e.phenotype_id, '')
                  AND n.phenotype_label = e.phenotype_label
                  AND COALESCE(n.evidence_source, '') = COALESCE(e.evidence_source, '')
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_phenotypes(openFDA) 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_phenotypes(openFDA) 적재 실패: {e}")

    # ------------------------------------------------------------------
    # Step 5: 타깃-질병 증거의 "종류별" 내역 적재 (같은 evidence 파일 재사용, 새로 받을 필요 없음)
    #   ref_target_disease_evidence 는 여러 증거(유전 연관성/체세포 돌연변이/알려진 약물/문헌 등)를
    #   최댓값 하나로 합쳐서 저장하는데, 이 단계는 종류(datatypeId)별로 나눠서 따로 저장한다.
    #   -> "왜 이 타깃-질병 쌍이 연관 있다고 보는지" 를 더 구체적으로 설명하는 데 쓴다 (step3 에서 사용).
    #   사전에 add_evidence_detail_table.sql 로 ref_target_disease_evidence_detail 테이블을 만들어야 한다.
    # ------------------------------------------------------------------
    print(f"\n5/6. Processing Target-Disease Evidence Detail (by datatype) ({evidence_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_disease_evidence_detail (target_id, disease_id, datatype, max_score)
            WITH new_data AS (
                SELECT
                    targetId AS target_id,
                    diseaseId AS disease_id,
                    datatypeId AS datatype,
                    MAX(score) AS max_score
                FROM read_parquet('{evidence_path}', hive_partitioning=false)
                WHERE datatypeId IS NOT NULL
                GROUP BY targetId, diseaseId, datatypeId
            )
            SELECT n.*
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_disease_evidence_detail e
                   ON n.target_id = e.target_id AND n.disease_id = e.disease_id AND n.datatype = e.datatype
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_disease_evidence_detail 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_disease_evidence_detail 적재 실패: {e}")
        print("     -> add_evidence_detail_table.sql 로 테이블을 먼저 만들었는지 확인하세요.")

    # ------------------------------------------------------------------
    # Step 6: Evidence & Disease 적재 (중복 방지 & GROUP BY 추가)
    # ------------------------------------------------------------------
    print(f"\n6/6. Processing Target-Disease Evidence & Join ({evidence_path})...")
    try:
        con.execute(f"""
            INSERT INTO sqlite_db.ref_target_disease_evidence (target_id, disease_id, disease_name, overall_score)
            WITH filtered_evidence AS (
                -- targetId와 diseaseId 기준으로 최고 점수(MAX)만 추출해 중복 제거
                SELECT 
                    targetId AS target_id,
                    diseaseId AS disease_id,
                    MAX(score) AS overall_score
                FROM read_parquet('{evidence_path}', hive_partitioning=false)
                WHERE score >= 0.4
                GROUP BY targetId, diseaseId
            ),
            disease_meta AS (
                SELECT DISTINCT
                    id AS disease_id,
                    name AS disease_name
                FROM read_parquet('{diseases_path}', hive_partitioning=false)
            ),
            new_data AS (
                SELECT DISTINCT
                    e.target_id,
                    e.disease_id,
                    d.disease_name,
                    e.overall_score
                FROM filtered_evidence e
                JOIN disease_meta d ON e.disease_id = d.disease_id
            )
            -- 기존 DB에 없는 신규 (target_id, disease_id) 조합만 필터링 후 삽입
            SELECT n.* 
            FROM new_data n
            LEFT JOIN sqlite_db.ref_target_disease_evidence e 
                   ON n.target_id = e.target_id AND n.disease_id = e.disease_id
            WHERE e.target_id IS NULL;
        """)
        print("  ✅ ref_target_disease_evidence 적재 완료!")
    except Exception as e:
        print(f"  ❌ ref_target_disease_evidence 적재 실패: {e}")

    con.close()
    print("\n🎉 모든 Open Targets 참조 데이터 적재 작업이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    load_open_targets_to_sqlite()