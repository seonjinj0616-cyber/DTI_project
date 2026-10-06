-- Foreign Key 제약 조건 활성화
PRAGMA foreign_keys = ON;

-- ====================================================================
-- PART 1: 정적 참조 데이터베이스 (Open Targets 기반 Pre-built Tables)
-- ====================================================================

-- 1.1 타깃 - Reactome Pathway 매핑 테이블 (Step 2 참조)
CREATE TABLE IF NOT EXISTS ref_target_pathway (
    target_id TEXT NOT NULL,          -- Ensembl Gene ID (ENSG...)
    gene_symbol TEXT,                 -- 유전자 기호 (예: EGFR)
    pathway_id TEXT NOT NULL,         -- Reactome Pathway ID (R-HSA-...)
    pathway_name TEXT NOT NULL,       -- Pathway 명칭
    PRIMARY KEY (target_id, pathway_id)
);
CREATE INDEX IF NOT EXISTS idx_ref_target_pathway_target ON ref_target_pathway(target_id);

-- 1.2 타깃 - MoA (약리 작용기전) 매핑 테이블 (Step 2 참조)
CREATE TABLE IF NOT EXISTS ref_target_moa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id TEXT NOT NULL,          -- Ensembl Gene ID (ENSG...)
    chembl_id TEXT,                   -- 관련 약물 ChEMBL ID
    action_type TEXT NOT NULL,        -- INHIBITOR, AGONIpipST, ANTAGONIST 등
    mechanism_of_action TEXT          -- 세부 작용기전 설명
);
CREATE INDEX IF NOT EXISTS idx_ref_target_moa_target ON ref_target_moa(target_id);

-- 1.3 타깃 - 질병 검증 증거 점수 테이블 (Step 3 참조)
CREATE TABLE IF NOT EXISTS ref_target_disease_evidence (
    target_id TEXT NOT NULL,          -- Ensembl Gene ID (ENSG...)
    disease_id TEXT NOT NULL,         -- EFO / MONDO ID
    disease_name TEXT NOT NULL,       -- 질병명
    overall_score REAL NOT NULL,      -- Open Targets 증거 점수 (>= 0.4)
    PRIMARY KEY (target_id, disease_id)
);
CREATE INDEX IF NOT EXISTS idx_ref_target_evidence_target ON ref_target_disease_evidence(target_id);
CREATE INDEX IF NOT EXISTS idx_ref_target_evidence_score ON ref_target_disease_evidence(overall_score DESC);

-- 1.4 타깃 - 표현형 및 예상 부작용 테이블 (Step 3 참조)
CREATE TABLE IF NOT EXISTS ref_target_phenotypes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id TEXT NOT NULL,          -- Ensembl Gene ID (ENSG...)
    phenotype_id TEXT,                -- HPO ID 등 표현형 코드
    phenotype_label TEXT NOT NULL,    -- 부작용/표현형 명칭 (예: Rash, Cardiac toxicity)
    evidence_source TEXT              -- 출처 (Open Targets / Literature)
);
CREATE INDEX IF NOT EXISTS idx_ref_phenotypes_target ON ref_target_phenotypes(target_id);


-- ====================================================================
-- PART 2: 동적 파이프라인 분석 결과 테이블 (CDSS Inference Results)
-- ====================================================================

-- 2.1 CDSS 세션/분석 요청 이력 관리 테이블
CREATE TABLE IF NOT EXISTS analysis_history (
    analysis_id TEXT PRIMARY KEY,     -- UUID 기반 분석 식별자
    input_smiles TEXT NOT NULL,       -- 사용자 입력 분자 SMILES
    compound_name TEXT,               -- 신물질 또는 약물 이름 (옵션)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2.2 Step 1: ChemBERTa 타깃 단백질 예측 결과 테이블
CREATE TABLE IF NOT EXISTS step1_target_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,        -- analysis_history 연동 FK
    target_id TEXT NOT NULL,          -- 예측된 Ensembl Gene ID
    gene_symbol TEXT,                 -- 유전자 기호
    binding_probability REAL NOT NULL,-- 결합 예측 확률/스코어 (0.0 ~ 1.0)
    rank_order INTEGER NOT NULL,      -- 상위 순위 (1, 2, 3...)
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_step1_analysis ON step1_target_predictions(analysis_id);

-- 2.3 Step 3: 적응증 및 부작용 탐색 결과 테이블 (DB + TxGNN 융합)
CREATE TABLE IF NOT EXISTS step3_indication_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,        -- analysis_history 연동 FK
    target_id TEXT NOT NULL,          -- 타깃 ID
    disease_id TEXT NOT NULL,         -- 질병 ID
    disease_name TEXT NOT NULL,       -- 질병명
    result_type TEXT NOT NULL,        -- 'EVIDENCE_DB' (기존) 또는 'TXGNN_AI' (AI추론)
    confidence_score REAL NOT NULL,   -- Evidence Score 또는 TxGNN Probability
    is_side_effect BOOLEAN DEFAULT 0, -- 0: 적응증(Indication), 1: 부작용(Side Effect)
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_step3_analysis ON step3_indication_results(analysis_id);

-- 2.4 Step 4: SynergyX 멀티모달 병용 약물 추천 결과 테이블
CREATE TABLE IF NOT EXISTS step4_synergy_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id TEXT NOT NULL,        -- analysis_history 연동 FK
    candidate_chembl_id TEXT NOT NULL,-- 병용 추천 약물 B ChEMBL ID
    candidate_drug_name TEXT,         -- 병용 추천 약물 B 이름
    candidate_smiles TEXT NOT NULL,   -- 병용 추천 약물 B SMILES
    synergy_score REAL NOT NULL,      -- SynergyX 예측 점수 (Loewe Score)
    combination_mechanism TEXT,       -- 시너지 기전 설명 (예: Dual EGFR/MET blockade)
    rank_order INTEGER NOT NULL,      -- 추천 순위 (Top-N)
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_step4_analysis ON step4_synergy_predictions(analysis_id);