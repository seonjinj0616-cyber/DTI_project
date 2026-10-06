-- ref_target_disease_evidence 는 여러 증거 종류(유전 연관성/체세포 돌연변이/알려진 약물/문헌 등)의
-- 점수를 최댓값 하나로 합쳐서 저장한다. 이 테이블은 그 내역을 종류별로 남겨서,
-- "왜 이 타깃-질병 쌍이 연관 있다고 보는지"를 더 구체적으로 설명하는 데 쓴다.
CREATE TABLE IF NOT EXISTS ref_target_disease_evidence_detail (
    target_id TEXT NOT NULL,          -- Ensembl Gene ID (ENSG...)
    disease_id TEXT NOT NULL,         -- EFO / MONDO ID
    datatype TEXT NOT NULL,           -- 예: genetic_association, somatic_mutation, known_drug, literature, rna_expression, affected_pathway, animal_model 등
    max_score REAL NOT NULL,          -- 그 종류 안에서의 최고 증거 점수
    PRIMARY KEY (target_id, disease_id, datatype)
);
CREATE INDEX IF NOT EXISTS idx_ref_evidence_detail_target ON ref_target_disease_evidence_detail(target_id);
CREATE INDEX IF NOT EXISTS idx_ref_evidence_detail_pair ON ref_target_disease_evidence_detail(target_id, disease_id);
