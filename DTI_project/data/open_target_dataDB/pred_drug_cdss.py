import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: SMILES 입력 -> 타깃 단백질 예측 (PyTDC / ChemBERTa)
# ---------------------------------------------------------------------------
def predict_target_proteins(smiles_str: str, top_k: int = 5) -> pd.DataFrame:
    """
    입력된 SMILES 분자 구조로부터 결합 가능성이 높은 Target Protein 스코어링
    """
    print(f"[Step 1] SMILES 구조식 분석 및 Target Protein 예측 중...")

    # 실제 구현: PyTDC DTI / ChemBERTa 모델을 통한 Affinity 계산
    predicted_targets = [
        {"target_id": "P00533", "gene_name": "EGFR", "affinity_score": 0.94},
        {"target_id": "P35968", "gene_name": "KDR (VEGFR2)", "affinity_score": 0.81},
        {"target_id": "P04626", "gene_name": "ERBB2 (HER2)", "affinity_score": 0.75},
        {"target_id": "P15056", "gene_name": "BRAF", "affinity_score": 0.42},
        {"target_id": "P53779", "gene_name": "MAPK10", "affinity_score": 0.38},
    ]

    return pd.DataFrame(predicted_targets[:top_k])


# ---------------------------------------------------------------------------
# Step 2: 타깃 단백질 기반 MoA (Mechanism of Action) 경로 추출
# ---------------------------------------------------------------------------
def predict_moa_pathways(target_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    1단계에서 도출된 타깃 단백질을 바탕으로 관련 MoA (Pathway) 영향도 스코어링
    """
    print("[Step 2] 타깃 단백질 기반 MoA (Mechanism of Action) 경로 분석 중...")

    # 실제 구현: Open Targets / Reactome 지식 그래프 매핑
    pathways_data = [
        {"pathway_id": "R-HSA-177929", "pathway_name": "Signaling by EGFR", "moa_impact_score": 0.92},
        {"pathway_id": "R-HSA-1227123", "pathway_name": "PI3K/AKT Activation", "moa_impact_score": 0.88},
        {"pathway_id": "R-HSA-5683057", "pathway_name": "MAPK Family Signaling Cascades", "moa_impact_score": 0.79},
        {"pathway_id": "R-HSA-169911", "pathway_name": "Regulation of RAS by GAPs", "moa_impact_score": 0.65},
        {"pathway_id": "R-HSA-19539", "pathway_name": "FOXO-mediated transcription", "moa_impact_score": 0.54},
    ]

    return pd.DataFrame(pathways_data[:top_k])


# ---------------------------------------------------------------------------
# Step 3: 지식 그래프(TxGNN) 기반 적응증 질병군 및 부작용 예측
# ---------------------------------------------------------------------------
def predict_diseases_and_side_effects(pathway_df: pd.DataFrame, target_df: pd.DataFrame):
    """
    TxGNN 등의 지식 그래프 파운데이션 모델을 통해 적응증 질병군과 Off-target 부작용 탐색
    """
    print("[Step 3] 지식 그래프(TxGNN) 기반 치료 후보 질병군 및 부작용 탐색 중...")

    # 1. 치료 가능 후보 질병군 (Indications)
    candidate_diseases = [
        {"disease_code": "MONDO:0003588", "disease_name": "Non-Small Cell Lung Cancer", "confidence_score": 0.89},
        {"disease_code": "MONDO:0005061", "disease_name": "Glioblastoma", "confidence_score": 0.76},
        {"disease_code": "MONDO:0005233", "disease_name": "Colorectal Cancer", "confidence_score": 0.71},
    ]

    # 2. 오프타깃/MoA 기반 예상 부작용 (Side Effects)
    predicted_side_effects = [
        {"side_effect": "Diarrhea", "off_target_related": "EGFR Inhibition in Gut", "risk_level": "High"},
        {"side_effect": "Skin Rash", "off_target_related": "Cutaneous EGFR Inhibition", "risk_level": "High"},
        {"side_effect": "QT Prolongation", "off_target_related": "hERG Channel interaction", "risk_level": "Medium"},
    ]

    return pd.DataFrame(candidate_diseases), pd.DataFrame(predicted_side_effects)


# ---------------------------------------------------------------------------
# Step 4: SynergyX 멀티모달 모델을 통한 시너지 후보 병용 약물 예측
# ---------------------------------------------------------------------------
def predict_synergistic_drug_combinations(
    primary_smiles: str,
    target_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    cell_line_id: str = "A549",
    top_k: int = 5
) -> pd.DataFrame:
    """
    SynergyX 모델 구동:
    Step 1의 Target + Step 2의 MoA Pathway 정보를 멀티모달 피처로 결합하여 병용 시너지 예측
    """
    print(f"[Step 4] SynergyX 멀티모달 모델 구동: 타겟 세포주({cell_line_id}) 및 MoA 경로 피처 결합 분석 중...")

    # SynergyX 멀티모달 입력 파이프라인:
    # 1. Primary SMILES + Partner SMILES -> GNN Encoder
    # 2. Step 1 (Target IDs) + Step 2 (MoA Pathway IDs) -> Target/Pathway Feature Vector
    # 3. Cell Line ID -> Gene Expression Vector
    # 4. Multi-modal Fusion -> Synergy Score 계산

    synergy_candidates = [
        {"partner_drug_name": "Osimertinib", "partner_smiles": "COC1=C...", "synergy_score": 88.4, "status": "High Synergy"},
        {"partner_drug_name": "Trametinib", "partner_smiles": "CC1=C...", "synergy_score": 77.1, "status": "High Synergy"},
        {"partner_drug_name": "Cetuximab", "partner_smiles": "Protein...", "synergy_score": 58.5, "status": "Moderate"},
        {"partner_drug_name": "Paclitaxel", "partner_smiles": "CC1=C...", "synergy_score": 32.0, "status": "Low Synergy"},
    ]

    return pd.DataFrame(synergy_candidates[:top_k])


# ===========================================================================
# CDSS 통합 파이프라인 실행
# ===========================================================================
def run_cdss_pipeline(input_smiles: str, cell_line: str = "A549"):
    print("===============================================================")
    print("         차세대 신약 재창출 및 정밀 처방 지원 CDSS 시스템         ")
    print("===============================================================\n")

    # 1단계: 타깃 예측 (SMILES -> Protein Target)
    targets = predict_target_proteins(input_smiles)

    # 2단계: MoA 경로 예측 (Target -> Pathway)
    moa_pathways = predict_moa_pathways(targets)

    # 3단계: 질병군 및 부작용 예측 (TxGNN Explainer)
    diseases, side_effects = predict_diseases_and_side_effects(moa_pathways, targets)

    # 4단계: SynergyX 기반 병용 약물 예측 (2단계 MoA 데이터 주입)
    synergies = predict_synergistic_drug_combinations(
        primary_smiles=input_smiles,
        target_df=targets,
        pathway_df=moa_pathways,  # 2단계의 MoA 경로 결과 전달
        cell_line_id=cell_line
    )

    # 리포트 출력
    print("\n[최종 CDSS 분석 결과 리포트]")
    print("\n1. 예측된 타깃 단백질 (Top-K):")
    print(targets.to_string(index=False))

    print("\n2. 영향 받는 주요 MoA 경로 (Top 5):")
    print(moa_pathways.to_string(index=False))

    print("\n3-1. 치료 대상 후보 질병군 (Indications):")
    print(diseases.to_string(index=False))

    print("\n3-2. 예상 주요 부작용 (Side Effects):")
    print(side_effects.to_string(index=False))

    print(f"\n4. SynergyX 예측 병용 추천 약물 (세포주: {cell_line}):")
    print(synergies.to_string(index=False))


# 실행 테스트 (Gefitinib SMILES 예시)
example_smiles = "COC1=C(OCC2=CC=CC=C2)C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl"
run_cdss_pipeline(example_smiles, cell_line="A549")