import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any


class ChemBERTaTargetPredictor:
    def __init__(self, model_name_or_path: str = "DeepChem/ChemBERTa-77M-MTR", num_targets: int = 100):
        """
        ChemBERTa-v2 모델 및 토크나이저 초기화
        :param model_name_or_path: Hugging Face 모델명 또는 사전 학습된 로컬 파인튜닝 모델 경로
        :param num_targets: 예측할 타깃 단백질(ENSG ID) 개수
        """
        print(f"🔄 모델 로딩 중: {model_name_or_path}...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ 실행 디바이스: {self.device}")

        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # 파인튜닝된 타깃 예측 분류 헤드가 포함된 모델 로드 (없을 시 베이스 모델 로드)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_targets
            ).to(self.device)
        except Exception:
            print("⚠️ 파인튜닝 헤드가 없는 기본 모델입니다. Feature Extractor 기반 임시 분류기로 동작합니다.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                ignore_mismatched_sizes=True
            ).to(self.device)

        self.model.eval()

        # 예시용 대표 타깃 단백질 ENSG ID 매핑 테이블 (실제 파인튜닝 모델의 target_id 목록으로 교체 가능)
        self.target_labels = [
            "ENSG00000146648",  # EGFR
            "ENSG00000171094",  # ALK
            "ENSG00000157764",  # BRAF
            "ENSG00000133703",  # KRAS
            "ENSG00000105976",  # MET
            "ENSG00000141510",  # TP53
            "ENSG00000121879",  # PIK3CA
            "ENSG00000171862",  # PTEN
            "ENSG00000136997",  # MYC
            "ENSG00000142192"  # APP
        ]

    def predict(self, smiles: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        SMILES 문자열을 입력받아 상위 top_k 타깃 단백질 예측 결과를 반환
        """
        if not smiles or not isinstance(smiles, str):
            raise ValueError("유효한 SMILES 문자열을 입력해주세요.")

        # 1. SMILES 토큰화
        inputs = self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 2. 모델 순전파 (Inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Softmax를 이용한 확률값 산출
            probs = F.softmax(logits, dim=-1)[0]

        # 3. Top-K 추출
        k = min(top_k, len(self.target_labels), probs.shape[-1])
        top_probs, top_indices = torch.topk(probs, k=k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            target_id = self.target_labels[idx.item()] if idx.item() < len(
                self.target_labels) else f"ENSG_UNKNOWN_{idx.item()}"
            results.append({
                "target_id": target_id,
                "confidence_score": round(prob.item(), 4)
            })

        return results


if __name__ == "__main__":
    # 테스트용 대표 약물 SMILES (Gefitinib / EGFR Inhibitor)
    test_smiles = "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"

    print("==================================================")
    print("🧪 Step 1: ChemBERTa-v2 Target Prediction")
    print("==================================================")
    print(f"🔹 입력 SMILES: {test_smiles}\n")

    # 예측기 초기화 및 예측 수행
    predictor = ChemBERTaTargetPredictor()
    predictions = predictor.predict(smiles=test_smiles, top_k=3)

    print("🎯 예측된 상위 타깃 단백질 (Top-3):")
    for rank, pred in enumerate(predictions, 1):
        print(f"   {rank}. Target ID: {pred['target_id']} (신뢰도 스코어: {pred['confidence_score']})")

    print("==================================================")