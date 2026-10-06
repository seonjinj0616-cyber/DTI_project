import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. Dataset 및 동적 패딩 (Collate) 정의
# ==========================================
class DTIDataset(Dataset):
    def __init__(self, df, mol_tokenizer, esm_tokenizer, task_type):
        self.smiles = df["smiles"].values
        self.sequences = df["protein_sequence"].values
        self.paffinity = df["paffinity"].values

        self.mol_tokenizer = mol_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.task_type = task_type

        # Ki, Kd 분류용 (pAffinity >= 6.0 이면 Active(1), 미만이면 Inactive(0)으로 간주)
        # * 필요시 이 임계값(6.0)을 수정하세요. (pAffinity 6.0은 1uM, 1000nM을 의미합니다)
        if task_type == "classification":
            self.targets = (self.paffinity >= 6.0).astype(float)
        else:
            self.targets = self.paffinity

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # MoLFormer 토큰화
        mol_inputs = self.mol_tokenizer(
            self.smiles[idx], padding=False, truncation=True, return_tensors="pt"
        )
        # ESM-2 토큰화 (이미 1024로 잘려있지만 안정성을 위해 truncation 유지)
        prot_inputs = self.esm_tokenizer(
            self.sequences[idx], padding=False, truncation=True, max_length=1024, return_tensors="pt"
        )

        return {
            "mol_ids": mol_inputs["input_ids"].squeeze(0),
            "mol_mask": mol_inputs["attention_mask"].squeeze(0),
            "prot_ids": prot_inputs["input_ids"].squeeze(0),
            "prot_mask": prot_inputs["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def collate_fn(batch):
    # 배치 내 최대 길이 탐색
    max_mol = max(len(item["mol_ids"]) for item in batch)
    max_prot = max(len(item["prot_ids"]) for item in batch)

    mol_ids, mol_mask = [], []
    prot_ids, prot_mask = [], []
    targets = []

    for item in batch:
        # 분자 패딩 (MoLFormer pad_token_id: 0)
        mol_pad = max_mol - len(item["mol_ids"])
        mol_ids.append(torch.nn.functional.pad(item["mol_ids"], (0, mol_pad), value=0))
        mol_mask.append(torch.nn.functional.pad(item["mol_mask"], (0, mol_pad), value=0))

        # 단백질 패딩 (ESM-2 pad_token_id: 1)
        prot_pad = max_prot - len(item["prot_ids"])
        prot_ids.append(torch.nn.functional.pad(item["prot_ids"], (0, prot_pad), value=1))
        prot_mask.append(torch.nn.functional.pad(item["prot_mask"], (0, prot_pad), value=0))

        targets.append(item["target"])

    return {
        "mol_ids": torch.stack(mol_ids),
        "mol_mask": torch.stack(mol_mask),
        "prot_ids": torch.stack(prot_ids),
        "prot_mask": torch.stack(prot_mask),
        "targets": torch.stack(targets)
    }


# ==========================================
# 2. 2D Cross-Attention 모델 아키텍처
# ==========================================
class DTICrossAttentionModel(nn.Module):
    def __init__(self, task_type="classification", hidden_dim=512):
        super(DTICrossAttentionModel, self).__init__()

        # 사전 학습 모델 로드
        self.esm_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.mol_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        # [중요] 백본 가중치 동결
        for param in self.esm_model.parameters(): param.requires_grad = False
        for param in self.mol_model.parameters(): param.requires_grad = False

        # 차원 정렬
        self.prot_proj = nn.Linear(1280, hidden_dim)
        self.mol_proj = nn.Linear(768, hidden_dim)

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # 최종 압축 및 예측 Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, mol_ids, mol_mask, prot_ids, prot_mask):
        with torch.no_grad():
            mol_out = self.mol_model(mol_ids, attention_mask=mol_mask).last_hidden_state
            prot_out = self.esm_model(prot_ids, attention_mask=prot_mask).last_hidden_state

        mol_feat = self.mol_proj(mol_out)
        prot_feat = self.prot_proj(prot_out)

        attn_out, _ = self.cross_attn(
            query=mol_feat,
            key=prot_feat,
            value=prot_feat,
            key_padding_mask=~prot_mask.bool()
        )

        pooled_out = attn_out.mean(dim=1)
        output = self.fc(pooled_out).squeeze(-1)
        return output


# ==========================================
# 3. 모델 훈련 스크립트
# ==========================================
def train_model(csv_path, model_save_name, task_type, mol_tokenizer, esm_tokenizer, epochs=5, batch_size=8):
    print(f"\n[{model_save_name}] 훈련 시작! 데이터 로딩 중: {csv_path}")

    df = pd.read_csv(csv_path)
    dataset = DTIDataset(df, mol_tokenizer, esm_tokenizer, task_type)

    # 데이터 로더 (Shuffle 필수)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTICrossAttentionModel(task_type=task_type, hidden_dim=512).to(device)

    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            mol_ids = batch["mol_ids"].to(device)
            mol_mask = batch["mol_mask"].to(device)
            prot_ids = batch["prot_ids"].to(device)
            prot_mask = batch["prot_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(mol_ids, mol_mask, prot_ids, prot_mask)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 100 배치마다 로그 출력
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"   Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=> Epoch {epoch + 1} 종료 | 평균 Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_save_name)
    print(f"✅ 모델 저장 완료: {model_save_name}")


# ==========================================
# 4. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    # 데이터베이스 경로 설정
    base_dir = r"/home/team5/workspace/sj/homo_protein_seq_fasta"
    ki_csv = os.path.join(base_dir, "Ki_ems2_fi.csv")
    kd_csv = os.path.join(base_dir, "Kd_ems2_fi.csv")
    ic50_csv = os.path.join(base_dir, "IC50_ems2_fi.csv")

    # 토크나이저 공통 로드
    print("토크나이저 로딩 중...")
    mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # ----------------------------------------------------
    # 훈련 1: Ki 기반 분류 모델 (Active/Inactive 판단용)
    # ----------------------------------------------------
    train_model(ki_csv, "step2_ki_classifier.pt", task_type="classification",
                mol_tokenizer=mol_tokenizer, esm_tokenizer=esm_tokenizer, epochs=5, batch_size=8)

    # ----------------------------------------------------
    # 훈련 2: Kd 기반 분류 모델 (Active/Inactive 판단용)
    # ----------------------------------------------------
    train_model(kd_csv, "step2_kd_classifier.pt", task_type="classification",
                mol_tokenizer=mol_tokenizer, esm_tokenizer=esm_tokenizer, epochs=5, batch_size=8)

    # ----------------------------------------------------
    # 훈련 3: IC50 기반 회귀 모델 (최종 pIC50 수치 예측용)
    # ----------------------------------------------------
    train_model(ic50_csv, "step2_ic50_regressor.pt", task_type="regression",
                mol_tokenizer=mol_tokenizer, esm_tokenizer=esm_tokenizer, epochs=5, batch_size=8)

    print("\n🎉 모든 모델 훈련이 완료되었습니다!")