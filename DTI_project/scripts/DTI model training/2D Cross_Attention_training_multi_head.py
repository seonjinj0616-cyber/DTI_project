import os
import gc
import random
import math
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 0. Configuration
# ============================================================

BASE_DIR = (
    r"/home/team5/workspace/sj/homo_protein_seq_fasta"
)

KI_CSV = os.path.join(BASE_DIR, "Ki_ems2_fi.csv")
KD_CSV = os.path.join(BASE_DIR, "Kd_ems2_fi.csv")
IC50_CSV = os.path.join(BASE_DIR, "IC50_ems2_fi.csv")

# Model
MOL_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

# Model dimensions
MOL_HIDDEN = 768
ESM_HIDDEN = 1280
HIDDEN_DIM = 512

# Cross Attention
NUM_HEADS = 8

# Training
# batch_size가 작으면(예: 8) Kd처럼 비중이 작은 task(4.1%)가
# "최소 1개 보장" 규칙 때문에 실제 비율보다 훨씬 많이 뽑힘
# (8 기준 12.5%, 실제의 3배). batch_size >= 25 정도부터는
# floor(batch_size * 0.041) >= 1이 자연스럽게 성립해서
# 최소-1 규칙이 실질적으로 개입하지 않는다.
# GPU 메모리가 허용하는 선에서 16/32로 올려 실측할 것.
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64          # validation은 grad 없이 도니까 더 크게 잡아도 됨
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 몇 step마다 중간 체크포인트를 저장할지 (epoch 도중 끊겨도 복구 가능하도록)
CHECKPOINT_EVERY_N_STEPS = 2000

# Validation
VAL_RATIO = 0.2

# Classification threshold
PAFFINITY_THRESHOLD = 6.0

# Random seed
SEED = 42

# Sequence preprocessing
MAX_PROTEIN_AA = 1024

# Checkpoint
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_last.pt")
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
RESUME_TRAINING = True        # True면 LAST_CKPT_PATH가 있을 때 자동으로 이어서 학습


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("DEVICE")
print("=" * 80)
print(device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "CUDA Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )


# ============================================================
# 3. Load, Merge, Train/Val Split
# ============================================================

def prepare_multitask_dataframe(base_dir):
    """
    3개 CSV를 읽어 task_idx(0=Ki, 1=Kd, 2=IC50)를 부여하고 합친 뒤,
    task_idx 기준 stratified split으로 train/val (80/20)을 나눈다.

    -> val 안에도 Ki:Kd:IC50 비율이 원본과 동일하게 유지됨.
    -> 동일 화합물/단백질이 task를 넘나드는 것은 허용
       (Ki/Kd/IC50는 서로 다른 assay이므로 값 자체가 다를 수 있음).
    """

    print("\n" + "=" * 80)
    print("1. LOAD DATA")
    print("=" * 80)

    print("Loading Ki...")
    df_ki = pd.read_csv(os.path.join(base_dir, "Ki_ems2_fi.csv"))
    df_ki["task_idx"] = 0

    print("Loading Kd...")
    df_kd = pd.read_csv(os.path.join(base_dir, "Kd_ems2_fi.csv"))
    df_kd["task_idx"] = 1

    print("Loading IC50...")
    df_ic50 = pd.read_csv(os.path.join(base_dir, "IC50_ems2_fi.csv"))
    df_ic50["task_idx"] = 2

    print("\nOriginal dataset size")
    print(f"Ki   : {len(df_ki):,}")
    print(f"Kd   : {len(df_kd):,}")
    print(f"IC50 : {len(df_ic50):,}")

    df_combined = pd.concat(
        [df_ki, df_kd, df_ic50],
        ignore_index=True
    )

    df_combined = df_combined.sample(
        frac=1.0, random_state=SEED
    ).reset_index(drop=True)

    print(f"\nTotal: {len(df_combined):,}")

    print("\nTask distribution (combined)")
    print(df_combined["task_idx"].value_counts().sort_index())

    # --------------------------------------------------------
    # Stratified train / val split by task_idx
    # --------------------------------------------------------

    df_train, df_val = train_test_split(
        df_combined,
        test_size=VAL_RATIO,
        random_state=SEED,
        stratify=df_combined["task_idx"]
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"\nTrain size: {len(df_train):,} ({1 - VAL_RATIO:.0%})")
    print(df_train["task_idx"].value_counts().sort_index())

    print(f"\nVal size  : {len(df_val):,} ({VAL_RATIO:.0%})")
    print(df_val["task_idx"].value_counts().sort_index())

    return df_train, df_val


# ============================================================
# 4. MultiTask Dataset
# ============================================================

class MultiTaskDTIDataset(Dataset):

    def __init__(self, df, mol_tokenizer, esm_tokenizer):

        self.smiles = df["smiles"].values
        self.sequences = df["protein_sequence"].values
        self.paffinity = df["paffinity"].values.astype(np.float32)
        self.task_idx = df["task_idx"].values.astype(np.int64)

        self.mol_tokenizer = mol_tokenizer
        self.esm_tokenizer = esm_tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):

        smiles = self.smiles[idx]
        protein_sequence = self.sequences[idx]

        if len(protein_sequence) > MAX_PROTEIN_AA:
            protein_sequence = protein_sequence[:MAX_PROTEIN_AA]

        mol_inputs = self.mol_tokenizer(
            smiles, padding=False, truncation=True, return_tensors="pt"
        )

        # protein_sequence는 이미 1024aa 이하로 슬라이싱되어 있으므로
        # 여기서 다시 truncation하지 않는다 (위에서 안전장치로 한 번 더 컷).
        prot_inputs = self.esm_tokenizer(
            protein_sequence, padding=False, truncation=False, return_tensors="pt"
        )

        return {
            "mol_ids": mol_inputs["input_ids"].squeeze(0),
            "mol_mask": mol_inputs["attention_mask"].squeeze(0),
            "prot_ids": prot_inputs["input_ids"].squeeze(0),
            "prot_mask": prot_inputs["attention_mask"].squeeze(0),
            "target": torch.tensor(self.paffinity[idx], dtype=torch.float32),
            "task": torch.tensor(self.task_idx[idx], dtype=torch.long)
        }


# ============================================================
# 5. Dynamic Padding
# ============================================================

class MultiTaskCollator:

    def __init__(self, mol_pad_token_id, esm_pad_token_id):
        self.mol_pad_token_id = mol_pad_token_id
        self.esm_pad_token_id = esm_pad_token_id

    def __call__(self, batch):

        max_mol_len = max(len(x["mol_ids"]) for x in batch)
        max_prot_len = max(len(x["prot_ids"]) for x in batch)

        mol_ids, mol_masks = [], []
        prot_ids, prot_masks = [], []
        targets, tasks = [], []

        for item in batch:

            mol_pad_len = max_mol_len - len(item["mol_ids"])
            mol_ids.append(torch.nn.functional.pad(
                item["mol_ids"], (0, mol_pad_len), value=self.mol_pad_token_id
            ))
            mol_masks.append(torch.nn.functional.pad(
                item["mol_mask"], (0, mol_pad_len), value=0
            ))

            prot_pad_len = max_prot_len - len(item["prot_ids"])
            prot_ids.append(torch.nn.functional.pad(
                item["prot_ids"], (0, prot_pad_len), value=self.esm_pad_token_id
            ))
            prot_masks.append(torch.nn.functional.pad(
                item["prot_mask"], (0, prot_pad_len), value=0
            ))

            targets.append(item["target"])
            tasks.append(item["task"])

        return {
            "mol_ids": torch.stack(mol_ids),
            "mol_mask": torch.stack(mol_masks),
            "prot_ids": torch.stack(prot_ids),
            "prot_mask": torch.stack(prot_masks),
            "targets": torch.stack(targets),
            "tasks": torch.stack(tasks)
        }


# ============================================================
# 6. Proportional Task-Balanced Batch Sampler
# ============================================================

class TaskBalancedBatchSampler(Sampler):
    """
    이전 버전의 문제:
        epoch 길이를 가장 적은 task(Kd)에 맞춰서
        Ki/IC50 데이터 대부분이 학습에 쓰이지 못했음.

    수정:
        1) 배치 내 task별 비율을 '실제 데이터 비율'에 비례하게 자동 계산
           (각 task 최소 1개는 보장)
        2) epoch 길이 = 전체 데이터 수 // batch_size
           (= 정상적인 '1 epoch = 전체 데이터 한 바퀴' 정의)
        -> 배치 구성 비율이 실제 비율과 같으므로, 이렇게 하면
           각 task가 한 epoch 동안 자신의 전체 데이터를 대략 한 번씩 소진하게 됨.
           (부족분은 재순환/재샘플링으로 채움)

    주의:
        전체 데이터가 크기 때문에(예: 150만 rows) epoch 당 배치 수가
        매우 커질 수 있다 (batch_size=8 기준 약 18~19만 배치/epoch).
        Frozen encoder라도 매 step forward가 필요하므로 학습 시간이
        상당히 길어질 수 있다. 필요 시 batch_size를 늘리거나
        임베딩 사전 캐싱을 고려할 것.
    """

    def __init__(self, dataset, batch_size=8, seed=42):

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.task_indices = {
            0: np.where(dataset.task_idx == 0)[0].tolist(),
            1: np.where(dataset.task_idx == 1)[0].tolist(),
            2: np.where(dataset.task_idx == 2)[0].tolist()
        }

        counts = {t: len(idx) for t, idx in self.task_indices.items()}
        total = sum(counts.values())

        print("\nTask sampler (proportional)")
        print(f"Ki   : {counts[0]:,} ({counts[0] / total:.2%})")
        print(f"Kd   : {counts[1]:,} ({counts[1] / total:.2%})")
        print(f"IC50 : {counts[2]:,} ({counts[2] / total:.2%})")

        # ----------------------------------------------------
        # 배치 내 task별 개수를 실제 비율에 비례하게 계산
        # (최소 1개는 보장, 남는 슬롯은 소수점 나머지가 큰 순으로 배분)
        # ----------------------------------------------------

        raw = {t: batch_size * counts[t] / total for t in [0, 1, 2]}

        task_batch_sizes = {t: max(1, int(math.floor(raw[t]))) for t in [0, 1, 2]}

        assigned = sum(task_batch_sizes.values())
        remainder = batch_size - assigned

        # 소수점 나머지가 큰 task 순으로 남은 슬롯 배분
        # (remainder가 음수가 될 일은 없음: floor + 최소1 보장이므로
        #  batch_size >= 3일 때 항상 assigned <= batch_size)
        if remainder > 0:
            frac_order = sorted(
                [0, 1, 2],
                key=lambda t: (raw[t] - math.floor(raw[t])),
                reverse=True
            )
            i = 0
            while remainder > 0:
                task_batch_sizes[frac_order[i % 3]] += 1
                remainder -= 1
                i += 1

        self.task_batch_sizes = task_batch_sizes

        print(
            f"\nBatch composition (size={batch_size}): "
            f"Ki={task_batch_sizes[0]}, "
            f"Kd={task_batch_sizes[1]}, "
            f"IC50={task_batch_sizes[2]}"
        )

        # 실제 데이터 비율 대비 batch 내 비율이 얼마나 어긋나는지 표시
        # (특히 Kd처럼 비중이 작은 task는 "최소 1개 보장" 때문에
        #  batch_size가 작을수록 과대 대표될 수 있음)
        print("Sampling ratio vs. actual data ratio:")
        for t, name in [(0, "Ki"), (1, "Kd"), (2, "IC50")]:
            actual_ratio = counts[t] / total
            batch_ratio = task_batch_sizes[t] / batch_size
            skew = batch_ratio / actual_ratio if actual_ratio > 0 else float("inf")
            print(
                f"  {name:5s}: actual={actual_ratio:.2%}  "
                f"batch={batch_ratio:.2%}  "
                f"(x{skew:.2f} oversampling)"
            )

        # ----------------------------------------------------
        # epoch 길이 = 전체 데이터 / batch_size
        # (배치 구성이 실제 비율에 비례하므로, 이렇게 하면
        #  각 task가 한 epoch 동안 자기 데이터를 거의 다 소진함)
        # ----------------------------------------------------

        self.num_batches = total // batch_size

        print(f"Batches per epoch: {self.num_batches:,}")

    def __iter__(self):

        rng = np.random.default_rng(self.seed + self.epoch)

        shuffled = {}
        for task in [0, 1, 2]:
            indices = np.array(self.task_indices[task])
            rng.shuffle(indices)
            shuffled[task] = indices.tolist()

        pointers = {0: 0, 1: 0, 2: 0}

        for _ in range(self.num_batches):

            batch = []

            for task in [0, 1, 2]:

                n = self.task_batch_sizes[task]

                for _ in range(n):

                    if pointers[task] >= len(shuffled[task]):
                        indices = np.array(self.task_indices[task])
                        rng.shuffle(indices)
                        shuffled[task] = indices.tolist()
                        pointers[task] = 0

                    batch.append(shuffled[task][pointers[task]])
                    pointers[task] += 1

            rng.shuffle(batch)

            yield batch

        self.epoch += 1

    def __len__(self):
        return self.num_batches


# ============================================================
# 7. Multi-Task Model
# ============================================================

class MultiTaskDTIModel(nn.Module):

    def __init__(self, hidden_dim=512, num_heads=8):

        super().__init__()

        print("\nLoading ESM-2...")
        self.esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME)

        print("Loading MoLFormer...")
        self.mol_model = AutoModel.from_pretrained(
            MOL_MODEL_NAME, trust_remote_code=True
        )

        print("Freezing pretrained backbones...")
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.mol_model.parameters():
            param.requires_grad = False

        self.prot_proj = nn.Linear(ESM_HIDDEN, hidden_dim)
        self.mol_proj = nn.Linear(MOL_HIDDEN, hidden_dim)

        # Drug = Query, Protein = Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.interaction_dropout = nn.Dropout(0.2)

        self.ki_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.kd_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )
        self.ic50_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
        )

    def forward(self, mol_ids, mol_mask, prot_ids, prot_mask):

        with torch.no_grad():
            mol_out = self.mol_model(
                mol_ids, attention_mask=mol_mask
            ).last_hidden_state
            prot_out = self.esm_model(
                prot_ids, attention_mask=prot_mask
            ).last_hidden_state

        mol_feat = self.mol_proj(mol_out)
        prot_feat = self.prot_proj(prot_out)

        attn_out, _ = self.cross_attn(
            query=mol_feat,
            key=prot_feat,
            value=prot_feat,
            key_padding_mask=~prot_mask.bool(),
            need_weights=False
        )

        interaction = mol_feat + attn_out
        interaction = self.interaction_norm(interaction)
        interaction = self.interaction_dropout(interaction)

        # Masked mean pooling (molecule padding 제외)
        mol_mask_float = mol_mask.unsqueeze(-1).float()
        interaction = interaction * mol_mask_float
        pooled = interaction.sum(dim=1) / mol_mask_float.sum(dim=1).clamp(min=1e-6)

        ki_pred = self.ki_head(pooled).squeeze(-1)
        kd_pred = self.kd_head(pooled).squeeze(-1)
        ic50_pred = self.ic50_head(pooled).squeeze(-1)

        return ki_pred, kd_pred, ic50_pred


# ============================================================
# 8. Multi-Task Loss
# ============================================================

def multitask_loss(ki_pred, kd_pred, ic50_pred, targets, tasks):

    total_loss = 0.0
    loss_count = 0
    loss_dict = {}

    ki_mask = (tasks == 0)
    if ki_mask.any():
        ki_target = (targets[ki_mask] >= PAFFINITY_THRESHOLD).float()
        ki_loss = nn.functional.binary_cross_entropy_with_logits(
            ki_pred[ki_mask], ki_target
        )
        total_loss += ki_loss
        loss_count += 1
        loss_dict["ki"] = ki_loss.detach().item()

    kd_mask = (tasks == 1)
    if kd_mask.any():
        kd_target = (targets[kd_mask] >= PAFFINITY_THRESHOLD).float()
        kd_loss = nn.functional.binary_cross_entropy_with_logits(
            kd_pred[kd_mask], kd_target
        )
        total_loss += 1.5 * kd_loss
        loss_count += 1
        loss_dict["kd"] = kd_loss.detach().item()

    ic50_mask = (tasks == 2)
    if ic50_mask.any():
        ic50_loss = nn.functional.mse_loss(
            ic50_pred[ic50_mask], targets[ic50_mask]
        )
        total_loss += ic50_loss
        loss_count += 1
        loss_dict["ic50"] = ic50_loss.detach().item()

    if loss_count == 0:
        return (
            torch.tensor(0.0, device=targets.device, requires_grad=True),
            loss_dict
        )

    total_loss = total_loss / loss_count
    return total_loss, loss_dict


# ============================================================
# 9. Validation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Validation set 전체에 대해:
        - loss (train과 동일한 multitask_loss)
        - Ki / Kd: Accuracy, AUC
        - IC50: RMSE, R2
    를 계산한다.
    """

    model.eval()

    total_loss = 0.0
    n_batches = 0

    ki_probs, ki_labels = [], []
    kd_probs, kd_labels = [], []
    ic50_preds, ic50_labels = [], []

    for batch in dataloader:

        mol_ids = batch["mol_ids"].to(device)
        mol_mask = batch["mol_mask"].to(device)
        prot_ids = batch["prot_ids"].to(device)
        prot_mask = batch["prot_mask"].to(device)
        targets = batch["targets"].to(device)
        tasks = batch["tasks"].to(device)

        ki_pred, kd_pred, ic50_pred = model(mol_ids, mol_mask, prot_ids, prot_mask)

        loss, _ = multitask_loss(ki_pred, kd_pred, ic50_pred, targets, tasks)
        total_loss += loss.item()
        n_batches += 1

        ki_mask = (tasks == 0)
        if ki_mask.any():
            probs = torch.sigmoid(ki_pred[ki_mask]).cpu().numpy()
            labels = (targets[ki_mask] >= PAFFINITY_THRESHOLD).float().cpu().numpy()
            ki_probs.extend(probs.tolist())
            ki_labels.extend(labels.tolist())

        kd_mask = (tasks == 1)
        if kd_mask.any():
            probs = torch.sigmoid(kd_pred[kd_mask]).cpu().numpy()
            labels = (targets[kd_mask] >= PAFFINITY_THRESHOLD).float().cpu().numpy()
            kd_probs.extend(probs.tolist())
            kd_labels.extend(labels.tolist())

        ic50_mask = (tasks == 2)
        if ic50_mask.any():
            preds = ic50_pred[ic50_mask].cpu().numpy()
            labels = targets[ic50_mask].cpu().numpy()
            ic50_preds.extend(preds.tolist())
            ic50_labels.extend(labels.tolist())

    metrics = {"val_loss": total_loss / max(1, n_batches)}

    if len(ki_labels) > 0:
        ki_pred_label = [1 if p >= 0.5 else 0 for p in ki_probs]
        metrics["ki_acc"] = accuracy_score(ki_labels, ki_pred_label)
        # 라벨이 한 클래스만 있는 극단적인 경우 AUC 계산 불가 -> 예외 처리
        try:
            metrics["ki_auc"] = roc_auc_score(ki_labels, ki_probs)
        except ValueError:
            metrics["ki_auc"] = float("nan")

    if len(kd_labels) > 0:
        kd_pred_label = [1 if p >= 0.5 else 0 for p in kd_probs]
        metrics["kd_acc"] = accuracy_score(kd_labels, kd_pred_label)
        try:
            metrics["kd_auc"] = roc_auc_score(kd_labels, kd_probs)
        except ValueError:
            metrics["kd_auc"] = float("nan")

    if len(ic50_labels) > 0:
        metrics["ic50_rmse"] = mean_squared_error(
            ic50_labels, ic50_preds, squared=False
        )
        metrics["ic50_r2"] = r2_score(ic50_labels, ic50_preds)

    model.train()

    return metrics


def print_metrics(prefix, metrics):

    print(f"\n{prefix}")
    print(f"  loss      : {metrics.get('val_loss', float('nan')):.6f}")

    if "ki_acc" in metrics:
        print(f"  Ki  acc   : {metrics['ki_acc']:.4f}  | AUC: {metrics['ki_auc']:.4f}")
    if "kd_acc" in metrics:
        print(f"  Kd  acc   : {metrics['kd_acc']:.4f}  | AUC: {metrics['kd_auc']:.4f}")
    if "ic50_rmse" in metrics:
        print(f"  IC50 RMSE : {metrics['ic50_rmse']:.4f}  | R2: {metrics['ic50_r2']:.4f}")


# ============================================================
# 10. Checkpoint helpers
# ============================================================

def save_checkpoint(path, model, optimizer, next_epoch, next_batch_idx, best_val_loss):
    """
    저장 시점에 관계없이(epoch 중간이든 끝이든) 항상
    '다음에 재개해야 할 지점'을 next_epoch / next_batch_idx로 명시한다.

    - epoch 도중 저장: next_epoch=현재 epoch, next_batch_idx=지금까지 처리한 batch 수
    - epoch 완료 후 저장: next_epoch=현재epoch+1, next_batch_idx=0

    TaskBalancedBatchSampler는 self.seed + self.epoch로만 시드가 결정되는
    완전 결정론적 구조이므로(num_workers=0 기준), RNG state를 별도로
    저장하지 않아도 next_epoch/next_batch_idx만 있으면 동일한 batch 순서를
    정확히 재현할 수 있다.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "next_batch_idx": next_batch_idx,
        "best_val_loss": best_val_loss,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "threshold": PAFFINITY_THRESHOLD,
        "mol_model": MOL_MODEL_NAME,
        "esm_model": ESM_MODEL_NAME,
        "task_definition": {
            "Ki": "pAffinity >= 6.0 -> Active (classification)",
            "Kd": "pAffinity >= 6.0 -> Active (classification)",
            "IC50": "pAffinity regression"
        }
    }, path)


def load_checkpoint(path, model, optimizer, device):

    print(f"\nResuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt["next_epoch"]
    start_batch_idx = ckpt["next_batch_idx"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print(
        f"  -> resuming at epoch {start_epoch + 1}, "
        f"batch {start_batch_idx + 1}, "
        f"best_val_loss={best_val_loss:.6f}"
    )

    return start_epoch, start_batch_idx, best_val_loss


# ============================================================
# 11. Training
# ============================================================

def train_multitask():

    print("\n" + "=" * 80)
    print("MULTI-TASK DTI TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # Data (train/val split 포함)
    # --------------------------------------------------------

    df_train, df_val = prepare_multitask_dataframe(BASE_DIR)

    # --------------------------------------------------------
    # Tokenizers
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("2. LOAD TOKENIZERS")
    print("=" * 80)

    mol_tokenizer = AutoTokenizer.from_pretrained(
        MOL_MODEL_NAME, trust_remote_code=True
    )
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)

    print("MoLFormer pad token:", mol_tokenizer.pad_token_id)
    print("ESM-2 pad token:", esm_tokenizer.pad_token_id)

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = MultiTaskDTIDataset(df_train, mol_tokenizer, esm_tokenizer)
    val_dataset = MultiTaskDTIDataset(df_val, mol_tokenizer, esm_tokenizer)

    collator = MultiTaskCollator(
        mol_pad_token_id=mol_tokenizer.pad_token_id,
        esm_pad_token_id=esm_tokenizer.pad_token_id
    )

    # --------------------------------------------------------
    # Train: proportional task-balanced sampler
    # --------------------------------------------------------

    batch_sampler = TaskBalancedBatchSampler(
        train_dataset, batch_size=BATCH_SIZE, seed=SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Val: 밸런싱 없이 그냥 순차 평가
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nTrain batches per epoch:", len(train_loader))
    print("Val batches:", len(val_loader))

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("3. LOAD MODEL")
    print("=" * 80)

    model = MultiTaskDTIModel(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Resume (optional)
    # --------------------------------------------------------

    start_epoch = 0
    start_batch_idx = 0
    best_val_loss = float("inf")

    if RESUME_TRAINING and os.path.exists(LAST_CKPT_PATH):
        start_epoch, start_batch_idx, best_val_loss = load_checkpoint(
            LAST_CKPT_PATH, model, optimizer, device
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("4. TRAINING")
    print("=" * 80)

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        total_loss = 0.0
        total_ki_loss = 0.0
        total_kd_loss = 0.0
        total_ic50_loss = 0.0
        ki_steps = 0
        kd_steps = 0
        ic50_steps = 0
        processed_batches = 0   # 평균 계산용 (resume 시 len(train_loader)와 다를 수 있음)

        batch_sampler.epoch = epoch

        # ----------------------------------------------------
        # Resume: 이 epoch에서 재개해야 할 경우, 이미 처리한
        # batch만큼 스킵한다. sampler가 epoch 번호로만 시드가
        # 결정되는 결정론적 구조이므로, 같은 epoch을 다시 돌리면
        # 항상 동일한 batch 순서가 나온다 -> RNG state 저장 불필요.
        # ----------------------------------------------------

        resume_batch_idx = start_batch_idx if epoch == start_epoch else 0

        loader_iter = iter(train_loader)

        if resume_batch_idx > 0:
            print(
                f"\n  -> Resuming epoch {epoch + 1}: "
                f"skipping {resume_batch_idx}/{len(train_loader)} "
                f"already-completed batches..."
            )
            for _ in range(resume_batch_idx):
                next(loader_iter)

        # ----------------------------------------------------
        # Throughput / ETA 측정
        # ----------------------------------------------------

        epoch_start_time = time.time()
        window_start_time = epoch_start_time
        window_start_idx = resume_batch_idx

        for batch_idx, batch in enumerate(loader_iter, start=resume_batch_idx):

            mol_ids = batch["mol_ids"].to(device, non_blocking=True)
            mol_mask = batch["mol_mask"].to(device, non_blocking=True)
            prot_ids = batch["prot_ids"].to(device, non_blocking=True)
            prot_mask = batch["prot_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            tasks = batch["tasks"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            ki_pred, kd_pred, ic50_pred = model(mol_ids, mol_mask, prot_ids, prot_mask)

            loss, loss_dict = multitask_loss(
                ki_pred, kd_pred, ic50_pred, targets, tasks
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            processed_batches += 1

            if "ki" in loss_dict:
                total_ki_loss += loss_dict["ki"]
                ki_steps += 1
            if "kd" in loss_dict:
                total_kd_loss += loss_dict["kd"]
                kd_steps += 1
            if "ic50" in loss_dict:
                total_ic50_loss += loss_dict["ic50"]
                ic50_steps += 1

            if (batch_idx + 1) % 500 == 0:

                now = time.time()
                steps_in_window = (batch_idx + 1) - window_start_idx
                sec_per_step = (now - window_start_time) / max(1, steps_in_window)
                remaining_steps = len(train_loader) - (batch_idx + 1)
                eta_sec = remaining_steps * sec_per_step

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Step [{batch_idx + 1}/{len(train_loader)}] "
                    f"Total Loss: {loss.item():.4f} "
                    f"| Ki: {loss_dict.get('ki', 0):.4f} "
                    f"| Kd: {loss_dict.get('kd', 0):.4f} "
                    f"| IC50: {loss_dict.get('ic50', 0):.4f} "
                    f"| {sec_per_step:.3f}s/step "
                    f"| ETA(this epoch): {eta_sec / 3600:.2f}h"
                )

                window_start_time = now
                window_start_idx = batch_idx + 1

            # ----------------------------------------------------
            # 안전장치: 일정 간격마다 중간 체크포인트 저장
            # (epoch 중간에 끊겨도 정확히 이 batch부터 재개 가능)
            # ----------------------------------------------------

            if (batch_idx + 1) % CHECKPOINT_EVERY_N_STEPS == 0:
                save_checkpoint(
                    LAST_CKPT_PATH, model, optimizer,
                    next_epoch=epoch,
                    next_batch_idx=batch_idx + 1,
                    best_val_loss=best_val_loss
                )

        avg_total_loss = total_loss / max(1, processed_batches)

        elapsed = time.time() - epoch_start_time
        print("\n" + "-" * 80)
        print(f"Epoch {epoch + 1} completed ({elapsed / 3600:.2f}h, {processed_batches} batches processed)")
        print(f"Average Total Loss : {avg_total_loss:.6f}")

        if ki_steps > 0:
            print(f"Average Ki Loss    : {total_ki_loss / ki_steps:.6f}")
        if kd_steps > 0:
            print(f"Average Kd Loss    : {total_kd_loss / kd_steps:.6f}")
        if ic50_steps > 0:
            print(f"Average IC50 Loss  : {total_ic50_loss / ic50_steps:.6f}")

        print("-" * 80)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(model, val_loader, device)
        print_metrics(f"[Validation] Epoch {epoch + 1}", val_metrics)

        # ----------------------------------------------------
        # Checkpoint: epoch 완료 -> 다음 epoch 처음부터 재개하도록 저장
        # ----------------------------------------------------

        save_checkpoint(
            LAST_CKPT_PATH, model, optimizer,
            next_epoch=epoch + 1, next_batch_idx=0,
            best_val_loss=best_val_loss
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                BEST_CKPT_PATH, model, optimizer,
                next_epoch=epoch + 1, next_batch_idx=0,
                best_val_loss=best_val_loss
            )
            print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Last checkpoint : {LAST_CKPT_PATH}")
    print(f"Best checkpoint : {BEST_CKPT_PATH} (val_loss={best_val_loss:.6f})")
    print("=" * 80)


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":
    train_multitask()
