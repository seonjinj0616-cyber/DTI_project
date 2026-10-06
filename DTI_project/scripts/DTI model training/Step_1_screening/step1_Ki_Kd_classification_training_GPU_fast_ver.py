import os
import gc
import copy
import time
import random

import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset,
    Sampler
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

# 결합 여부 기준 (paffinity >= THRESHOLD -> positive)
THRESHOLD = 7.0

# ---- Epoch / Early stopping --------------------------------
# MAX_EPOCHS는 "상한선"일 뿐이고, 실제로는 early stopping이
# 검증 성능이 더 이상 좋아지지 않을 때 학습을 멈춘다.
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10      # 이 epoch 수만큼 개선이 없으면 종료
MIN_DELTA = 1e-4              # 이 값 이상 좋아져야 "개선"으로 인정

# ---- LR scheduler (plateau 시 model LR 감소) ---------------
USE_LR_SCHEDULER = True
LR_FACTOR = 0.5
LR_PATIENCE = 3               # EARLY_STOP_PATIENCE보다 작아야 의미 있음
MIN_LR = 1e-6

BATCH_SIZE = 1024

GRADNORM_ALPHA = 1.5

MODEL_LR = 1e-3
WEIGHT_LR = 1e-3

NUM_WORKERS = 0
PIN_MEMORY = True

# 기존 threshold 6.0 모델 파일을 덮어쓰지 않도록 파일명 변경
# (스크리닝 스크립트의 KIKD_MODEL_PATH도 이 이름으로 바꿔줘야 함)
SAVE_PATH = (
    "/home/team5/workspace/sj/"
    "step1_multitask_kikd_gradnorm_thr8.pt"
)


# ============================================================
# 1. Seed
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. In-Memory H5 Dataset
# ============================================================

class DTIVectorDatasetInMemory(Dataset):

    def __init__(self, h5_path, task_idx=0, threshold=7.0):

        print(f"\n[{os.path.basename(h5_path)}]")
        print("RAM 메모리에 데이터 로딩 중...")

        start_time = time.time()

        with h5py.File(h5_path, "r") as f:

            self.drug_vectors = torch.tensor(
                f["interactions"]["drug_vector"][:],
                dtype=torch.float32
            )

            self.prot_vectors = torch.tensor(
                f["interactions"]["protein_vector"][:],
                dtype=torch.float32
            )

            paffinity = f["interactions"]["paffinity"][:]

        targets = (paffinity >= threshold).astype(np.float32)

        self.targets = torch.tensor(targets, dtype=torch.float32)

        self.task_idx = torch.tensor(task_idx, dtype=torch.long)

        elapsed = time.time() - start_time

        pos_ratio = float(self.targets.mean().item())

        print(f"로드 완료: {len(self.targets):,} samples")
        print(f"로드 시간: {elapsed:.2f} sec")
        print(f"Drug vector shape: {tuple(self.drug_vectors.shape)}")
        print(f"Protein vector shape: {tuple(self.prot_vectors.shape)}")
        print(f"Positive ratio (paffinity >= {threshold}): {pos_ratio:.4f}")
        print(f"RAM 사용량 추정: {self.memory_size_gb():.2f} GB")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.drug_vectors[idx],
            self.prot_vectors[idx],
            self.targets[idx],
            self.task_idx
        )

    def memory_size_gb(self):

        total_bytes = (
            self.drug_vectors.numel() * self.drug_vectors.element_size()
            + self.prot_vectors.numel() * self.prot_vectors.element_size()
            + self.targets.numel() * self.targets.element_size()
        )

        return total_bytes / 1024**3


# ============================================================
# 3. Exact Ratio Batch Sampler
# ============================================================

class ExactRatioBatchSampler(Sampler):

    def __init__(self, ki_indices, kd_indices, batch_size=1024):

        self.ki_indices = torch.tensor(ki_indices, dtype=torch.long)
        self.kd_indices = torch.tensor(kd_indices, dtype=torch.long)

        self.batch_size = batch_size

        # 실제 Ki : Kd 비율
        ratio = len(self.ki_indices) / len(self.kd_indices)

        # Kd가 모든 batch에 포함되도록 계산
        self.num_kd = max(1, int(batch_size / (ratio + 1.0)))
        self.num_ki = batch_size - self.num_kd

        # 완전한 batch만 사용
        self.num_batches = min(
            len(self.ki_indices) // self.num_ki,
            len(self.kd_indices) // self.num_kd
        )

    def __iter__(self):

        ki_shuffled = self.ki_indices[torch.randperm(len(self.ki_indices))]
        kd_shuffled = self.kd_indices[torch.randperm(len(self.kd_indices))]

        for i in range(self.num_batches):

            ki_batch = ki_shuffled[i * self.num_ki:(i + 1) * self.num_ki]
            kd_batch = kd_shuffled[i * self.num_kd:(i + 1) * self.num_kd]

            batch = torch.cat([ki_batch, kd_batch]).tolist()

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================
# 4. Multi-Task Ki/Kd Model
# ============================================================

class MultiTask_KiKd_Classifier(nn.Module):

    def __init__(self, mol_dim=768, prot_dim=1280, hidden_dim=512):

        super().__init__()

        input_dim = mol_dim + prot_dim

        # Shared encoder
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task heads
        self.ki_head = nn.Linear(256, 1)
        self.kd_head = nn.Linear(256, 1)

    def forward(self, mol_vec, prot_vec):

        combined = torch.cat([mol_vec, prot_vec], dim=1)

        features = self.shared_body(combined)

        ki_logits = self.ki_head(features).squeeze(-1)
        kd_logits = self.kd_head(features).squeeze(-1)

        return ki_logits, kd_logits

    def get_shared_layer(self):
        # GradNorm에서 gradient norm을 측정할 shared parameter
        return self.shared_body[4].weight


# ============================================================
# 5. Metrics
# ============================================================

METRIC_NAMES = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]


def calculate_metrics(targets, probs):

    targets = np.asarray(targets)
    probs = np.asarray(probs)

    nan_result = {name: np.nan for name in METRIC_NAMES}

    if len(targets) == 0:
        return nan_result

    if len(np.unique(targets)) < 2:
        return nan_result

    preds = (probs >= 0.5).astype(int)

    return {
        "AUROC": roc_auc_score(targets, probs),
        "AUPRC": average_precision_score(targets, probs),
        "F1": f1_score(targets, preds, zero_division=0),
        "Precision": precision_score(targets, preds, zero_division=0),
        "Recall": recall_score(targets, preds, zero_division=0)
    }


def to_python_floats(metrics):
    return {k: float(v) for k, v in metrics.items()}


def compute_selection_score(ki_metrics, kd_metrics):
    """
    Best epoch 선택 기준: Ki AUPRC와 Kd AUPRC의 평균.

    threshold 8.0에서는 positive 비율이 낮아질 가능성이 커서
    AUROC/F1(고정 0.5 cutoff)보다 AUPRC가 더 안정적인 기준이다.
    스크리닝 단계에서는 '확률 순위'가 중요하므로 순위 기반 지표(AUPRC)가 적합하다.
    """
    values = [ki_metrics["AUPRC"], kd_metrics["AUPRC"]]

    if any(np.isnan(v) for v in values):
        return float("-inf")

    return float(np.mean(values))


# ============================================================
# 6. GPU Information
# ============================================================

def print_gpu_info():

    print("\n" + "=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA 사용 불가")
        print("현재 CPU로 실행됩니다.")
        print("=" * 70)
        return

    device = torch.device("cuda")

    print(f"Device : {device}")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"CUDA : {torch.version.cuda}")

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"VRAM : {total_memory:.2f} GB")
    print("=" * 70)


# ============================================================
# 7. Validation
# ============================================================

def evaluate(model, val_loader, device, criterion):

    model.eval()

    ki_targets, ki_probs = [], []
    kd_targets, kd_probs = [], []

    ki_loss_total = 0.0
    kd_loss_total = 0.0

    ki_count = 0
    kd_count = 0

    with torch.no_grad():

        for mol, prot, targets, tasks in val_loader:

            mol = mol.to(device, non_blocking=True)
            prot = prot.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tasks = tasks.to(device, non_blocking=True)

            ki_logits, kd_logits = model(mol, prot)

            ki_mask = (tasks == 0)
            kd_mask = (tasks == 1)

            # ----------------------------
            # Ki
            # ----------------------------
            if ki_mask.any():

                ki_loss = criterion(ki_logits[ki_mask], targets[ki_mask])

                count = ki_mask.sum().item()

                ki_loss_total += ki_loss.item() * count
                ki_count += count

                probs = torch.sigmoid(ki_logits[ki_mask])

                ki_targets.extend(targets[ki_mask].cpu().numpy())
                ki_probs.extend(probs.cpu().numpy())

            # ----------------------------
            # Kd
            # ----------------------------
            if kd_mask.any():

                kd_loss = criterion(kd_logits[kd_mask], targets[kd_mask])

                count = kd_mask.sum().item()

                kd_loss_total += kd_loss.item() * count
                kd_count += count

                probs = torch.sigmoid(kd_logits[kd_mask])

                kd_targets.extend(targets[kd_mask].cpu().numpy())
                kd_probs.extend(probs.cpu().numpy())

    ki_loss = ki_loss_total / ki_count
    kd_loss = kd_loss_total / kd_count

    ki_metrics = calculate_metrics(ki_targets, ki_probs)
    kd_metrics = calculate_metrics(kd_targets, kd_probs)

    return ki_loss, kd_loss, ki_metrics, kd_metrics


# ============================================================
# 8. Initial Loss
# ============================================================

def calculate_initial_losses(model, train_loader, device, criterion):

    print("\nInitial Loss L_i(0) 계산 중...")

    model.eval()

    ki_loss_total = 0.0
    kd_loss_total = 0.0

    ki_count = 0
    kd_count = 0

    start_time = time.time()

    with torch.no_grad():

        for mol, prot, targets, tasks in train_loader:

            mol = mol.to(device, non_blocking=True)
            prot = prot.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tasks = tasks.to(device, non_blocking=True)

            ki_logits, kd_logits = model(mol, prot)

            ki_mask = (tasks == 0)
            kd_mask = (tasks == 1)

            if ki_mask.any():
                loss = criterion(ki_logits[ki_mask], targets[ki_mask])
                count = ki_mask.sum().item()
                ki_loss_total += loss.item() * count
                ki_count += count

            if kd_mask.any():
                loss = criterion(kd_logits[kd_mask], targets[kd_mask])
                count = kd_mask.sum().item()
                kd_loss_total += loss.item() * count
                kd_count += count

    initial_losses = torch.tensor(
        [ki_loss_total / ki_count, kd_loss_total / kd_count],
        device=device,
        dtype=torch.float32
    )

    elapsed = time.time() - start_time

    print(f"L_i(0) 계산 완료 ({elapsed:.2f} sec)")
    print(f"Ki : {initial_losses[0].item():.6f}")
    print(f"Kd : {initial_losses[1].item():.6f}")

    return initial_losses


# ============================================================
# 9. Checkpoint saving
# ============================================================

def save_best_checkpoint(
    path,
    model_state,
    log_task_weights,
    initial_losses,
    best_epoch,
    best_score,
    best_ki_metrics,
    best_kd_metrics,
    best_ki_val_loss,
    best_kd_val_loss,
    history,
    batch_size,
    alpha
):

    task_weights = (
        2.0 * torch.softmax(log_task_weights.detach().cpu(), dim=0)
    )

    torch.save(
        {
            "model_state_dict": model_state,
            "log_task_weights": log_task_weights.detach().cpu(),
            "task_weights": task_weights,
            "initial_losses": initial_losses.detach().cpu(),

            "threshold": THRESHOLD,
            "batch_size": batch_size,
            "alpha": alpha,
            "max_epochs": MAX_EPOCHS,
            "seed": SEED,

            # best epoch 정보
            "best_epoch": best_epoch,
            "best_score_mean_auprc": best_score,
            "best_ki_val_loss": best_ki_val_loss,
            "best_kd_val_loss": best_kd_val_loss,
            "best_ki_metrics": to_python_floats(best_ki_metrics),
            "best_kd_metrics": to_python_floats(best_kd_metrics),

            # epoch별 학습 기록
            "history": history
        },
        path
    )


# ============================================================
# 10. Main Training
# ============================================================

def train_multitask_gradnorm(
    ki_path,
    kd_path,
    max_epochs=100,
    batch_size=1024,
    alpha=1.5
):

    print("\n" + "=" * 70)
    print("Ki & Kd Multi-Task Training")
    print("In-Memory H5 + GradNorm + Best-Epoch Selection")
    print(f"Threshold : {THRESHOLD}")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_gpu_info()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    ki_dataset = DTIVectorDatasetInMemory(
        ki_path, task_idx=0, threshold=THRESHOLD
    )

    kd_dataset = DTIVectorDatasetInMemory(
        kd_path, task_idx=1, threshold=THRESHOLD
    )

    print("\nDataset")
    print(f"Ki : {len(ki_dataset):,}")
    print(f"Kd : {len(kd_dataset):,}")

    # --------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(SEED)

    ki_train_size = int(0.8 * len(ki_dataset))
    ki_val_size = len(ki_dataset) - ki_train_size

    ki_train, ki_val = random_split(
        ki_dataset, [ki_train_size, ki_val_size], generator=generator
    )

    generator = torch.Generator().manual_seed(SEED)

    kd_train_size = int(0.8 * len(kd_dataset))
    kd_val_size = len(kd_dataset) - kd_train_size

    kd_train, kd_val = random_split(
        kd_dataset, [kd_train_size, kd_val_size], generator=generator
    )

    print("\nSplit")
    print(f"Ki train : {len(ki_train):,}")
    print(f"Ki val   : {len(ki_val):,}")
    print(f"Kd train : {len(kd_train):,}")
    print(f"Kd val   : {len(kd_val):,}")

    # --------------------------------------------------------
    # Combined train dataset
    #
    # ConcatDataset index
    #   Ki: 0 ~ len(ki_train)-1
    #   Kd: len(ki_train) ~ end
    # --------------------------------------------------------

    train_dataset = ConcatDataset([ki_train, kd_train])

    train_ki_indices = list(range(len(ki_train)))

    train_kd_indices = list(
        range(len(ki_train), len(ki_train) + len(kd_train))
    )

    # --------------------------------------------------------
    # Exact ratio sampler
    # --------------------------------------------------------

    batch_sampler = ExactRatioBatchSampler(
        train_ki_indices, train_kd_indices, batch_size=batch_size
    )

    print("\nBatch configuration")
    print(f"Ki / batch : {batch_sampler.num_ki}")
    print(f"Kd / batch : {batch_sampler.num_kd}")
    print(f"Total / batch : {batch_sampler.num_ki + batch_sampler.num_kd}")
    print(f"Batches / epoch : {len(batch_sampler)}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_dataset = ConcatDataset([ki_val, kd_val])

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTask_KiKd_Classifier().to(device)

    total_parameters = sum(p.numel() for p in model.parameters())

    print(f"\nModel parameters : {total_parameters:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer_model = optim.Adam(model.parameters(), lr=MODEL_LR)

    # --------------------------------------------------------
    # GradNorm weights (log weights -> softmax -> positive, sum = 2)
    # --------------------------------------------------------

    log_task_weights = nn.Parameter(torch.zeros(2, device=device))

    optimizer_weights = optim.Adam([log_task_weights], lr=WEIGHT_LR)

    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # LR scheduler (검증 점수가 정체되면 model LR 감소)
    # --------------------------------------------------------

    scheduler = None

    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_model,
            mode="max",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR
        )

    # --------------------------------------------------------
    # Initial losses
    # --------------------------------------------------------

    initial_losses = calculate_initial_losses(
        model, train_loader, device, criterion
    )

    # ========================================================
    # Training
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING START")
    print(f"Max epochs : {max_epochs}")
    print(f"Early stopping patience : {EARLY_STOP_PATIENCE}")
    print(f"Selection metric : mean(Ki AUPRC, Kd AUPRC)")
    print("=" * 70)

    total_start = time.time()

    # Best-model tracking
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    best_log_task_weights = None
    best_ki_metrics = None
    best_kd_metrics = None
    best_ki_val_loss = None
    best_kd_val_loss = None
    epochs_without_improvement = 0
    stopped_early = False

    history = []

    for epoch in range(1, max_epochs + 1):

        epoch_start = time.time()

        model.train()

        total_train_loss = 0.0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for batch_idx, (mol, prot, targets, tasks) in enumerate(train_loader):

            mol = mol.to(device, non_blocking=True)
            prot = prot.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            tasks = tasks.to(device, non_blocking=True)

            ki_mask = (tasks == 0)
            kd_mask = (tasks == 1)

            # =================================================
            # Forward #1 : GradNorm calculation
            # =================================================

            ki_logits, kd_logits = model(mol, prot)

            loss_ki = criterion(ki_logits[ki_mask], targets[ki_mask])
            loss_kd = criterion(kd_logits[kd_mask], targets[kd_mask])

            # Current task weights
            task_weights = 2.0 * torch.softmax(log_task_weights, dim=0)

            # Shared parameter
            shared_layer = model.get_shared_layer()

            # Ki gradient
            grad_ki = torch.autograd.grad(
                task_weights[0] * loss_ki,
                shared_layer,
                retain_graph=True,
                create_graph=True
            )[0]

            # Kd gradient
            grad_kd = torch.autograd.grad(
                task_weights[1] * loss_kd,
                shared_layer,
                retain_graph=True,
                create_graph=True
            )[0]

            # Gradient norms
            G_ki = torch.norm(grad_ki, p=2)
            G_kd = torch.norm(grad_kd, p=2)

            G_avg = (G_ki + G_kd) / 2.0

            # Relative training rate
            loss_ratio_ki = loss_ki.detach() / initial_losses[0]
            loss_ratio_kd = loss_kd.detach() / initial_losses[1]

            mean_loss_ratio = (loss_ratio_ki + loss_ratio_kd) / 2.0

            r_ki = loss_ratio_ki / (mean_loss_ratio + 1e-8)
            r_kd = loss_ratio_kd / (mean_loss_ratio + 1e-8)

            # Target gradient norms
            target_ki = (G_avg * torch.pow(r_ki, alpha)).detach()
            target_kd = (G_avg * torch.pow(r_kd, alpha)).detach()

            # GradNorm loss
            loss_gradnorm = (
                torch.abs(G_ki - target_ki)
                + torch.abs(G_kd - target_kd)
            )

            # Update task weights
            optimizer_weights.zero_grad(set_to_none=True)

            loss_gradnorm.backward()

            optimizer_weights.step()

            # =================================================
            # IMPORTANT
            # task weight update 후 새로운 forward graph 생성
            # =================================================

            ki_logits, kd_logits = model(mol, prot)

            loss_ki = criterion(ki_logits[ki_mask], targets[ki_mask])
            loss_kd = criterion(kd_logits[kd_mask], targets[kd_mask])

            # Updated task weights
            task_weights = 2.0 * torch.softmax(log_task_weights, dim=0)

            # Weighted model loss
            # (task weight 자체는 model optimizer가 업데이트하지 않도록 detach)
            weighted_loss = (
                task_weights[0].detach() * loss_ki
                + task_weights[1].detach() * loss_kd
            )

            # Model update
            optimizer_model.zero_grad(set_to_none=True)

            weighted_loss.backward()

            optimizer_model.step()

            total_train_loss += weighted_loss.item()

            # Progress
            if (batch_idx + 1) % 50 == 0:

                print(
                    f"\r"
                    f"Epoch {epoch:02d}/{max_epochs} "
                    f"| Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss {weighted_loss.item():.4f}",
                    end="",
                    flush=True
                )

        print()

        # ====================================================
        # Validation
        # ====================================================

        (
            ki_val_loss,
            kd_val_loss,
            ki_metrics,
            kd_metrics
        ) = evaluate(model, val_loader, device, criterion)

        score = compute_selection_score(ki_metrics, kd_metrics)

        # ====================================================
        # LR scheduler step
        # ====================================================

        current_lr = optimizer_model.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step(score if np.isfinite(score) else 0.0)

        # ====================================================
        # Best-model check
        # ====================================================

        improved = score > best_score + MIN_DELTA

        if improved:

            best_score = score
            best_epoch = epoch
            best_ki_metrics = dict(ki_metrics)
            best_kd_metrics = dict(kd_metrics)
            best_ki_val_loss = ki_val_loss
            best_kd_val_loss = kd_val_loss
            best_log_task_weights = log_task_weights.detach().clone()

            # GPU 텐서 참조가 아닌 CPU 복사본을 저장
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ====================================================
        # Task weights / history
        # ====================================================

        final_weights = (
            2.0 * torch.softmax(log_task_weights, dim=0)
        ).detach().cpu().numpy()

        avg_train_loss = total_train_loss / len(train_loader)

        history.append({
            "epoch": epoch,
            "lr": float(current_lr),
            "train_loss": float(avg_train_loss),
            "ki_val_loss": float(ki_val_loss),
            "kd_val_loss": float(kd_val_loss),
            "ki_metrics": to_python_floats(ki_metrics),
            "kd_metrics": to_python_floats(kd_metrics),
            "score": float(score),
            "w_ki": float(final_weights[0]),
            "w_kd": float(final_weights[1]),
            "is_best": bool(improved)
        })

        # ====================================================
        # Timing
        # ====================================================

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        # ====================================================
        # GPU memory
        # ====================================================

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_allocated = 0.0
            gpu_reserved = 0.0

        # ====================================================
        # Epoch result
        # ====================================================

        print("\n" + "=" * 70)

        print(f"Epoch {epoch:02d}/{max_epochs} 완료"
              + ("   *** NEW BEST ***" if improved else ""))

        print(f"Epoch time : {epoch_time / 60:.2f} min")
        print(f"Elapsed : {total_elapsed / 3600:.2f} h")
        print(f"Learning rate : {current_lr:.2e}")
        print(f"Train Loss : {avg_train_loss:.6f}")
        print(f"Ki Val Loss : {ki_val_loss:.6f}")
        print(f"Kd Val Loss : {kd_val_loss:.6f}")

        print("\n[GradNorm Weights]")
        print(f"Ki : {final_weights[0]:.4f}")
        print(f"Kd : {final_weights[1]:.4f}")

        print("\n[Ki Metrics]")
        for name in METRIC_NAMES:
            print(f"{name:<10}: {ki_metrics[name]:.4f}")

        print("\n[Kd Metrics]")
        for name in METRIC_NAMES:
            print(f"{name:<10}: {kd_metrics[name]:.4f}")

        print("\n[Model Selection]")
        print(f"Score (mean AUPRC) : {score:.4f}")
        print(f"Best score         : {best_score:.4f} (epoch {best_epoch})")
        print(f"No-improve count   : "
              f"{epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if torch.cuda.is_available():
            print("\n[GPU Memory]")
            print(f"Allocated : {gpu_allocated:.2f} GB")
            print(f"Reserved  : {gpu_reserved:.2f} GB")

        print("=" * 70)

        # ====================================================
        # Best checkpoint 즉시 저장 (학습 도중 중단돼도 보존)
        # ====================================================

        if improved:

            save_best_checkpoint(
                SAVE_PATH,
                best_state,
                best_log_task_weights,
                initial_losses,
                best_epoch,
                best_score,
                best_ki_metrics,
                best_kd_metrics,
                best_ki_val_loss,
                best_kd_val_loss,
                history,
                batch_size,
                alpha
            )

            print(f"[Checkpoint] best model 저장: {SAVE_PATH}")

        # ====================================================
        # Early stopping
        # ====================================================

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:

            print(
                f"\nEarly stopping: {EARLY_STOP_PATIENCE} epoch 동안 "
                f"검증 점수 개선 없음 -> epoch {epoch}에서 종료"
            )

            stopped_early = True
            break

    # ========================================================
    # Finalize: best 모델 복원 + history 갱신하여 최종 저장
    # ========================================================

    if best_state is None:
        raise RuntimeError(
            "유효한 검증 점수를 얻은 epoch이 없습니다 "
            "(AUPRC가 NaN일 수 있음: validation에 positive/negative가 모두 있는지 확인)."
        )

    model.load_state_dict(best_state)

    # 마지막 history(전체 epoch 기록)까지 반영해서 다시 저장
    save_best_checkpoint(
        SAVE_PATH,
        best_state,
        best_log_task_weights,
        initial_losses,
        best_epoch,
        best_score,
        best_ki_metrics,
        best_kd_metrics,
        best_ki_val_loss,
        best_kd_val_loss,
        history,
        batch_size,
        alpha
    )

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)
    print(f"실행 epoch 수      : {len(history)} / {max_epochs}"
          + (" (early stopping)" if stopped_early else " (max epoch 도달)"))
    print(f"Best epoch         : {best_epoch}")
    print(f"Best score (AUPRC) : {best_score:.4f}")
    print(f"Ki  AUROC / AUPRC  : "
          f"{best_ki_metrics['AUROC']:.4f} / {best_ki_metrics['AUPRC']:.4f}")
    print(f"Kd  AUROC / AUPRC  : "
          f"{best_kd_metrics['AUROC']:.4f} / {best_kd_metrics['AUPRC']:.4f}")
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Saved (best model) : {SAVE_PATH}")

    if not stopped_early and best_epoch >= max_epochs - 2:
        print(
            "\n[참고] best epoch이 max epoch 부근입니다. "
            "아직 성능이 오르는 중일 수 있으니 MAX_EPOCHS를 늘려서 재학습을 고려하세요."
        )

    print("=" * 70)

    # ========================================================
    # Cleanup
    # ========================================================

    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del ki_dataset
    del kd_dataset
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    base_dir = (
        "/home/team5/workspace/sj/"
        "homo_protein_seq_fasta"
    )

    ki_path = os.path.join(base_dir, "Ki_vectorDB.h5")
    kd_path = os.path.join(base_dir, "Kd_vectorDB.h5")

    train_multitask_gradnorm(
        ki_path=ki_path,
        kd_path=kd_path,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        alpha=GRADNORM_ALPHA
    )