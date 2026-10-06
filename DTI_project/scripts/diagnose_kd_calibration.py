"""
Kd(그리고 Ki) 확률이 실제 "검증 데이터(val set)"에서도 1 쪽으로 쏠려 있는지 확인한다.

지금까지 본 "Kd 확률이 1에 가깝다"는 증상은 전부 imatinib 하나로, 학습 때 전혀
안 봤을 수도 있는 단백질 113,853개에 돌린 결과였다. 이게:
  (a) 모델 자체가 학습 중에 이미 이렇게 쏠리도록 학습됐다 (calibration 문제), 아니면
  (b) 학습 데이터 안(val set)에서는 정상인데, 학습 때 못 본 새로운 입력에서만
      극단적으로 나온다 (일반화/커버리지 문제)
중 어느 쪽인지 구분한다.

체크포인트에 저장된 threshold/seed/batch_size 를 그대로 읽어서, 학습 스크립트와
똑같은 방식(random_split, 같은 seed)으로 Kd/Ki 의 val set 을 재현한다.

    python diagnose_kd_calibration.py --kd-path Kd_vectorDB.h5 --ki-path Ki_vectorDB.h5 --ckpt step1_multitask_kikd_gradnorm_thr7.pt
"""

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import roc_auc_score, average_precision_score


class MultiTask_KiKd_Classifier(nn.Module):
    """train_kikd_gradnorm_best.py 와 완전히 동일한 구조여야 state_dict 가 맞는다."""

    def __init__(self, mol_dim=768, prot_dim=1280, hidden_dim=512):
        super().__init__()
        input_dim = mol_dim + prot_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.ki_head = nn.Linear(256, 1)
        self.kd_head = nn.Linear(256, 1)

    def forward(self, mol_vec, prot_vec):
        combined = torch.cat([mol_vec, prot_vec], dim=1)
        features = self.shared_body(combined)
        return self.ki_head(features).squeeze(-1), self.kd_head(features).squeeze(-1)


class DTIVectorDatasetInMemory(Dataset):
    """train_kikd_gradnorm_best.py 와 동일 (H5 -> (drug_vector, protein_vector, target))."""

    def __init__(self, h5_path, threshold):
        with h5py.File(h5_path, "r") as f:
            self.drug_vectors = torch.tensor(f["interactions"]["drug_vector"][:], dtype=torch.float32)
            self.prot_vectors = torch.tensor(f["interactions"]["protein_vector"][:], dtype=torch.float32)
            paffinity = f["interactions"]["paffinity"][:]
        self.targets = torch.tensor((paffinity >= threshold).astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.drug_vectors[idx], self.prot_vectors[idx], self.targets[idx]


def build_val_split(h5_path, threshold, seed):
    """학습 스크립트와 완전히 같은 방식(0.8/0.2, 같은 seed)으로 val subset 을 재현한다."""

    dataset = DTIVectorDatasetInMemory(h5_path, threshold=threshold)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    _, val = random_split(dataset, [train_size, val_size], generator=generator)

    return val


def collect_probs(model, val_subset, head, batch_size=1024, device="cpu"):
    """head: 'ki' 또는 'kd'. 반환: (probs, targets) numpy 배열."""

    loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    probs, targets = [], []

    model.eval()

    with torch.no_grad():
        for mol, prot, y in loader:
            mol, prot = mol.to(device), prot.to(device)
            ki_logits, kd_logits = model(mol, prot)
            logits = ki_logits if head == "ki" else kd_logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())

    return np.concatenate(probs), np.concatenate(targets)


def report_distribution(name, probs, targets, log=print):

    log(f"\n{'=' * 90}\n{name} 검증(val) 데이터 확률 분포  (표본 {len(probs):,}개)\n{'=' * 90}")

    pos = probs[targets == 1]
    neg = probs[targets == 0]

    log(f"진짜 양성(label=1) {len(pos):,}개: 평균 {pos.mean():.4f} / 중앙값 {np.median(pos):.4f}")
    log(f"진짜 음성(label=0) {len(neg):,}개: 평균 {neg.mean():.4f} / 중앙값 {np.median(neg):.4f}")

    bins = np.arange(0, 1.01, 0.1)
    pos_hist, _ = np.histogram(pos, bins=bins)
    neg_hist, _ = np.histogram(neg, bins=bins)

    log("\n구간          진짜양성    진짜음성")
    for i in range(len(bins) - 1):
        log(f"{bins[i]:.1f}~{bins[i + 1]:.1f}      {pos_hist[i]:>8,}    {neg_hist[i]:>8,}")

    try:
        auroc = roc_auc_score(targets, probs)
        auprc = average_precision_score(targets, probs)
        log(f"\nAUROC {auroc:.4f} / AUPRC {auprc:.4f}  (체크포인트에 저장된 값과 비슷해야 정상)")
    except ValueError:
        pass

    frac_above_half = float((probs >= 0.5).mean())
    neg_above_half = float((neg >= 0.5).mean()) if len(neg) else float("nan")

    log(f"\n전체 중 확률 >= 0.5 인 비율: {frac_above_half * 100:.1f}%")
    log(f"진짜 음성인데도 확률 >= 0.5 로 나온 비율: {neg_above_half * 100:.1f}%  "
        "(이 값이 크면: 임계값 0.5 가 사실상 필터링을 못 하고 있다는 뜻)")

    if neg_above_half > 0.3:
        log("\n-> [경고] 진짜 음성의 30% 이상이 0.5 를 넘습니다. 검증 데이터 자체에서도 확률이 "
            "높은 쪽으로 쏠려 있다는 뜻이라, 학습 중 calibration 문제일 가능성이 높습니다.")
    elif frac_above_half > 0.9 and len(pos) / max(len(pos) + len(neg), 1) < 0.7:
        log("\n-> [주의] 양성 비율(라벨)보다 훨씬 많은 비율이 0.5 를 넘습니다. 확률이 전반적으로 "
            "쏠려 있을 수 있습니다.")
    else:
        log("\n-> 검증 데이터에서는 확률이 라벨과 비교적 잘 맞습니다. 그렇다면 imatinib 스크리닝에서 "
            "본 쏠림은 '학습 데이터 밖(새로운 화합물/단백질)에 대한 일반화 문제'에 더 가깝습니다.")


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--kd-path", required=True)
    p.add_argument("--ki-path", default=None, help="같이 확인하고 싶으면 Ki_vectorDB.h5 경로도 지정")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=None, help="생략하면 체크포인트에 저장된 값 사용")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    threshold = ckpt.get("threshold", 7.0)
    seed = ckpt.get("seed", 42)
    batch_size = args.batch_size or ckpt.get("batch_size", 1024)

    print(f"체크포인트에서 읽은 설정: threshold={threshold}, seed={seed}, batch_size={batch_size}")
    print(f"체크포인트 저장 당시 Kd 지표: {ckpt.get('best_kd_metrics')}")
    print(f"체크포인트 저장 당시 Ki 지표: {ckpt.get('best_ki_metrics')}")

    model = MultiTask_KiKd_Classifier().to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    kd_val = build_val_split(args.kd_path, threshold, seed)
    kd_probs, kd_targets = collect_probs(model, kd_val, "kd", batch_size, device)
    report_distribution("Kd", kd_probs, kd_targets)

    if args.ki_path:
        ki_val = build_val_split(args.ki_path, threshold, seed)
        ki_probs, ki_targets = collect_probs(model, ki_val, "ki", batch_size, device)
        report_distribution("Ki", ki_probs, ki_targets)


if __name__ == "__main__":
    main()
