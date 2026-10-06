"""
Top100 cutoff 와 무관하게: 특정 SMILES 화합물에 대해 "이미 알려진 표적"으로 지정한 UniProt
accession 들이 Step1(Ki/Kd 분류 모델)에서 실제로 몇 등/몇 %의 확률을 받는지 직접 확인한다.

Top100 은 (1) Ki 확률 상위 1000행 풀 안에 들고, (2) 그 풀 안에서 Kd 확률도 높아야만 뽑힌다.
이 스크립트는 그 두 단계 cutoff 를 완전히 무시하고, 전체 113,xxx개 단백질 중 정확히 몇 등인지를
보여줘서 "그냥 풀 기준에 살짝 못 미친 것"인지 "모델이 애초에 아주 낮게 보고 있는 것"인지 구분한다.

protein_Top100_classification.py 와 같은 폴더(model\\)에 두고 실행하세요:
    python diagnose_known_targets.py --smiles "<SMILES>"
    python diagnose_known_targets.py --smiles "<SMILES>" --targets P00519 P10721 P16234 P09619 Q08345
"""

import argparse

import numpy as np
import pandas as pd
import torch

import protein_Top100_classification as step1

# [진단용] CPU 멀티스레드 부동소수점 비결정성이 원인인지 확인하기 위해 스레드를 1개로 고정
import torch as _torch
_torch.set_num_threads(1)
print(f"[진단] 스레드를 1개로 고정함 (재현성 테스트)")


def canonical_accession(acc):
    """이소폼 표기(-N)를 뗀 UniProt 기본 accession (step2_site_compare.py 와 같은 규칙)."""
    return str(acc).split("-")[0]


def rank_desc(values):
    """값이 클수록 1등이 되는 순위 (동점은 같은 등수, method='min')."""
    return pd.Series(-np.asarray(values, dtype=np.float64)).rank(method="min").astype(int).to_numpy()


def diagnose_targets(protein_ids, protein_meta, ki_probs, kd_probs, targets,
                      top_ki=None, top_kd_threshold=None, log=print):
    """
    targets(UniProt accession 목록) 각각에 대해:
      - 벡터 DB(원본 CSV)에 그 accession 이 아예 있는지
      - 있다면 그 accession 의 모든 서열 변이체 중 Ki 확률이 가장 높은 행의 ki_prob/kd_prob 와
        전체 단백질 대비 순위
    를 출력한다. 반환: 요약 DataFrame (target, found, best_id, ki_prob, ki_rank, kd_prob, kd_rank, n_variants)
    """

    n_proteins = len(protein_ids)
    canon = np.array([canonical_accession(u) for u in protein_ids])

    ki_rank = rank_desc(ki_probs)
    kd_rank = rank_desc(kd_probs)

    log(f"\n전체 단백질 수: {n_proteins:,}")

    rows = []

    for target in targets:

        mask = canon == target

        if not mask.any():
            log(f"\n[{target}] 벡터 DB(원본 CSV)에 이 UniProt ID 자체가 없습니다.")
            rows.append({"target": target, "found": False, "best_id": None, "n_variants": 0,
                        "ki_prob": None, "ki_rank": None, "kd_prob": None, "kd_rank": None})
            continue

        idxs = np.where(mask)[0]

        sub = protein_meta.iloc[idxs].copy()
        sub["ki_prob"] = ki_probs[idxs]
        sub["kd_prob"] = kd_probs[idxs]
        sub["ki_rank"] = ki_rank[idxs]
        sub["kd_rank"] = kd_rank[idxs]
        sub = sub.sort_values("ki_rank")

        best = sub.iloc[0]

        log(f"\n[{target}] 서열 변이체 {len(sub)}개 존재. 최고 Ki 확률 행:")
        log(f"    id={best['id']}  type={best['type']}  length={best['length']}")
        log(f"    ki_prob={best['ki_prob']:.4f}  (전체 {n_proteins:,}개 중 {best['ki_rank']:,}등)")
        log(f"    kd_prob={best['kd_prob']:.4f}  (전체 {n_proteins:,}개 중 {best['kd_rank']:,}등)")

        if top_ki is not None:
            verdict = "Ki 상위 %d 풀 안에 듦" % top_ki if best["ki_rank"] <= top_ki else \
                "Ki 상위 %d 풀 밖 -> 지금 설정으론 Top100 에 절대 들어올 수 없음" % top_ki
            log("    -> " + verdict)

        if len(sub) > 1:
            log(f"    (그 외 변이체 {len(sub) - 1}개: ki_prob 범위 "
                f"{sub['ki_prob'].min():.4f} ~ {sub['ki_prob'].max():.4f})")

        rows.append({"target": target, "found": True, "best_id": best["id"], "n_variants": len(sub),
                    "ki_prob": float(best["ki_prob"]), "ki_rank": int(best["ki_rank"]),
                    "kd_prob": float(best["kd_prob"]), "kd_rank": int(best["kd_rank"])})

    return pd.DataFrame(rows)


def compute_wildtype_only_ranks(protein_meta, ki_probs, kd_probs):
    """
    "애초에 야생형만 스크리닝했다면" 을 가정한 순위를 계산한다. 변이체는 pool 에서
    완전히 제외하고, 야생형끼리만 비교해서 다시 순위를 매긴다.
    반환: (wt_mask, ki_rank_wt, kd_rank_wt, n_wt)
      ki_rank_wt / kd_rank_wt : 전체와 같은 길이의 배열. 야생형이 아닌 위치는 NaN.
    """

    wt_mask = (protein_meta["type"].to_numpy() == "wildtype")
    n_wt = int(wt_mask.sum())

    ki_rank_wt = np.full(len(wt_mask), np.nan)
    kd_rank_wt = np.full(len(wt_mask), np.nan)

    if n_wt > 0:
        ki_rank_wt[wt_mask] = rank_desc(ki_probs[wt_mask])
        kd_rank_wt[wt_mask] = rank_desc(kd_probs[wt_mask])

    return wt_mask, ki_rank_wt, kd_rank_wt, n_wt


def diagnose_targets_wildtype_only(protein_ids, protein_meta, ki_probs, kd_probs, targets, log=print):
    """
    targets 각각에 대해 "전체(변이체 포함) 중 몇 등" vs "야생형만 남겼을 때 몇 등"을 나란히 보여준다.
    """

    wt_mask, ki_rank_wt, kd_rank_wt, n_wt = compute_wildtype_only_ranks(protein_meta, ki_probs, kd_probs)
    n_all = len(protein_ids)

    log(f"\n{'=' * 90}")
    log(f"야생형만 남긴다면? (전체 {n_all:,}개 중 야생형 {n_wt:,}개)")
    log(f"{'=' * 90}")

    canon = np.array([canonical_accession(u) for u in protein_ids])
    rows = []

    for target in targets:

        wt_idx = np.where((canon == target) & wt_mask)[0]

        if len(wt_idx) == 0:
            log(f"\n[{target}] 이 단백질의 야생형 서열이 벡터 DB에 없습니다 "
                "(변이체만 있거나, 단백질 자체가 없음) -> 비교 불가")
            rows.append({"target": target, "has_wildtype": False})
            continue

        idx = int(wt_idx[0])  # 보통 한 accession 당 야생형은 1개

        # "전체(변이체 포함)" 기준 이 야생형 행 자체의 순위 (diagnose_targets 의 "최고 확률 행"과는
        # 다를 수 있음 -- 거긴 변이체가 이겼을 수도 있는 자리라서, 여기서는 "야생형 그 자체"만 본다)
        ki_rank_all = int(1 + (ki_probs > ki_probs[idx]).sum())
        kd_rank_all = int(1 + (kd_probs > kd_probs[idx]).sum())

        ki_rank_wt_val = int(ki_rank_wt[idx])
        kd_rank_wt_val = int(kd_rank_wt[idx])

        log(f"\n[{target}] 야생형 (id={protein_meta.iloc[idx]['id']})")
        log(f"    Ki: 전체 {n_all:,}개 중 {ki_rank_all:,}등  ->  야생형만 {n_wt:,}개 중 {ki_rank_wt_val:,}등")
        log(f"    Kd: 전체 {n_all:,}개 중 {kd_rank_all:,}등  ->  야생형만 {n_wt:,}개 중 {kd_rank_wt_val:,}등")

        ki_pct_all = 100 * ki_rank_all / n_all
        ki_pct_wt = 100 * ki_rank_wt_val / n_wt
        moved = "개선" if ki_pct_wt < ki_pct_all else ("악화" if ki_pct_wt > ki_pct_all else "동일")
        log(f"    Ki 백분위: 전체 상위 {ki_pct_all:.1f}%  ->  야생형만 상위 {ki_pct_wt:.1f}%  ({moved})")

        rows.append({"target": target, "has_wildtype": True,
                    "ki_rank_all": ki_rank_all, "ki_rank_wt": ki_rank_wt_val,
                    "kd_rank_all": kd_rank_all, "kd_rank_wt": kd_rank_wt_val,
                    "ki_pct_all": ki_pct_all, "ki_pct_wt": ki_pct_wt})

    valid = [r for r in rows if r.get("has_wildtype")]

    if valid:
        mean_pct_all = np.mean([r["ki_pct_all"] for r in valid])
        mean_pct_wt = np.mean([r["ki_pct_wt"] for r in valid])
        log(f"\n{len(valid)}개 평균 Ki 백분위: 전체 {mean_pct_all:.1f}%  ->  야생형만 {mean_pct_wt:.1f}%")

    return pd.DataFrame(rows)


def compute_all_probs(smiles):
    """protein_Top100_classification.py 와 완전히 같은 방식으로 SMILES -> 전체 단백질 Ki/Kd 확률."""

    mol_tokenizer = step1.AutoTokenizer.from_pretrained(step1.MOL_MODEL_NAME, trust_remote_code=True)
    mol_model = step1.AutoModel.from_pretrained(step1.MOL_MODEL_NAME, trust_remote_code=True)
    mol_model.eval()

    drug_vector = step1.compute_drug_vector(smiles, mol_tokenizer, mol_model)
    del mol_model

    protein_vectors_np, protein_ids, protein_meta = step1.load_protein_database(step1.PROTEIN_VECTOR_H5)
    n_proteins = protein_vectors_np.shape[0]
    protein_vectors = torch.from_numpy(protein_vectors_np)

    model = step1.MultiTask_KiKd_Classifier(
        mol_dim=step1.MOL_DIM, prot_dim=step1.PROT_DIM, hidden_dim=step1.HIDDEN_DIM
    )
    ckpt = torch.load(step1.KIKD_MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ki_probs = np.zeros(n_proteins, dtype=np.float32)
    kd_probs = np.zeros(n_proteins, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_proteins, step1.INFERENCE_BATCH_SIZE):
            end = min(start + step1.INFERENCE_BATCH_SIZE, n_proteins)
            prot_batch = protein_vectors[start:end]
            mol_batch = drug_vector.unsqueeze(0).expand(end - start, -1)
            ki_logits, kd_logits = model(mol_batch, prot_batch)
            ki_probs[start:end] = torch.sigmoid(ki_logits).numpy()
            kd_probs[start:end] = torch.sigmoid(kd_logits).numpy()

    return protein_ids, protein_meta, ki_probs, kd_probs


def report_type_bias(protein_meta, ki_probs, kd_probs, log=print):
    """
    전체 단백질을 type(wildtype/mutant 등)별로 나눠 평균/중앙값 Ki, Kd 확률을 비교한다.
    특정 단백질 몇 개에서 "돌연변이가 야생형보다 항상 위"로 나온 것이,
    그 단백질들만의 우연인지 전체적인 편향인지 구분하기 위한 것이다.
    """

    df = pd.DataFrame({"type": protein_meta["type"].to_numpy(),
                       "ki_prob": ki_probs, "kd_prob": kd_probs})

    log("\n" + "=" * 90)
    log("전체 단백질: type(야생형/변이체)별 Ki, Kd 확률 비교")
    log("=" * 90)

    summary = df.groupby("type").agg(
        n=("ki_prob", "size"),
        ki_mean=("ki_prob", "mean"), ki_median=("ki_prob", "median"),
        kd_mean=("kd_prob", "mean"), kd_median=("kd_prob", "median"),
    ).reset_index()

    log(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wt = summary[summary["type"] == "wildtype"]
    mut = summary[summary["type"] == "mutant"]

    if len(wt) and len(mut):

        wt_ki, mut_ki = float(wt["ki_mean"].iloc[0]), float(mut["ki_mean"].iloc[0])

        log("")

        if mut_ki > wt_ki * 1.15:
            log(f"-> [주의] 전체적으로도 변이체(mutant)의 평균 Ki 확률({mut_ki:.4f})이 "
                f"야생형(wildtype, {wt_ki:.4f})보다 뚜렷이 높습니다.")
            log("   특정 단백질 5개만의 우연이 아니라, 모델이 'type' 자체와 관련된 편향된 신호를 "
                "학습했을 가능성이 있습니다 (학습 데이터의 야생형/변이체 비율·라벨링을 재검토 필요).")
        elif wt_ki > mut_ki * 1.15:
            log(f"-> 전체적으로는 야생형 평균 Ki 확률({wt_ki:.4f})이 변이체({mut_ki:.4f})보다 높습니다.")
            log("   즉 앞서 본 5개 단백질에서 변이체가 위로 온 것은 전체 경향과 반대이므로, "
                "그 5개 단백질에 국한된 우연/개별 현상에 가깝습니다.")
        else:
            log(f"-> 전체적으로는 야생형({wt_ki:.4f})과 변이체({mut_ki:.4f})의 평균 차이가 크지 않습니다.")
            log("   앞서 본 5개 단백질의 패턴은 이 정도 표본(5개)에서 우연히 나타났을 수 있습니다.")

    return summary


def main():

    p = argparse.ArgumentParser(description="Top100 cutoff 와 무관하게 특정 단백질의 실제 순위/확률을 확인")
    p.add_argument("--smiles", required=True)
    p.add_argument("--targets", nargs="+", default=["P00519", "P10721", "P16234", "P09619", "Q08345"],
                   help="확인할 UniProt accession 목록 (기본: imatinib 의 잘 알려진 표적 5개: "
                        "ABL1, KIT, PDGFRA, PDGFRB, DDR1)")
    args = p.parse_args()

    print("=" * 90)
    print("알려진 표적 진단: Top100 cutoff 와 무관하게 전체 순위/확률을 직접 확인")
    print("=" * 90)
    print(f"SMILES  : {args.smiles}")
    print(f"targets : {args.targets}")

    protein_ids, protein_meta, ki_probs, kd_probs = compute_all_probs(args.smiles)

    summary = diagnose_targets(protein_ids, protein_meta, ki_probs, kd_probs, args.targets,
                               top_ki=step1.TOP_KI)

    print("\n" + "=" * 90)
    print("요약")
    print("=" * 90)
    print(summary.to_string(index=False))

    report_type_bias(protein_meta, ki_probs, kd_probs)

    diagnose_targets_wildtype_only(protein_ids, protein_meta, ki_probs, kd_probs, args.targets)


if __name__ == "__main__":
    main()
