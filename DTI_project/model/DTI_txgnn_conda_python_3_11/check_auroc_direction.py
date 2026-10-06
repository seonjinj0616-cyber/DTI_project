"""
TxGNN 이 반환하는 prediction/label 딕셔너리를 가지고 sklearn.roc_auc_score 로 직접 AUROC 를 계산해서,
"점수가 높을수록 유력하다"는 가정이 실제 데이터에서 맞는지 확인한다.

(처음엔 TxGNN 내부 함수가 자체 계산하는 raw['result']['AUROC'] 를 쓰려 했으나, 그 함수 안의
 fixed_keys 후보 풀 제한 때문에 양성 표본이 0개로 걸러져 AUROC 를 못 얻는 경우가 있어,
 우리가 이미 안정적으로 받고 있는 prediction/label 원본으로 직접 계산하는 방식으로 바꿨다.)

로직: 같은 질병에 대해 prediction[drug] 과 label[drug](0/1) 을 모으고,
      sklearn.metrics.roc_auc_score(label_array, prediction_array) 를 그대로 호출한다.
      AUROC > 0.5 (특히 0.7 이상) -> "점수가 높을수록 유력하다" 가 맞다는 뜻
      AUROC < 0.5 (특히 0.3 이하) -> 거꾸로(점수가 낮을수록 유력하다) 라는 뜻
      AUROC ~ 0.5               -> 방향과 무관하게 이 표본에 대해 판별력이 거의 없다는 뜻

같은 DTI_txgnn 환경에서 실행하세요:
    python check_auroc_direction.py --data-dir "..." --ckpt-dir "..." --n-sample 30
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import step3_txgnn_score as S


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def per_disease_auroc(raw, disease_key):
    """
    raw['prediction'][disease_key], raw['label'][disease_key] 에서 직접 (pred_array, lab_array) 를 만들어
    sklearn.roc_auc_score 를 계산한다. 양성/음성이 둘 다 있어야 계산 가능.
    반환: (auroc 또는 None, n_pos, n_neg)
    """

    from sklearn.metrics import roc_auc_score

    pred_map = S._get_key(raw.get("prediction"), disease_key)
    lab_map = S._get_key(raw.get("label"), disease_key)

    if not isinstance(pred_map, dict) or not isinstance(lab_map, dict):
        return None, 0, 0

    preds, labs = [], []

    for drug_id, p in pred_map.items():

        pv = _as_float(p)
        lv = _as_float(lab_map.get(drug_id))

        if pv is None or lv is None or lv == -1:
            continue

        preds.append(pv)
        labs.append(1 if lv >= 0.5 else 0)

    n_pos = sum(labs)
    n_neg = len(labs) - n_pos

    if n_pos == 0 or n_neg == 0:
        return None, n_pos, n_neg

    return roc_auc_score(labs, preds), n_pos, n_neg


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--n-hid", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--relation", type=str, default="indication")
    p.add_argument("--n-sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tx_data, model, evaluator = S.load_txgnn(args)

    df = tx_data.df
    pairs = S.sample_known_pairs(df, args.relation, args.n_sample, args.seed)

    if pairs.empty:
        raise SystemExit("표본을 뽑지 못했습니다 (relation='%s' 인 행이 없음)." % args.relation)

    disease_idxs = sorted(pairs["disease_idx"].unique())
    disease_key_of = dict(zip(pairs["disease_idx"], pairs["disease_key"]))

    print("=" * 70)
    print("TxGNN prediction/label 로 직접 계산한 AUROC (relation=%s, 질병 %d개)" % (args.relation, len(disease_idxs)))
    print("=" * 70)

    aurocs = []
    skipped = 0

    for start in range(0, len(disease_idxs), 20):

        chunk = disease_idxs[start:start + 20]

        raw = evaluator.eval_disease_centric(disease_idxs=chunk, relation=args.relation,
                                             save_result=False, return_raw=True)

        for d in chunk:

            key = disease_key_of.get(d, str(d))
            auroc, n_pos, n_neg = per_disease_auroc(raw, key)

            if auroc is None:
                print("  질병 %s -> AUROC 계산 불가 (양성 %d / 음성 %d)" % (key, n_pos, n_neg))
                skipped += 1
            else:
                aurocs.append(auroc)
                print("  질병 %s -> AUROC %.4f (양성 %d / 음성 %d)" % (key, auroc, n_pos, n_neg))

    if not aurocs:
        raise SystemExit("AUROC 를 하나도 못 얻었습니다 (모든 질병에서 양성 또는 음성 표본이 0개).")

    import statistics

    mean_auroc = statistics.mean(aurocs)
    median_auroc = statistics.median(aurocs)

    print("\n" + "-" * 70)
    print("AUROC 평균 %.4f / 중앙값 %.4f  (계산됨 %d개, 계산 불가 %d개, 0.5 = 무작위 수준)"
          % (mean_auroc, median_auroc, len(aurocs), skipped))

    if mean_auroc > 0.6:
        print("-> 0.5 보다 뚜렷이 높음: '점수가 높을수록 유력하다' 는 가정이 맞습니다.")
        print("   (이전에 뒤집은 방향은 잘못된 것이므로 원래대로 되돌려야 합니다.)")
    elif mean_auroc < 0.4:
        print("-> 0.5 보다 뚜렷이 낮음: '점수가 높을수록 유력하다' 는 가정이 이 데이터에서는 거꾸로입니다.")
        print("   (뒤집은 방향이 맞았다는 재확인입니다.)")
    else:
        print("-> 0.5 근처: 방향과 무관하게 이 표본에 대해서는 모델이 뚜렷하게 판별하지 못하는 것으로 보입니다.")


if __name__ == "__main__":
    main()
