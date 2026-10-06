"""
MoLFormer 인코딩 자체가 결정적인지 확인하는 진단 스크립트.

같은 모델을 딱 한 번만 불러온 뒤, 같은 SMILES 를 그 안에서 여러 번 인코딩해서 벡터가
매번 똑같이 나오는지 본다. 이러면 "모델을 다시 불러올 때마다 달라지는가" 와
"한 번 불러온 모델도 호출할 때마다 다르게 계산하는가"를 구분할 수 있다.

    python test_molformer_determinism.py --smiles "<SMILES>"
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModel

MOL_MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"


def compute_drug_vector(smiles, mol_tokenizer, mol_model):
    """protein_Top100_classification.py 와 완전히 동일한 방식."""

    encoded = mol_tokenizer([smiles], padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = mol_model(**encoded)
        hidden = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
    masked_hidden = hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts

    return pooled.squeeze(0)


def main():

    p = argparse.ArgumentParser()
    p.add_argument("--smiles", required=True)
    p.add_argument("--n-repeat", type=int, default=5, help="같은 프로세스 안에서 반복 인코딩할 횟수")
    args = p.parse_args()

    print("=" * 90)
    print("MoLFormer 인코딩 결정성(determinism) 진단")
    print("=" * 90)

    torch.set_num_threads(1)

    print("모델을 한 번만 로드합니다...")
    mol_tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
    mol_model = AutoModel.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)

    print(f"model.training = {mol_model.training}  (True 면 아직 eval 모드가 아님)")
    mol_model.eval()
    print(f"eval() 호출 후 model.training = {mol_model.training}")

    # 모델 안에 dropout 계열 서브모듈이 있는지, 있다면 eval() 이후에도 self.training 이 True 인지 확인
    dropout_modules = [(name, m) for name, m in mol_model.named_modules()
                       if "drop" in type(m).__name__.lower()]

    print(f"\ndropout 계열 서브모듈 {len(dropout_modules)}개 발견"
          + (":" if dropout_modules else " (없음)"))

    still_training = [name for name, m in dropout_modules if getattr(m, "training", False)]

    for name, m in dropout_modules[:10]:
        print(f"  {name:<50} {type(m).__name__:<20} training={getattr(m, 'training', '?')}")

    if still_training:
        print(f"\n[중요] eval() 을 호출했는데도 training=True 로 남아있는 dropout 모듈 "
              f"{len(still_training)}개 발견 -> 이게 원인일 가능성이 매우 높습니다:")
        for name in still_training[:10]:
            print(f"    {name}")
    elif dropout_modules:
        print("\n모든 dropout 모듈이 eval() 을 정상적으로 따르고 있습니다 (training=False). "
              "다른 원인을 찾아야 합니다.")

    # dropout 이 아닌 무작위 요소(선형 어텐션의 무작위 투영 특징 등)가 있는지 이름으로 훑어본다
    suspicious_keywords = ("random", "projection", "feature", "redraw", "favor", "performer", "orf")
    suspicious = []

    for name, m in mol_model.named_modules():
        if any(k in name.lower() or k in type(m).__name__.lower() for k in suspicious_keywords):
            suspicious.append((name, type(m).__name__))

    for name, buf in mol_model.named_buffers():
        if any(k in name.lower() for k in suspicious_keywords):
            suspicious.append((name, "buffer"))

    if suspicious:
        print(f"\n[참고] 이름에 '무작위 특징/투영' 계열 키워드가 들어간 항목 {len(suspicious)}개 발견 "
              "(선형 어텐션의 무작위 투영 특징일 가능성):")
        for name, kind in suspicious[:15]:
            print(f"    {name}  ({kind})")
    else:
        print("\n[참고] 이름으로는 '무작위 투영' 계열 요소를 찾지 못했습니다 "
              "(그래도 존재할 수 있음 - 이름이 다를 수 있음).")

    print(f"\n같은 SMILES 를 이 모델 하나로 {args.n_repeat}번 반복 인코딩합니다...")
    print(f"SMILES: {args.smiles}\n")

    vectors = []

    for i in range(args.n_repeat):
        v = compute_drug_vector(args.smiles, mol_tokenizer, mol_model)
        vectors.append(v)
        print(f"  [{i + 1}] 앞 5개 값: {v[:5].tolist()}")

    ref = vectors[0]
    max_diffs = [float((v - ref).abs().max()) for v in vectors[1:]]

    print(f"\n1번째 결과 대비 최대 절댓값 차이: {max_diffs}")

    if all(d < 1e-6 for d in max_diffs):
        print("\n-> 완전히 결정적입니다 (차이가 소수점 6자리보다 작음). "
              "MoLFormer 인코딩 자체는 문제가 없어 보입니다. 다른 단계(Ki/Kd 분류기, "
              "혹은 매번 모델을 새로 불러오는 과정)를 봐야 합니다.")
    elif all(d < 1e-3 for d in max_diffs):
        print("\n-> 아주 작은 차이만 있습니다 (부동소수점 수준). 이 정도는 이후 층을 거치며 "
              "커질 수 있으니, 원인은 맞지만 어디서 증폭되는지 더 봐야 합니다.")
    else:
        print("\n-> [중요] 같은 모델, 같은 입력인데도 벡터 자체가 눈에 띄게 다릅니다. "
              "일반 dropout 이 아닌 다른 무작위 요소(예: 선형 어텐션의 무작위 투영 특징)가 "
              "매 forward 마다 다시 뽑히고 있을 가능성이 높습니다.")

        print("\n" + "=" * 90)
        print("해결책 검증: 매번 계산 직전에 시드를 고정하면 결과가 똑같아지는지 확인")
        print("=" * 90)

        seeded_vectors = []

        for i in range(args.n_repeat):
            torch.manual_seed(42)
            v = compute_drug_vector(args.smiles, mol_tokenizer, mol_model)
            seeded_vectors.append(v)
            print(f"  [시드 고정 {i + 1}] 앞 5개 값: {v[:5].tolist()}")

        seeded_diffs = [float((v - seeded_vectors[0]).abs().max()) for v in seeded_vectors[1:]]
        print(f"\n시드 고정 시 1번째 결과 대비 최대 절댓값 차이: {seeded_diffs}")

        if all(d < 1e-6 for d in seeded_diffs):
            print("\n-> [해결] torch.manual_seed() 로 매번 시드를 고정하니 완전히 똑같은 결과가 나옵니다.")
            print("   compute_drug_vector() 호출 직전에 항상 같은 시드를 고정하면 재현성 문제가 해결됩니다.")
        else:
            print("\n-> 시드를 고정해도 여전히 다릅니다. 원인이 파이썬 난수/torch 시드로 제어되지 않는 "
                  "다른 곳(예: C 확장, 비결정적 CPU 커널)에 있을 수 있어 추가 조사가 필요합니다.")


if __name__ == "__main__":
    main()