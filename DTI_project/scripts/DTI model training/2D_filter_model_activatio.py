import torch


def find_target_protein_pipeline(smiles_input, step1_top100_proteins):
    """
    smiles_input: 탐색하고자 하는 신약 화합물 1개
    step1_top100_proteins: Step 1(1D 대조학습)을 통과한 Top 100 단백질 리스트 [ (id, seq), ... ]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. 훈련된 3개의 2D 모델 불러오기
    # ---------------------------------------------------------
    model_ki = DTICrossAttentionModel(task_type="classification").to(device)
    model_ki.load_state_dict(torch.load("step2_ki_classifier.pt"))
    model_ki.eval()

    model_kd = DTICrossAttentionModel(task_type="classification").to(device)
    model_kd.load_state_dict(torch.load("step2_kd_classifier.pt"))
    model_kd.eval()

    model_ic50 = DTICrossAttentionModel(task_type="regression").to(device)
    model_ic50.load_state_dict(torch.load("step2_ic50_regressor.pt"))
    model_ic50.eval()

    # (이하 과정은 편의상 단백질들을 2D 텐서로 변환하는 함수가 있다고 가정합니다)
    # def prepare_2d_tensor(smiles, protein_seq_list): ... return dataloader

    # ---------------------------------------------------------
    # [Step 2-A] Ki 기반 1차 정밀 선별 (100개 -> 30개)
    # ---------------------------------------------------------
    ki_loader = prepare_2d_tensor(smiles_input, step1_top100_proteins)

    ki_passed_proteins = []
    with torch.no_grad():
        for prot, batch in zip(step1_top100_proteins, ki_loader):
            outputs = model_ki(batch["mol_ids"], batch["mol_mask"], batch["prot_ids"], batch["prot_mask"])
            prob = torch.sigmoid(outputs).item()  # Logit을 0~1 확률로 변환

            # 결합 확률이 50% (혹은 지정한 임계값) 이상인 단백질만 통과
            if prob > 0.5:
                ki_passed_proteins.append(prot)

    print(f"Ki 필터 통과: {len(ki_passed_proteins)}개 생존")

    # ---------------------------------------------------------
    # [Step 2-B] Kd 기반 2차 정밀 선별 (30개 -> 10개)
    # ---------------------------------------------------------
    kd_loader = prepare_2d_tensor(smiles_input, ki_passed_proteins)

    kd_passed_proteins = []
    with torch.no_grad():
        for prot, batch in zip(ki_passed_proteins, kd_loader):
            outputs = model_kd(batch["mol_ids"], batch["mol_mask"], batch["prot_ids"], batch["prot_mask"])
            prob = torch.sigmoid(outputs).item()

            # Kd 기준으로도 결합 확률이 높은 단백질만 최종 선별
            if prob > 0.5:
                kd_passed_proteins.append(prot)

    print(f"Kd 필터 통과: {len(kd_passed_proteins)}개 생존")

    # ---------------------------------------------------------
    # [Step 2-C] IC50 기반 최종 수치 도출 (생존한 10개에 대해)
    # ---------------------------------------------------------
    ic50_loader = prepare_2d_tensor(smiles_input, kd_passed_proteins)

    final_results = []
    with torch.no_grad():
        for prot, batch in zip(kd_passed_proteins, ic50_loader):
            outputs = model_ic50(batch["mol_ids"], batch["mol_mask"], batch["prot_ids"], batch["prot_mask"])

            # Regression이므로 Sigmoid 없이 바로 수치(pIC50) 사용
            pic50_value = outputs.item()
            final_results.append({
                "protein_id": prot["id"],
                "pIC50": pic50_value
            })

    # pIC50 수치가 높은 순(결합력이 강한 순)으로 내림차순 정렬
    final_results = sorted(final_results, key=lambda x: x["pIC50"], reverse=True)

    return final_results