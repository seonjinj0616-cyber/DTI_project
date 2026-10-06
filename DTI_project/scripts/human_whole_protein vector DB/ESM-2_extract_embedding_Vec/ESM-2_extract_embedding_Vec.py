import os
import torch
import h5py
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
from torch.utils.data import Dataset, DataLoader


# ==========================================
# 1. 커스텀 데이터셋 클래스 정의
# ==========================================
class ProteinDataset(Dataset):
    def __init__(self, df):
        self.ids = df['id'].values
        self.seqs = df['sequence'].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.seqs[idx]


# ==========================================
# 2. 메인 임베딩 추출 함수
# ==========================================
def generate_embeddings(csv_path, output_h5, model_name="facebook/esm2_t30_150M_UR50D", batch_size=16):
    print("==========================================")
    print("🚀 [ESM-2 임베딩 추출 파이프라인 시작]")
    print(f"• 모델: {model_name}")
    print("==========================================")

    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 사용 디바이스: {device} (GPU 가속: {'활성화' if device.type == 'cuda' else '비활성화'})")

    # 1. 모델 및 토크나이저 로드
    print("⏳ 모델을 다운로드/로드 중입니다. (최초 실행 시 시간 소요)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()  # 추론 모드로 전환 (메모리 절약)

    # 2. 데이터 로드 및 이어하기(Resume) 처리
    df = pd.read_csv(csv_path)
    total_seqs = len(df)

    processed_ids = set()
    if os.path.exists(output_h5):
        with h5py.File(output_h5, 'r') as f:
            processed_ids = set(f.keys())
        print(f"🔄 기존 저장된 임베딩 발견: {len(processed_ids):,}개 (건너뛰고 이어서 진행합니다)")

    # 아직 처리되지 않은 데이터만 필터링
    df_to_process = df[~df['id'].isin(processed_ids)]

    if len(df_to_process) == 0:
        print("✅ 모든 단백질의 임베딩 추출이 이미 완료되어 있습니다!")
        return

    print(f"▶️ 추출 대상: {len(df_to_process):,} / {total_seqs:,} 개")

    # 3. 데이터로더 설정
    dataset = ProteinDataset(df_to_process)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4. 임베딩 추출 및 HDF5 저장 루프
    # 모드 'a'는 기존 파일에 데이터를 덧붙이는(Append) 모드입니다.
    with h5py.File(output_h5, 'a') as f:
        with torch.no_grad():  # 역전파 비활성화 (메모리 극대화)
            for batch_ids, batch_seqs in tqdm(dataloader, desc="🧬 임베딩 추출 진행률"):

                # 토크나이징 (특수 토큰 포함되므로 max_length=1026)
                inputs = tokenizer(
                    list(batch_seqs),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1026
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # 모델 통과
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

                # Mean Pooling (각 서열별로 특수 토큰 <cls>, <eos> 제외한 실제 아미노산 부분만 평균)
                for i, seq_id in enumerate(batch_ids):
                    seq_len = len(batch_seqs[i])
                    # index 1부터 seq_len까지가 실제 아미노산 토큰입니다.
                    actual_tokens = hidden_states[i, 1:seq_len + 1, :]

                    # 1D 벡터로 압축 (Mean Pooling)
                    protein_vector = actual_tokens.mean(dim=0).cpu().numpy()

                    # HDF5에 단백질 ID를 키(Key)로 하여 넘파이 배열 저장
                    if seq_id not in f:
                        f.create_dataset(seq_id, data=protein_vector)

    print("==========================================")
    print(f"🎉 임베딩 추출 및 저장 완료: {output_h5}")
    print("==========================================")


if __name__ == "__main__":
    base_dir = "/home/team5/workspace/sj/homo_protein_seq_fasta"

    # 입력 파일: 전처리가 끝난 최종 CSV
    input_csv = os.path.join(base_dir, "esm2_ready_dataset_cropped_final.csv")

    # 출력 파일: 대용량 벡터 저장을 위한 HDF5 포맷
    output_h5 = os.path.join(base_dir, "esm2_embeddings.h5")

    # 모델 선택 (150M 파라미터 모델 권장. VRAM이 부족하면 facebook/esm2_t12_35M_UR50D 로 변경)
    target_model = "facebook/esm2_t30_150M_UR50D"

    # GPU VRAM(그래픽카드 메모리) 사양에 따라 batch_size 조절 (8, 16, 32 등)
    # Out Of Memory (OOM) 에러가 나면 이 숫자를 8이나 4로 줄이세요.
    generate_embeddings(input_csv, output_h5, model_name=target_model, batch_size=16)