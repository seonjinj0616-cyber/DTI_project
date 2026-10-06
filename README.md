# DTI Project: 약물-표적 상호작용 및 적응증 예측 파이프라인

화합물 SMILES 하나를 입력하면, 결합 가능성이 높은 인간 단백질을 예측하고 그 표적들을 후보 질병(적응증)과 연결하는 파이프라인입니다.

전체 흐름은 3단계입니다.

| 단계 | 스크립트 | 하는 일 |
|------|----------|---------|
| 1. 스크리닝 | `model/protein_Top100_classification.py` | MoLFormer 로 SMILES 를 인코딩하고, Ki/Kd 멀티태스크 분류 모델로 인간 단백질 약 11만 3천 개(ESM-2, 1280차원)와 점수를 매겨 고유 UniProt ID Top100 을 뽑습니다. |
| 2. 재랭킹 | `model/step2_Top20_regression.py` | 화합물→단백질 cross-attention 모델로 IC50 을 예측해 Top20 으로 재랭킹하고, attention peak 가 가장 뚜렷한 최종 5개 단백질을 고릅니다. attention, gradient, ablation 으로 residue 단위 결합 부위도 분석합니다. |
| 3. 적응증 연결 | `model/step3_ot_txgnn_link.py` + `model/DTI_txgnn_conda_python_3_11/step3_txgnn_score.py` | 최종 5개 표적을 Open Targets 질병 근거와 연결하고, 질병을 PrimeKG 노드로 매핑한 뒤 TxGNN 으로 (약물, 질병) 점수를 계산합니다. |

## 빠른 시작

### 1. conda 환경 만들기

TxGNN 이 `torch`, `dgl`, `numpy`, `pandas` 의 예전 버전을 요구하기 때문에 conda 환경이 **두 개** 필요합니다.

| 환경 | 사용 단계 | 주요 패키지 |
|------|-----------|-------------|
| `p_proj` | 1단계, 2단계, 3단계(prepare / finalize) | torch(CPU), transformers, pandas, h5py, duckdb, scikit-learn |
| `DTI_txgnn` | 3단계 TxGNN 점수 계산 | torch 2.2.2, dgl 1.1.2, numpy 1.26.4, pandas 1.5.3, txgnn 0.0.3 |

Anaconda Prompt 에서 설치 스크립트를 한 번 실행합니다.

```bat
cd DTI_project\model
setup_environments.bat
```

`cmd.exe` 에서 `conda` 명령이 안 되면 `conda init cmd.exe` 를 한 번 실행한 뒤 새 창에서 다시 시도하세요.
스크립트는 CPU 전용 PyTorch 를 설치합니다. CUDA GPU 를 쓰려면 실행 전에 스크립트의 `torch` 설치 줄을 맞는 CUDA 버전으로 바꾸세요.

### 2. 데이터 준비

용량이 큰 데이터는 GitHub 에 포함되어 있지 않습니다. 아래 파일을 받아서 표의 위치에 넣어야 실행할 수 있습니다. 위치는 `DTI_project/data/` 기준입니다.

**직접 만든 파일** (별도 저장소에서 받기)

| 파일 | 넣을 위치 | 크기 | 받는 곳 |
|------|-----------|------|---------|
| `whole_human_protein_vector_esm2_t33_650M_1280d_sequence_v2.h5` | `esm2_human_whole_protein_vecterDB/` | 약 930 MB | `<다운로드 링크>` |
| `esm2_ready_dataset_cropped_final.csv` | `esm2_human_whole_protein_vecterDB/` | 약 80 MB | `<다운로드 링크>` |
| `cdss_integrated.db` | `open_target_dataDB/` | 약 860 MB | `<다운로드 링크>` |

**공개 데이터** (원래 출처에서 받기)

| 파일 | 넣을 위치 | 크기 | 받는 곳 |
|------|-----------|------|---------|
| `kg.csv` | `primekg/` | 약 940 MB | [PrimeKG, Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) |
| `kg.csv`, `node.csv`, `edges.csv` | `txgnn_data/` | 약 1.5 GB | Harvard Dataverse 파일 [7144484](https://dataverse.harvard.edu/api/access/datafile/7144484) (kg.csv), [7144482](https://dataverse.harvard.edu/api/access/datafile/7144482) (node.csv), [7144483](https://dataverse.harvard.edu/api/access/datafile/7144483) (edges.csv) |
| `model.pt`, `config.pkl` | `txgnn_ckpt/` | 약 100 MB | [TxGNN GitHub](https://github.com/mims-harvard/TxGNN) README 의 사전학습 가중치 |

**자동으로 만들어지는 파일** (받을 필요 없음)

- `esm2_human_whole_protein_vecterDB/consolidated_cache/` 는 첫 실행 때 h5 파일에서 만들어집니다. 첫 실행만 몇 분 더 걸립니다.
- `txgnn_data/full_graph_42/`, `txgnn_data/kg_directed.csv` 는 TxGNN 이 첫 실행 때 만듭니다.

**선택 사항**

- `open_target_dataDB/opentargets/disease/` 폴더(Open Targets disease parquet)가 있으면 질병 ID 를 PrimeKG 에 더 많이 매핑할 수 있습니다. `open_target_dataDB/` 의 다운로드 노트북으로 받을 수 있습니다.

다 넣은 뒤 `data` 폴더 구조는 다음과 같습니다.

```
DTI_project/data/
├── esm2_human_whole_protein_vecterDB/
│   ├── whole_human_protein_vector_esm2_t33_650M_1280d_sequence_v2.h5
│   └── esm2_ready_dataset_cropped_final.csv
├── open_target_dataDB/
│   └── cdss_integrated.db
├── primekg/
│   └── kg.csv
├── txgnn_data/
│   ├── kg.csv
│   ├── node.csv
│   └── edges.csv
└── txgnn_ckpt/
    ├── model.pt
    └── config.pkl
```

학습에 쓴 원본 데이터(BindingDB, ChEMBL, 2D 임베딩 캐시 등, 약 230 GB)는 실행에 필요하지 않습니다. 다시 만들려면 아래 "데이터와 모델을 만든 과정"을 참고하세요.

### 3. 전체 파이프라인 실행

```bat
conda activate p_proj
cd DTI_project\model
python step3_ot_txgnn_link.py prepare --auto-chain --smiles "<SMILES>"
```

이매티닙(imatinib) 예시:

```bat
python step3_ot_txgnn_link.py prepare --auto-chain --smiles "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
```

`--auto-chain` 을 주면 2단계 결과가 없을 때 1, 2단계부터 자동으로 실행합니다. 이어서 `DTI_txgnn` 환경으로 자동 전환해 TxGNN 점수를 계산하고 `finalize` 까지 마칩니다. 결과를 `cdss_integrated.db` 에도 저장하려면 `--write-db` 를 추가하세요.

첫 실행 때 Hugging Face 에서 `ibm-research/MoLFormer-XL-both-10pct` 를 내려받으므로 인터넷 연결이 필요합니다. 3단계는 UniProt 과 PubChem API 도 호출하며, 응답은 `results/` 아래에 캐시됩니다.

## 단계별 실행

```bat
:: 1단계 + 2단계 (--top100-csv 를 주지 않으면 1단계가 자동 실행됨)
python step2_Top20_regression.py --smiles "<SMILES>"

:: 3a단계: 가장 최근 final5_proteins.csv 로 TxGNN 입력 파일 생성
python step3_ot_txgnn_link.py prepare --smiles "<SMILES>"

:: 3b단계: TxGNN 점수 계산 (DTI_txgnn 환경)
conda activate DTI_txgnn
python DTI_txgnn_conda_python_3_11\step3_txgnn_score.py --run-dir ..\results\step3_XXXX

:: 3c단계: 점수를 합쳐 최종 표 생성 (다시 p_proj 환경)
conda activate p_proj
python step3_ot_txgnn_link.py finalize --run-dir ..\results\step3_XXXX
```

전체 옵션은 각 스크립트에 `--help` 를 붙여 확인하세요. 자주 쓰는 옵션은 다음과 같습니다.

- **`--top-n`, `--final-n`** 은 재랭킹과 최종 선택에서 남길 단백질 수입니다. 기본값은 20 과 5 입니다.
- **`--peak-method`** 는 attention peak 지표를 고릅니다. 기본값은 `window_enrichment` 입니다.
- **`--drug-id`** 로 DrugBank ID 를 직접 지정할 수 있습니다. PrimeKG 약물과 연결하는 가장 확실한 방법입니다.
- **`--proxy-if-missing`** 을 주면 입력 화합물이 PrimeKG 에 없을 때 같은 표적을 가진 약물을 대리(proxy)로 점수 계산합니다.

### 신규 화합물과 TxGNN

TxGNN 은 PrimeKG 에 이미 있는 약물 노드만 점수를 낼 수 있습니다. 신규 화합물이어도 Open Targets 질병 후보는 나오지만, TxGNN 점수를 보려면 proxy 옵션이 필요합니다. proxy 점수는 결과에 `txgnn_score_source = proxy` 로 표시되며, 입력 화합물 자체의 점수가 아닙니다.

## 출력 결과

실행할 때마다 `DTI_project/results/` 아래에 시각이 붙은 폴더가 생깁니다.

**`stage2_<시각>/`**
- `rerank_top100.csv`: 후보 전체의 IC50 예측값, 순위, peak 점수, 최종 선택 여부.
- `final5_proteins.csv`: 최종 선택된 표적. 3단계의 입력입니다.
- `top20_attention_*.csv`, `detail_*.csv`: attention 프로필과 residue 단위 결합 부위 분석.
- `run_config.json`: 실행 설정.

**`step3_<시각>_<id>/`**
- `disease_candidates_final.csv`: Open Targets 근거와 TxGNN 점수를 합친 최종 질병 표.
- `disease_candidates_chart.html`: 같은 결과의 인터랙티브 차트.
- `ot_target_map.csv`, `ot_target_disease_pairs.csv`: 표적 매핑과 표적-질병 쌍 상세.
- `target_safety_summary.csv`, `known_side_effects.csv`, `target_pathways_summary.csv`: 참고용 안전성, 부작용, 경로 정보.
- `txgnn_drugs.csv`, `txgnn_diseases.csv`, `txgnn_scores.csv`: TxGNN 입력과 원본 점수.

## 폴더 구조

```
DTI_project/
├── model/                    추론 파이프라인 (여기서 실행)
│   ├── model/                학습된 가중치 (1단계 Ki/Kd 분류, 2단계 IC50 cross-attention)
│   ├── script/               2단계 모델 정의와 site 비교 코드
│   ├── DTI_txgnn_conda_python_3_11/   TxGNN 점수 계산 (DTI_txgnn 환경)
│   └── setup_environments.bat
├── scripts/                  데이터 준비와 모델 학습 (아래 순서대로 한 번 실행)
├── data/                     데이터 (GitHub 에는 코드 파일만 포함, 아래 "데이터 준비" 참고)
├── results/                  파이프라인 결과와 API 캐시
└── 실행 방법.txt              기존 실행 안내
```

스크립트는 자기 위치에서 위로 올라가며 `DTI_project` 폴더를 찾습니다. 폴더 이름만 `DTI_project` 로 유지하면 다른 드라이브나 컴퓨터로 옮겨도 경로를 다시 찾습니다.

## 데이터와 모델을 만든 과정

`scripts/` 아래 폴더는 학습 데이터와 모델을 만든 과정을 대략 다음 순서로 기록합니다.

1. **`bindingdb & chembl to csv/`** 는 BindingDB 와 ChEMBL 37 을 친화도 CSV 로 변환하고 QC 합니다.
2. **`chembl_bindingdb merge & normalization/`** 는 SMILES 를 정리하고 두 출처를 합친 뒤 중복을 처리하고 pAffinity 로 변환합니다.
3. **`mergingdb_protein_seq_matching/`** 는 UniProt 에서 단백질 서열을 받아 ESM-2 한도인 1024 residue 로 자릅니다.
4. **`human_whole_protein vector DB/`** 는 인간 단백질 전체(wild type, humsavar 변이체, 긴 단백질의 window)를 만들고 ESM-2 `esm2_t33_650M_UR50D` 로 임베딩합니다.
5. **`mergingdb_embedding_to_vector/`** 는 합친 상호작용 데이터를 HDF5 의 1D, 2D 토큰 벡터로 임베딩합니다.
6. **`DTI model training/`** 은 1단계 Ki/Kd 멀티태스크 분류 모델과 2단계 IC50 cross-attention 회귀 모델을 학습합니다.
7. **`Data distribution/`** 은 친화도 분포와 데이터 겹침을 분석합니다.

`data/open_target_dataDB/` 에는 Open Targets Platform parquet 파일을 SQLite DB `cdss_integrated.db` 로 적재하는 스크립트가 있습니다.

## 진단 도구

- **`model/diagnose_known_targets.py`** 는 Top100 cutoff 와 상관없이, 알려진 표적이 1단계에서 정확히 몇 등인지 보여줍니다.
- **`model/DTI_txgnn_conda_python_3_11/check_auroc_direction.py`** 는 TxGNN 점수가 높을수록 유력한 적응증이라는 가정이 맞는지 확인합니다.
- **`model/test_molformer_determinism.py`** 는 MoLFormer 임베딩이 재현 가능한지 확인합니다.

## 주의 사항

- attention 은 모델이 어디를 봤는지일 뿐, 예측의 원인과 같지 않을 수 있습니다. 그래서 2단계는 gradient 와 ablation 점수, 방법 간 일치도를 함께 저장합니다.
- 단백질 서열은 학습 때와 같이 앞 1024 residue 까지만 사용합니다.
- Open Targets 테이블의 `overall_score` 는 개별 근거 점수 중 최댓값(0.4 이상)입니다. Open Targets 웹사이트의 종합 association score 와는 다릅니다.
- 실행에 필요한 데이터는 약 5.5 GB 입니다. 학습용 데이터까지 모두 만들면 약 233 GB 가 필요합니다.

## 데이터 출처

- [BindingDB](https://www.bindingdb.org/), [ChEMBL 37](https://www.ebi.ac.uk/chembl/): 친화도 데이터
- [UniProt](https://www.uniprot.org/): 서열과 교차참조
- [Open Targets Platform](https://platform.opentargets.org/): 표적-질병 근거
- [PrimeKG](https://github.com/mims-harvard/PrimeKG), [TxGNN](https://github.com/mims-harvard/TxGNN): 지식 그래프와 적응증 점수
- [ESM-2](https://github.com/facebookresearch/esm), [MoLFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct): 단백질과 화합물 임베딩
