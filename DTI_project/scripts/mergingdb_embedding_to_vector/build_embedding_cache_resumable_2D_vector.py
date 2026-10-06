import os
import gc
import hashlib

import h5py
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, EsmModel


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = r"/home/team5/workspace/sj"
BASE_DIR = r"/home/team5/workspace/sj/homo_protein_seq_fasta"

INPUT_FILES = {
    "Ki": os.path.join(BASE_DIR, "Ki_ems2_fi.csv"),
    "Kd": os.path.join(BASE_DIR, "Kd_ems2_fi.csv"),
    "IC50": os.path.join(BASE_DIR, "IC50_ems2_fi.csv"),
}

OUTPUT_H5 = os.path.join(BASE_DIR, "DTI_token_embedding_cache_float32.h5")
OUTPUT_INDEX = os.path.join(BASE_DIR, "DTI_interactions_indexed.csv")

MOL_MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

MOL_HIDDEN_SIZE = 768
ESM_HIDDEN_SIZE = 1280

MOL_BATCH_SIZE = 64
ESM_BATCH_SIZE = 8

MAX_PROTEIN_AA = 1024

SAVE_DTYPE = np.float32

H5_COMPRESSION = "gzip"
H5_COMPRESSION_LEVEL = 4

# 몇 배치마다 디스크에 실제 반영(flush+fsync)할지
# (중간에 죽어도 이 시점까지는 보존됨)
FLUSH_EVERY_N_BATCHES = 5

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("DTI Token Embedding Cache Builder (resumable)")
print("=" * 90)
print(f"Device              : {DEVICE}")
print(f"MoLFormer           : {MOL_MODEL_NAME}")
print(f"ESM-2               : {ESM_MODEL_NAME}")
print(f"Protein max AA      : {MAX_PROTEIN_AA}")
print(f"H5 dtype            : {SAVE_DTYPE}")
print(f"Output H5           : {OUTPUT_H5}")
print(f"Output index        : {OUTPUT_INDEX}")
print("=" * 90)


# ============================================================
# Utility
# ============================================================

def normalize_smiles(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x else None


def normalize_uniprot(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x if x else None


def normalize_sequence(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    return x if x else None


def sha256_text(x):
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def sync_to_disk(h5file):
    """
    h5py의 flush()는 HDF5 라이브러리 내부 버퍼를 OS로 넘기는 것까지만
    보장하고, OS 페이지 캐시에서 실제 디스크로 내려가는 것(fsync)까지는
    보장하지 않을 수 있다. kill -9로 프로세스가 죽으면 페이지 캐시에만
    있던 데이터는 사라질 수 있으므로, flush 뒤 반드시 fsync까지 강제한다.

    (이전에 겪은 "eoa=2048" 파일 손상 - 헤더만 디스크에 남고 나머지는
     전혀 기록되지 않았던 문제 - 이 부분이 누락되어 생겼을 가능성이 높다.)
    """
    h5file.flush()
    try:
        fd = h5file.id.get_vfd_handle()
        os.fsync(fd)
    except Exception as e:
        # 일부 드라이버(core 등)는 vfd handle을 못 줄 수 있음 -> flush만으로 최선
        print(f"  [경고] os.fsync 실패 (flush만 적용됨): {e}")


# ============================================================
# 1. Load CSVs
# ============================================================

print("\n" + "=" * 90)
print("1. Loading interaction CSVs")
print("=" * 90)

dfs = []

for task_idx, (task_name, path) in enumerate(INPUT_FILES.items()):

    print(f"\n[{task_name}] {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    print(f"Rows: {len(df):,}")

    required = [
        "smiles", "uniprot_id", "target_name", "affinity_type",
        "affinity_value_mean", "paffinity", "protein_sequence",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{task_name}: missing columns: {missing}")

    df["smiles"] = df["smiles"].map(normalize_smiles)
    df["uniprot_id"] = df["uniprot_id"].map(normalize_uniprot)
    df["protein_sequence"] = df["protein_sequence"].map(normalize_sequence)
    df["affinity_type"] = df["affinity_type"].astype(str).str.upper().str.strip()

    df["task_idx"] = task_idx
    df["task_name"] = task_name

    dfs.append(df)


# ============================================================
# 2. Combine
# ============================================================

print("\n" + "=" * 90)
print("2. Combining datasets")
print("=" * 90)

all_df = pd.concat(dfs, axis=0, ignore_index=True)
del dfs
gc.collect()

print(f"Total interactions: {len(all_df):,}")


# ============================================================
# 3. Basic QC
# ============================================================

print("\n" + "=" * 90)
print("3. Basic QC")
print("=" * 90)

# NOTE: paffinity를 필수 non-null 컬럼에 추가.
# 이 값이 비어있으면 나중에 학습 시 loss가 조용히 NaN이 될 수 있음.
required_nonnull = ["smiles", "uniprot_id", "protein_sequence", "paffinity"]

for col in required_nonnull:
    n_missing = all_df[col].isna().sum()
    print(f"{col:20s}: missing = {n_missing:,}")

if all_df[required_nonnull].isna().any().any():
    before = len(all_df)
    all_df = all_df.dropna(subset=required_nonnull).reset_index(drop=True)
    print(f"Dropped rows with missing identifiers/paffinity: {before - len(all_df):,}")


# ============================================================
# 4. Protein UniProt <-> sequence consistency
# ============================================================

print("\n" + "=" * 90)
print("4. Checking UniProt -> protein sequence consistency")
print("=" * 90)

protein_check = all_df[["uniprot_id", "protein_sequence"]].drop_duplicates()
protein_counts = protein_check.groupby("uniprot_id")["protein_sequence"].nunique()
conflicting_proteins = protein_counts[protein_counts > 1]

if len(conflicting_proteins) > 0:
    print(f"ERROR: {len(conflicting_proteins):,} UniProt IDs have multiple sequences.")
    print(conflicting_proteins.head(20))
    raise ValueError("UniProt -> sequence mapping is not unique.")

print("PASS: every UniProt ID maps to exactly one sequence.")


# ============================================================
# 5. Clip protein sequences exactly as training
# ============================================================

print("\n" + "=" * 90)
print("5. Applying protein sequence length rule")
print("=" * 90)

all_df["original_protein_length"] = all_df["protein_sequence"].str.len()
all_df["protein_sequence_used"] = all_df["protein_sequence"].str.slice(0, MAX_PROTEIN_AA)
all_df["used_protein_length"] = all_df["protein_sequence_used"].str.len()

print(
    "Sequences > 1024 AA:",
    (all_df["original_protein_length"] > MAX_PROTEIN_AA).sum()
)


# ============================================================
# 6. Stable compound IDs
# ============================================================

print("\n" + "=" * 90)
print("6. Creating compound IDs")
print("=" * 90)

unique_smiles = all_df["smiles"].drop_duplicates().sort_values().reset_index(drop=True)

compound_map = pd.DataFrame({
    "compound_id": np.arange(len(unique_smiles), dtype=np.int32),
    "smiles": unique_smiles.values,
})
compound_map["smiles_hash"] = compound_map["smiles"].map(sha256_text)

print(f"Unique compounds: {len(compound_map):,}")


# ============================================================
# 7. Stable protein IDs
# ============================================================

print("\n" + "=" * 90)
print("7. Creating protein IDs")
print("=" * 90)

protein_map = (
    all_df[["uniprot_id", "protein_sequence_used", "original_protein_length", "used_protein_length"]]
    .drop_duplicates(subset=["uniprot_id"])
    .sort_values("uniprot_id")
    .reset_index(drop=True)
)
protein_map.insert(0, "protein_id", np.arange(len(protein_map), dtype=np.int32))
protein_map["sequence_hash"] = protein_map["protein_sequence_used"].map(sha256_text)

print(f"Unique proteins: {len(protein_map):,}")


# ============================================================
# 8. Verify IDs + positional-index assumption
# ============================================================

assert protein_map["protein_id"].is_unique
assert protein_map["uniprot_id"].is_unique
assert compound_map["compound_id"].is_unique
assert compound_map["smiles"].is_unique

# NOTE: 아래 루프들은 "compound_map.iloc[i]가 compound_id==i인 행"이라는
# 가정에 의존한다. arange로 ID를 부여하고 이후 재정렬하지 않았으므로
# 지금은 항상 참이지만, 이 가정이 깨지면 조용히 잘못된 smiles/서열이
# 잘못된 embedding에 매핑될 수 있으므로 명시적으로 검증한다.
assert np.array_equal(compound_map["compound_id"].values, np.arange(len(compound_map)))
assert np.array_equal(protein_map["protein_id"].values, np.arange(len(protein_map)))

# 이후 코드에서 O(1) 위치 접근을 위해 numpy 배열로 고정
compound_smiles_arr = compound_map["smiles"].to_numpy()
protein_sequence_arr = protein_map["protein_sequence_used"].to_numpy()


# ============================================================
# 9. Merge IDs into interaction table
# ============================================================

print("\n" + "=" * 90)
print("9. Building indexed interaction table")
print("=" * 90)

all_df = all_df.merge(
    compound_map[["smiles", "compound_id", "smiles_hash"]],
    on="smiles", how="left", validate="many_to_one",
)
all_df = all_df.merge(
    protein_map[["uniprot_id", "protein_id", "sequence_hash"]],
    on="uniprot_id", how="left", validate="many_to_one",
)

if all_df["compound_id"].isna().any():
    raise RuntimeError(f"Missing compound_id: {all_df['compound_id'].isna().sum():,}")
if all_df["protein_id"].isna().any():
    raise RuntimeError(f"Missing protein_id: {all_df['protein_id'].isna().sum():,}")

all_df["compound_id"] = all_df["compound_id"].astype(np.int32)
all_df["protein_id"] = all_df["protein_id"].astype(np.int32)

print("PASS: all interactions assigned compound_id and protein_id.")


# ============================================================
# 10. interaction_id + column ordering + save index
# ============================================================

all_df.insert(0, "interaction_id", np.arange(len(all_df), dtype=np.int64))

priority_columns = [
    "interaction_id", "compound_id", "protein_id", "task_idx", "task_name",
    "smiles", "uniprot_id", "target_name", "affinity_type",
    "affinity_value_mean", "paffinity",
]
existing_priority = [c for c in priority_columns if c in all_df.columns]
remaining_columns = [c for c in all_df.columns if c not in existing_priority]
all_df = all_df[existing_priority + remaining_columns]

print("\nSaving interaction index...")
all_df.to_csv(OUTPUT_INDEX, index=False)
print(f"Saved: {OUTPUT_INDEX}")


# ============================================================
# 11. Load models
# ============================================================

print("\n" + "=" * 90)
print("11. Loading MoLFormer")
print("=" * 90)

mol_tokenizer = AutoTokenizer.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
mol_model = AutoModel.from_pretrained(MOL_MODEL_NAME, trust_remote_code=True)
mol_model.to(DEVICE)
mol_model.eval()
for p in mol_model.parameters():
    p.requires_grad = False

print("\n" + "=" * 90)
print("12. Loading ESM-2")
print("=" * 90)

esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME)
esm_model.to(DEVICE)
esm_model.eval()
for p in esm_model.parameters():
    p.requires_grad = False


# ============================================================
# 13. HDF5 open (신규 생성 또는 기존 파일 이어받기)
# ============================================================

print("\n" + "=" * 90)
print("13. Opening / creating HDF5 cache")
print("=" * 90)

n_compounds = len(compound_map)
n_proteins = len(protein_map)

resuming = os.path.exists(OUTPUT_H5)

if resuming:

    print(f"기존 H5 파일 발견 -> 이어서 진행합니다: {OUTPUT_H5}")

    h5 = h5py.File(OUTPUT_H5, "a")

    # ----------------------------------------------------
    # 기존 파일이 지금 이 CSV/설정으로 만들어진 게 맞는지 검증.
    # 이게 다르면 잘못된 캐시에 이어쓰기를 하게 되므로 반드시 확인.
    # ----------------------------------------------------

    def _check_attr(name, expected):
        actual = h5.attrs.get(name)
        if actual != expected:
            raise RuntimeError(
                f"기존 H5의 '{name}' 속성({actual})이 현재 설정({expected})과 다릅니다. "
                f"다른 데이터/설정으로 만들어진 캐시일 수 있으니 수동으로 확인해주세요."
            )

    _check_attr("mol_model", MOL_MODEL_NAME)
    _check_attr("esm_model", ESM_MODEL_NAME)
    _check_attr("mol_hidden_size", MOL_HIDDEN_SIZE)
    _check_attr("esm_hidden_size", ESM_HIDDEN_SIZE)
    _check_attr("max_protein_aa", MAX_PROTEIN_AA)

    compound_group = h5["compound"]
    protein_group = h5["protein"]

    compound_embeddings = compound_group["embeddings"]
    compound_offsets = compound_group["offsets"]
    compound_lengths = compound_group["lengths"]
    compound_ids = compound_group["ids"]
    compound_smiles = compound_group["smiles"]
    compound_hashes = compound_group["hashes"]

    protein_embeddings = protein_group["embeddings"]
    protein_offsets = protein_group["offsets"]
    protein_lengths = protein_group["lengths"]
    protein_ids = protein_group["ids"]
    protein_uniprot = protein_group["uniprot_ids"]
    protein_hashes = protein_group["hashes"]
    protein_original_lengths = protein_group["original_lengths"]
    protein_used_lengths = protein_group["used_lengths"]

    if compound_offsets.shape[0] != n_compounds:
        raise RuntimeError(
            f"기존 H5의 compound 수({compound_offsets.shape[0]:,})가 "
            f"현재 CSV로 계산한 compound 수({n_compounds:,})와 다릅니다. "
            f"입력 CSV가 바뀐 것 같습니다 - 이어쓰기를 중단합니다."
        )
    if protein_offsets.shape[0] != n_proteins:
        raise RuntimeError(
            f"기존 H5의 protein 수({protein_offsets.shape[0]:,})가 "
            f"현재 CSV로 계산한 protein 수({n_proteins:,})와 다릅니다. "
            f"입력 CSV가 바뀐 것 같습니다 - 이어쓰기를 중단합니다."
        )

    print("검증 통과: 기존 캐시가 현재 설정/데이터와 일치합니다.")

else:

    print(f"새 H5 파일을 생성합니다: {OUTPUT_H5}")

    h5 = h5py.File(OUTPUT_H5, "w")

    h5.attrs["cache_version"] = "1.1"
    h5.attrs["embedding_dtype"] = "float32"
    h5.attrs["mol_model"] = MOL_MODEL_NAME
    h5.attrs["esm_model"] = ESM_MODEL_NAME
    h5.attrs["mol_hidden_size"] = MOL_HIDDEN_SIZE
    h5.attrs["esm_hidden_size"] = ESM_HIDDEN_SIZE
    h5.attrs["max_protein_aa"] = MAX_PROTEIN_AA

    compound_group = h5.create_group("compound")

    compound_embeddings = compound_group.create_dataset(
        "embeddings", shape=(0, MOL_HIDDEN_SIZE), maxshape=(None, MOL_HIDDEN_SIZE),
        dtype=SAVE_DTYPE, chunks=(4096, MOL_HIDDEN_SIZE),
        compression=H5_COMPRESSION, compression_opts=H5_COMPRESSION_LEVEL,
    )
    compound_offsets = compound_group.create_dataset("offsets", shape=(n_compounds,), dtype=np.int64)
    compound_lengths = compound_group.create_dataset("lengths", shape=(n_compounds,), dtype=np.int32)
    compound_ids = compound_group.create_dataset(
        "ids", data=compound_map["compound_id"].values, dtype=np.int32
    )
    compound_smiles = compound_group.create_dataset(
        "smiles", shape=(n_compounds,), dtype=h5py.string_dtype("utf-8")
    )
    compound_hashes = compound_group.create_dataset(
        "hashes", shape=(n_compounds,), dtype=h5py.string_dtype("utf-8")
    )

    protein_group = h5.create_group("protein")

    protein_embeddings = protein_group.create_dataset(
        "embeddings", shape=(0, ESM_HIDDEN_SIZE), maxshape=(None, ESM_HIDDEN_SIZE),
        dtype=SAVE_DTYPE, chunks=(2048, ESM_HIDDEN_SIZE),
        compression=H5_COMPRESSION, compression_opts=H5_COMPRESSION_LEVEL,
    )
    protein_offsets = protein_group.create_dataset("offsets", shape=(n_proteins,), dtype=np.int64)
    protein_lengths = protein_group.create_dataset("lengths", shape=(n_proteins,), dtype=np.int32)
    protein_ids = protein_group.create_dataset(
        "ids", data=protein_map["protein_id"].values, dtype=np.int32
    )
    protein_uniprot = protein_group.create_dataset(
        "uniprot_ids", shape=(n_proteins,), dtype=h5py.string_dtype("utf-8")
    )
    protein_hashes = protein_group.create_dataset(
        "hashes", shape=(n_proteins,), dtype=h5py.string_dtype("utf-8")
    )
    protein_original_lengths = protein_group.create_dataset(
        "original_lengths", shape=(n_proteins,), dtype=np.int32
    )
    protein_used_lengths = protein_group.create_dataset(
        "used_lengths", shape=(n_proteins,), dtype=np.int32
    )


# ============================================================
# 14. Resume 지점 계산
# ============================================================
#
# lengths == 0 인 항목 = 아직 처리 안 됨 (정상적인 화합물/단백질은
# 항상 토큰이 1개 이상이므로 0은 "미처리"를 뜻하는 안전한 sentinel).
# 처리 순서와 무관하게 각 항목의 embedding은 offsets[id]에 정확히
# 기록되므로, 어떤 순서로 재개하든 데이터 무결성은 유지된다.

compound_lengths_arr = compound_lengths[:]
compound_remaining_ids = np.nonzero(compound_lengths_arr == 0)[0]

protein_lengths_arr = protein_lengths[:]
protein_remaining_ids = np.nonzero(protein_lengths_arr == 0)[0]

print(
    f"\nCompound: {n_compounds - len(compound_remaining_ids):,} / {n_compounds:,} 이미 처리됨 "
    f"-> {len(compound_remaining_ids):,}개 남음"
)
print(
    f"Protein : {n_proteins - len(protein_remaining_ids):,} / {n_proteins:,} 이미 처리됨 "
    f"-> {len(protein_remaining_ids):,}개 남음"
)


# ============================================================
# 15. Compound embedding generation (resume + batched write)
# ============================================================

if len(compound_remaining_ids) > 0:

    print("\n" + "=" * 90)
    print("15. Generating MoLFormer token embeddings")
    print("=" * 90)

    try:
        for batch_start in tqdm(
            range(0, len(compound_remaining_ids), MOL_BATCH_SIZE), desc="MoLFormer"
        ):

            batch_ids = compound_remaining_ids[batch_start:batch_start + MOL_BATCH_SIZE]
            batch_smiles = [compound_smiles_arr[i] for i in batch_ids]

            encoded = mol_tokenizer(
                batch_smiles, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = mol_model(**encoded)
                hidden = outputs.last_hidden_state.detach().cpu()

            attention_mask = encoded["attention_mask"].detach().cpu()

            # 배치 전체를 모아서 한 번만 write (item별 resize 반복 제거)
            batch_token_arrays = []
            batch_meta = []  # (compound_id, n_tokens)

            for i, compound_id in enumerate(batch_ids):
                valid = attention_mask[i].bool()
                tokens_np = hidden[i][valid].numpy().astype(SAVE_DTYPE)
                batch_token_arrays.append(tokens_np)
                batch_meta.append((int(compound_id), tokens_np.shape[0]))

            batch_concat = np.concatenate(batch_token_arrays, axis=0)

            old_size = compound_embeddings.shape[0]
            new_size = old_size + batch_concat.shape[0]
            compound_embeddings.resize((new_size, MOL_HIDDEN_SIZE))
            compound_embeddings[old_size:new_size] = batch_concat

            cursor = old_size
            for compound_id, n_tokens in batch_meta:
                compound_offsets[compound_id] = cursor
                compound_lengths[compound_id] = n_tokens
                compound_smiles[compound_id] = compound_map.iloc[compound_id]["smiles"]
                compound_hashes[compound_id] = compound_map.iloc[compound_id]["smiles_hash"]
                cursor += n_tokens

            del encoded, outputs, hidden

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            if (batch_start // MOL_BATCH_SIZE + 1) % FLUSH_EVERY_N_BATCHES == 0:
                sync_to_disk(h5)

    finally:
        sync_to_disk(h5)

    print("MoLFormer embedding 생성 완료 (flush 됨).")

else:
    print("\nCompound embedding: 이미 전부 처리되어 있어 건너뜁니다.")


# ============================================================
# 16. Protein embedding generation
#     (resume + 길이순 정렬 + batched write)
# ============================================================

if len(protein_remaining_ids) > 0:

    print("\n" + "=" * 90)
    print("16. Generating ESM-2 token embeddings")
    print("=" * 90)

    # ----------------------------------------------------
    # 길이순 정렬: 같은 배치 안에 길이가 비슷한 서열끼리 묶어서
    # padding 낭비를 줄인다. offsets는 protein_id 기준으로
    # 정확히 기록되므로 처리 순서와 무관하게 안전하다.
    # ----------------------------------------------------

    remaining_lengths = np.array(
        [len(protein_sequence_arr[i]) for i in protein_remaining_ids]
    )
    order = np.argsort(remaining_lengths)
    protein_remaining_ids_sorted = protein_remaining_ids[order]

    try:
        for batch_start in tqdm(
            range(0, len(protein_remaining_ids_sorted), ESM_BATCH_SIZE), desc="ESM-2"
        ):

            batch_ids = protein_remaining_ids_sorted[batch_start:batch_start + ESM_BATCH_SIZE]
            batch_sequences = [protein_sequence_arr[i] for i in batch_ids]

            encoded = esm_tokenizer(
                batch_sequences, padding=True, truncation=False, return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = esm_model(**encoded)
                hidden = outputs.last_hidden_state.detach().cpu()

            attention_mask = encoded["attention_mask"].detach().cpu()

            batch_token_arrays = []
            batch_meta = []

            for i, protein_id in enumerate(batch_ids):
                valid = attention_mask[i].bool()
                tokens_np = hidden[i][valid].numpy().astype(SAVE_DTYPE)
                batch_token_arrays.append(tokens_np)
                batch_meta.append((int(protein_id), tokens_np.shape[0]))

            batch_concat = np.concatenate(batch_token_arrays, axis=0)

            old_size = protein_embeddings.shape[0]
            new_size = old_size + batch_concat.shape[0]
            protein_embeddings.resize((new_size, ESM_HIDDEN_SIZE))
            protein_embeddings[old_size:new_size] = batch_concat

            cursor = old_size
            for protein_id, n_tokens in batch_meta:
                protein_offsets[protein_id] = cursor
                protein_lengths[protein_id] = n_tokens
                protein_uniprot[protein_id] = protein_map.iloc[protein_id]["uniprot_id"]
                protein_hashes[protein_id] = protein_map.iloc[protein_id]["sequence_hash"]
                protein_original_lengths[protein_id] = protein_map.iloc[protein_id]["original_protein_length"]
                protein_used_lengths[protein_id] = protein_map.iloc[protein_id]["used_protein_length"]
                cursor += n_tokens

            del encoded, outputs, hidden

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            if (batch_start // ESM_BATCH_SIZE + 1) % FLUSH_EVERY_N_BATCHES == 0:
                sync_to_disk(h5)

    finally:
        sync_to_disk(h5)

    print("ESM-2 embedding 생성 완료 (flush 됨).")

else:
    print("\nProtein embedding: 이미 전부 처리되어 있어 건너뜁니다.")


# ============================================================
# 17. Final QC
# ============================================================

print("\n" + "=" * 90)
print("17. Final QC")
print("=" * 90)

compound_lengths_final = compound_lengths[:]
protein_lengths_final = protein_lengths[:]

n_compound_done = int((compound_lengths_final > 0).sum())
n_protein_done = int((protein_lengths_final > 0).sum())

print(f"Compound done: {n_compound_done:,} / {n_compounds:,}")
print(f"Protein done : {n_protein_done:,} / {n_proteins:,}")

assert compound_embeddings.shape[0] == int(compound_lengths_final.sum())
assert protein_embeddings.shape[0] == int(protein_lengths_final.sum())

if n_compound_done < n_compounds or n_protein_done < n_proteins:
    print(
        "\n주의: 아직 전부 처리되지 않았습니다. "
        "이 스크립트를 그대로 다시 실행하면 남은 부분부터 이어서 진행됩니다."
    )
else:
    print("\n모든 compound/protein embedding 생성 완료.")


# ============================================================
# 18. Close
# ============================================================

sync_to_disk(h5)
h5.close()

del mol_model, esm_model
gc.collect()

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

h5_size_gb = os.path.getsize(OUTPUT_H5) / (1024 ** 3)
index_size_gb = os.path.getsize(OUTPUT_INDEX) / (1024 ** 3)

print("\n" + "=" * 90)
print("DONE (or safely paused - rerun to resume)")
print("=" * 90)
print(f"H5 file    : {OUTPUT_H5}  ({h5_size_gb:.3f} GB)")
print(f"Index file : {OUTPUT_INDEX}  ({index_size_gb:.3f} GB)")
print(f"Compounds  : {n_compounds:,}  (done: {n_compound_done:,})")
print(f"Proteins   : {n_proteins:,}  (done: {n_protein_done:,})")
print(f"Interactions: {len(all_df):,}")
print("=" * 90)
