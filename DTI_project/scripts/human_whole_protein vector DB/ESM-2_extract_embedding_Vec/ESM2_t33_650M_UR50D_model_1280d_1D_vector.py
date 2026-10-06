import os
import gc
import hashlib
import json

import torch
import h5py
import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

ESM_MAX_AA = 1024

# ESM-2 tokenizer:
# BOS + 1024 AA + EOS = 1026 tokens
ESM_MAX_TOKENS = 1026


# ============================================================
# 1. Sequence Hash
# ============================================================

def make_sequence_hash(sequence):
    """
    Create deterministic SHA-256 hash for protein sequence.

    Full 64-character hash is used to avoid practical collision
    concerns.
    """

    return hashlib.sha256(
        sequence.encode("utf-8")
    ).hexdigest()


# ============================================================
# 2. Sequence Normalization
# ============================================================

def normalize_sequence(sequence):
    """
    Normalize protein sequence.

    - convert to string
    - remove whitespace
    - uppercase
    """

    sequence = str(sequence)

    sequence = "".join(
        sequence.split()
    ).upper()

    return sequence


# ============================================================
# 3. Dataset
# ============================================================

class ProteinSequenceDataset(Dataset):

    def __init__(self, df):

        self.sequence_hashes = (
            df["sequence_hash"]
            .astype(str)
            .values
        )

        self.sequences = (
            df["sequence"]
            .astype(str)
            .values
        )

    def __len__(self):

        return len(
            self.sequence_hashes
        )

    def __getitem__(self, idx):

        return (
            self.sequence_hashes[idx],
            self.sequences[idx]
        )


# ============================================================
# 4. Prepare Sequence Dataset
# ============================================================

def prepare_sequence_dataset(
    csv_path
):

    print("\n" + "=" * 100)
    print("PREPARE HUMAN PROTEIN SEQUENCE DATASET")
    print("=" * 100)

    print(
        f"Input CSV : {csv_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path,
        low_memory=False
    )

    required_columns = [
        "uniprot_id",
        "sequence"
    ]

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(
        f"Original rows : {len(df):,}"
    )

    # --------------------------------------------------------
    # Remove missing
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "uniprot_id",
            "sequence"
        ]
    ).copy()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["uniprot_id"] = (
        df["uniprot_id"]
        .astype(str)
        .str.strip()
    )

    df["sequence"] = (
        df["sequence"]
        .map(normalize_sequence)
    )

    # --------------------------------------------------------
    # Remove empty sequences
    # --------------------------------------------------------

    df = df[
        df["sequence"].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    df["sequence_length"] = (
        df["sequence"].str.len()
    )

    # --------------------------------------------------------
    # Sequence hash
    # --------------------------------------------------------

    print(
        "\nGenerating sequence hashes..."
    )

    df["sequence_hash"] = (
        df["sequence"]
        .map(make_sequence_hash)
    )

    # --------------------------------------------------------
    # QC: UniProt → multiple sequences
    # --------------------------------------------------------

    sequence_per_uniprot = (
        df.groupby("uniprot_id")[
            "sequence_hash"
        ]
        .nunique()
    )

    multi_sequence_uniprot = (
        sequence_per_uniprot[
            sequence_per_uniprot > 1
        ]
    )

    print(
        "\nUniProt / sequence QC"
    )

    print(
        f"Unique UniProt IDs          : "
        f"{sequence_per_uniprot.shape[0]:,}"
    )

    print(
        f"UniProt IDs with multiple "
        f"sequences                    : "
        f"{len(multi_sequence_uniprot):,}"
    )

    # --------------------------------------------------------
    # Sequence-level deduplication
    # --------------------------------------------------------

    sequence_df = (
        df[
            [
                "sequence_hash",
                "sequence",
                "sequence_length"
            ]
        ]
        .drop_duplicates(
            subset=["sequence_hash"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Number of UniProt IDs per sequence
    # --------------------------------------------------------

    uniprot_mapping = (
        df.groupby("sequence_hash")[
            "uniprot_id"
        ]
        .apply(
            lambda x: sorted(
                set(x.astype(str))
            )
        )
        .to_dict()
    )

    sequence_df["uniprot_count"] = (
        sequence_df["sequence_hash"]
        .map(
            lambda x:
            len(uniprot_mapping[x])
        )
    )

    # --------------------------------------------------------
    # >1024 QC
    # --------------------------------------------------------

    long_sequence_df = sequence_df[
        sequence_df["sequence_length"]
        > ESM_MAX_AA
    ]

    print(
        f"\nUnique sequences            : "
        f"{len(sequence_df):,}"
    )

    print(
        f"> {ESM_MAX_AA} aa sequences      : "
        f"{len(long_sequence_df):,}"
    )

    print(
        f"Maximum sequence length     : "
        f"{sequence_df['sequence_length'].max():,}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print(
        "\nSequence length statistics"
    )

    print(
        sequence_df[
            "sequence_length"
        ].describe()
    )

    return (
        sequence_df,
        uniprot_mapping
    )


# ============================================================
# 5. Split Long Sequence into Chunks
# ============================================================

def split_sequence(
    sequence,
    chunk_size=ESM_MAX_AA
):

    return [
        sequence[i:i + chunk_size]
        for i in range(
            0,
            len(sequence),
            chunk_size
        )
    ]


# ============================================================
# 6. Mean Pooling
# ============================================================

def mean_pooling(
    hidden_states,
    attention_mask
):
    """
    Mean pool amino-acid tokens only.

    ESM-2:
        position 0 = BOS
        position 1..N = amino acids
        final token = EOS

    Padding is removed using attention mask.
    """

    token_hidden = hidden_states[:, 1:, :]
    token_mask = attention_mask[:, 1:]

    pooled_vectors = []

    for i in range(
        token_hidden.shape[0]
    ):

        valid_positions = (
            token_mask[i] == 1
        )

        valid_hidden = (
            token_hidden[i][
                valid_positions
            ]
        )

        # Last valid token is EOS
        if valid_hidden.shape[0] > 1:

            valid_hidden = (
                valid_hidden[:-1]
            )

        protein_vector = (
            valid_hidden
            .mean(dim=0)
        )

        pooled_vectors.append(
            protein_vector
        )

    return torch.stack(
        pooled_vectors
    )


# ============================================================
# 7. Generate ESM-2 Embeddings
# ============================================================

def generate_embeddings(
    csv_path,
    output_h5,
    model_name="facebook/esm2_t33_650M_UR50D",
    batch_size=4
):

    print("\n" + "=" * 100)
    print("ESM-2 WHOLE HUMAN PROTEIN VECTOR DB")
    print("=" * 100)

    print(
        f"Model       : {model_name}"
    )

    print(
        f"Input       : {csv_path}"
    )

    print(
        f"Output      : {output_h5}"
    )

    print(
        f"Batch size  : {batch_size}"
    )

    print(
        f"Max AA/chunk : {ESM_MAX_AA}"
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device      : {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    print(
        "Loading ESM-2 model..."
    )

    model = EsmModel.from_pretrained(
        model_name
    ).to(device)

    model.eval()

    hidden_dim = (
        model.config.hidden_size
    )

    print(
        f"Hidden dim  : {hidden_dim}"
    )

    if hidden_dim != 1280:

        print(
            f"WARNING: expected 1280 dimensions, "
            f"but model returned {hidden_dim}"
        )

    # ========================================================
    # Prepare dataset
    # ========================================================

    sequence_df, uniprot_mapping = (
        prepare_sequence_dataset(
            csv_path
        )
    )

    # ========================================================
    # Resume
    # ========================================================

    processed_hashes = set()

    if os.path.exists(output_h5):

        print(
            "\nExisting H5 detected."
        )

        with h5py.File(
            output_h5,
            "r"
        ) as f:

            if "vectors" in f:

                processed_hashes = set(
                    f["vectors"].keys()
                )

        print(
            f"Already embedded sequences : "
            f"{len(processed_hashes):,}"
        )

    df_to_process = sequence_df[
        ~sequence_df[
            "sequence_hash"
        ].isin(
            processed_hashes
        )
    ].copy()

    df_to_process = (
        df_to_process
        .reset_index(drop=True)
    )

    print(
        f"Remaining sequences          : "
        f"{len(df_to_process):,}"
    )

    if len(df_to_process) == 0:

        print(
            "\nAll sequences are already processed."
        )

        return

    # ========================================================
    # Dataset
    # ========================================================

    dataset = ProteinSequenceDataset(
        df_to_process
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ========================================================
    # HDF5
    # ========================================================

    with h5py.File(
        output_h5,
        "a"
    ) as f:

        vectors_group = (
            f.require_group(
                "vectors"
            )
        )

        metadata_group = (
            f.require_group(
                "metadata"
            )
        )

        # ----------------------------------------------------
        # Root attributes
        # ----------------------------------------------------

        f.attrs["model_name"] = model_name
        f.attrs["embedding_dim"] = hidden_dim

        f.attrs["pooling"] = (
            "mean_pooling_amino_acid_tokens_only"
        )

        f.attrs["max_aa_per_chunk"] = (
            ESM_MAX_AA
        )

        f.attrs["long_sequence_strategy"] = (
            "1024aa_chunking_then_length_weighted_mean"
        )

        f.attrs["sequence_key"] = (
            "sha256(sequence)"
        )

        f.attrs["version"] = "2.0"

        # ====================================================
        # Embedding Loop
        # ====================================================

        with torch.inference_mode():

            step_counter = 0

            for (
                batch_hashes,
                batch_sequences
            ) in tqdm(
                dataloader,
                desc="ESM-2 embedding"
            ):

                batch_sequences = list(
                    batch_sequences
                )

                for sequence_hash, sequence in zip(
                    batch_hashes,
                    batch_sequences
                ):

                    sequence_hash = str(
                        sequence_hash
                    )

                    sequence = str(
                        sequence
                    )

                    if (
                        sequence_hash
                        in vectors_group
                    ):
                        continue

                    # --------------------------------------------
                    # Split sequence into chunks
                    # --------------------------------------------

                    chunks = split_sequence(
                        sequence,
                        ESM_MAX_AA
                    )

                    # --------------------------------------------
                    # Batch Tokenization for Chunks
                    # --------------------------------------------

                    inputs = tokenizer(
                        chunks,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=ESM_MAX_TOKENS
                    )

                    inputs = {
                        k: v.to(device)
                        for k, v in inputs.items()
                    }

                    outputs = model(
                        **inputs
                    )

                    hidden_states = (
                        outputs.last_hidden_state
                    )

                    attention_mask = (
                        inputs["attention_mask"]
                    )

                    chunk_vectors = mean_pooling(
                        hidden_states,
                        attention_mask
                    )

                    chunk_lengths = [
                        len(chunk) for chunk in chunks
                    ]

                    # --------------------------------------------
                    # Length-weighted aggregation
                    # --------------------------------------------

                    total_length = sum(
                        chunk_lengths
                    )

                    weights = torch.tensor(
                        chunk_lengths,
                        dtype=torch.float32,
                        device=device
                    )

                    weights = (
                        weights
                        / weights.sum()
                    )

                    protein_vector = (
                        chunk_vectors
                        * weights.unsqueeze(1)
                    ).sum(dim=0)

                    # --------------------------------------------
                    # CPU / numpy conversion
                    # --------------------------------------------

                    protein_vector = (
                        protein_vector
                        .float()
                        .cpu()
                        .numpy()
                        .astype(
                            np.float32
                        )
                    )

                    # --------------------------------------------
                    # Save vector
                    # --------------------------------------------

                    vectors_group.create_dataset(
                        sequence_hash,
                        data=protein_vector,
                        dtype="float32"
                    )

                    # --------------------------------------------
                    # Save metadata
                    # --------------------------------------------

                    if (
                        sequence_hash
                        not in metadata_group
                    ):

                        protein_group = (
                            metadata_group.create_group(
                                sequence_hash
                            )
                        )

                        str_dtype = (
                            h5py.string_dtype(
                                encoding="utf-8"
                            )
                        )

                        protein_group.create_dataset(
                            "sequence",
                            data=sequence,
                            dtype=str_dtype
                        )

                        uniprot_ids = (
                            uniprot_mapping[
                                sequence_hash
                            ]
                        )

                        protein_group.create_dataset(
                            "uniprot_ids",
                            data=json.dumps(
                                uniprot_ids
                            ),
                            dtype=str_dtype
                        )

                        protein_group.attrs[
                            "sequence_length"
                        ] = len(sequence)

                        protein_group.attrs[
                            "uniprot_count"
                        ] = len(
                            uniprot_ids
                        )

                        protein_group.attrs[
                            "chunk_count"
                        ] = len(chunks)

                        protein_group.attrs[
                            "chunk_size"
                        ] = ESM_MAX_AA

                        protein_group.attrs[
                            "embedding_dim"
                        ] = hidden_dim

                        if len(chunks) == 1:
                            strategy = "direct_esm2_embedding"
                        else:
                            strategy = "1024aa_chunking_length_weighted_mean"

                        protein_group.attrs[
                            "embedding_strategy"
                        ] = strategy

                    # --------------------------------------------
                    # Clean up local variables per protein
                    # --------------------------------------------

                    del (
                        chunks,
                        chunk_vectors,
                        chunk_lengths,
                        weights,
                        protein_vector,
                        inputs,
                        outputs,
                        hidden_states,
                        attention_mask
                    )

                # --------------------------------------------
                # Periodic Garbage Collection (Every 100 batches)
                # --------------------------------------------

                step_counter += 1
                if step_counter % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    # ========================================================
    # Final QC
    # ========================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "FINAL QC"
    )
    print(
        "=" * 100
    )

    with h5py.File(
        output_h5,
        "r"
    ) as f:

        vector_count = len(
            f["vectors"]
        )

        metadata_count = len(
            f["metadata"]
        )

        print(
            f"Vector count        : "
            f"{vector_count:,}"
        )

        print(
            f"Metadata count      : "
            f"{metadata_count:,}"
        )

        print(
            f"Embedding dim       : "
            f"{f.attrs['embedding_dim']}"
        )

        print(
            f"Model               : "
            f"{f.attrs['model_name']}"
        )

        print(
            f"Pooling             : "
            f"{f.attrs['pooling']}"
        )

        print(
            f"Max AA/chunk        : "
            f"{f.attrs['max_aa_per_chunk']}"
        )

        print(
            f"Long sequence       : "
            f"{f.attrs['long_sequence_strategy']}"
        )

        print(
            f"Sequence key        : "
            f"{f.attrs['sequence_key']}"
        )

        print(
            f"Version             : "
            f"{f.attrs['version']}"
        )

        # Vector dimension QC
        wrong_dim = 0
        for key in f["vectors"].keys():
            shape = f["vectors"][key].shape
            if shape != (hidden_dim,):
                wrong_dim += 1

        print(
            f"Wrong vector shape  : "
            f"{wrong_dim:,}"
        )

        # Metadata / vector matching QC
        vector_keys = set(
            f["vectors"].keys()
        )
        metadata_keys = set(
            f["metadata"].keys()
        )

        missing_metadata = (
            vector_keys
            - metadata_keys
        )
        missing_vector = (
            metadata_keys
            - vector_keys
        )

        print(
            f"Missing metadata    : "
            f"{len(missing_metadata):,}"
        )

        print(
            f"Missing vector      : "
            f"{len(missing_vector):,}"
        )

    print(
        "\nDONE."
    )
    print(
        f"Output: {output_h5}"
    )


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    base_dir = (
        "/home/team5/workspace/sj/"
        "homo_protein_seq_fasta"
    )

    input_csv = os.path.join(
        base_dir,
        "esm2_ready_dataset_cropped_final.csv"
    )

    output_h5 = os.path.join(
        base_dir,
        "whole_human_protein_vector_"
        "esm2_t33_650M_1280d_sequence_v2.h5"
    )

    target_model = (
        "facebook/esm2_t33_650M_UR50D"
    )

    generate_embeddings(
        csv_path=input_csv,
        output_h5=output_h5,
        model_name=target_model,
        batch_size=4
    )