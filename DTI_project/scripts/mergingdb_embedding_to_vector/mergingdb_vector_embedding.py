import os
import gc
import time
import warnings

import numpy as np
import pandas as pd
import h5py
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 0. CONFIG
# ============================================================

BASE_DIR = "/home/team5/workspace/sj/homo_protein_seq_fasta"

INPUT_FILES = {
    "IC50": os.path.join(
        BASE_DIR,
        "IC50_ems2_fi.csv"
    ),
    "Ki": os.path.join(
        BASE_DIR,
        "Ki_ems2_fi.csv"
    ),
    "Kd": os.path.join(
        BASE_DIR,
        "Kd_ems2_fi.csv"
    ),
}

OUTPUT_FILES = {
    "IC50": os.path.join(
        BASE_DIR,
        "IC50_vectorDB.h5"
    ),
    "Ki": os.path.join(
        BASE_DIR,
        "Ki_vectorDB.h5"
    ),
    "Kd": os.path.join(
        BASE_DIR,
        "Kd_vectorDB.h5"
    ),
}


# ============================================================
# Models
# ============================================================

MOLFORMER_MODEL_NAME = (
    "ibm-research/MoLFormer-XL-both-10pct"
)

ESM_MODEL_NAME = (
    "facebook/esm2_t33_650M_UR50D"
)


# ============================================================
# Batch / HDF5 settings
# ============================================================

MOLFORMER_BATCH_SIZE = 64

# ESM-2 t33-650M
ESM_BATCH_SIZE = 8

INTERACTION_CHUNK_SIZE = 10_000

H5_COMPRESSION = "gzip"
H5_COMPRESSION_LEVEL = 4

EMBEDDING_DTYPE = np.float32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

warnings.filterwarnings("ignore")


# ============================================================
# Utility
# ============================================================

def print_separator(title=None):

    print()
    print("=" * 100)

    if title:
        print(title)

    print("=" * 100)


def get_file_size_gb(path):

    if not os.path.exists(path):
        return 0.0

    return os.path.getsize(
        path
    ) / (1024 ** 3)


def safe_str(x):

    if pd.isna(x):
        return ""

    return str(x)


def string_dtype(max_length):

    return h5py.string_dtype(
        encoding="utf-8",
        length=max_length
    )


def get_max_string_length(series):

    if len(series) == 0:
        return 1

    return max(
        1,
        int(
            series.astype(str)
            .str.len()
            .max()
        )
    )


# ============================================================
# GPU info
# ============================================================

def print_device_info():

    print_separator("DEVICE")

    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "CUDA available  :",
        torch.cuda.is_available()
    )

    print(
        "Device          :",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

        props = torch.cuda.get_device_properties(0)

        print(
            "GPU memory      : %.2f GB"
            % (
                props.total_memory
                / (1024 ** 3)
            )
        )


# ============================================================
# Load MoLFormer
# ============================================================

def load_molformer():

    print_separator(
        "LOAD MoLFormer-XL"
    )

    print(
        "Model:",
        MOLFORMER_MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MOLFORMER_MODEL_NAME,
        trust_remote_code=True
    )

    model = model.to(DEVICE)

    model.eval()

    hidden_size = model.config.hidden_size

    print(
        "MoLFormer hidden size:",
        hidden_size
    )

    return tokenizer, model, hidden_size


# ============================================================
# Load ESM-2
# ============================================================

def load_esm2():

    print_separator(
        "LOAD ESM-2"
    )

    print(
        "Model:",
        ESM_MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        ESM_MODEL_NAME
    )

    model = model.to(DEVICE)

    model.eval()

    hidden_size = model.config.hidden_size

    print(
        "ESM-2 hidden size:",
        hidden_size
    )

    if hidden_size != 1280:

        raise ValueError(
            f"Expected ESM-2 dimension 1280, "
            f"but got {hidden_size}"
        )

    return tokenizer, model, hidden_size


# ============================================================
# MoLFormer embedding
# ============================================================

@torch.no_grad()
def molformer_encode(
    smiles_list,
    tokenizer,
    model,
    batch_size
):

    vectors = []

    total = len(smiles_list)

    start_time = time.time()

    progress = tqdm(
        range(
            0,
            total,
            batch_size
        ),
        desc="MoLFormer embedding",
        unit="batch",
        dynamic_ncols=True
    )

    for start in progress:

        batch = smiles_list[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        outputs = model(
            **encoded
        )

        hidden = (
            outputs.last_hidden_state
        )

        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(-1)
            .float()
        )

        masked_hidden = (
            hidden * attention_mask
        )

        summed = masked_hidden.sum(
            dim=1
        )

        counts = (
            attention_mask.sum(
                dim=1
            )
            .clamp(min=1)
        )

        pooled = (
            summed / counts
        )

        pooled = (
            pooled
            .float()
            .cpu()
            .numpy()
        )

        vectors.append(
            pooled
        )

        processed = min(
            start + len(batch),
            total
        )

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total - processed
        )

        eta_seconds = (
            remaining / speed
            if speed > 0
            else 0
        )

        progress.set_postfix(
            processed=f"{processed:,}/{total:,}",
            speed=f"{speed:.2f}/s",
            ETA=time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    eta_seconds
                )
            )
        )

        del encoded
        del outputs
        del hidden
        del pooled

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(
        vectors,
        axis=0
    ).astype(
        EMBEDDING_DTYPE,
        copy=False
    )


# ============================================================
# ESM-2 embedding
# ============================================================

@torch.no_grad()
def esm2_encode(
    sequences,
    tokenizer,
    model,
    batch_size
):

    vectors = []

    total = len(sequences)

    start_time = time.time()

    progress = tqdm(
        range(
            0,
            total,
            batch_size
        ),
        desc="ESM-2 embedding",
        unit="batch",
        dynamic_ncols=True
    )

    for start in progress:

        batch = sequences[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=1026,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(DEVICE)
            for k, v in encoded.items()
        }

        outputs = model(
            **encoded
        )

        hidden = (
            outputs.last_hidden_state
        )

        batch_vectors = []

        for i, seq in enumerate(batch):

            seq_len = len(seq)

            # --------------------------------------------
            # ESM-2 token structure
            #
            # 0       = BOS
            # 1~N     = amino acids
            # N+1     = EOS
            #
            # amino acid tokens only
            # --------------------------------------------

            aa_hidden = hidden[
                i,
                1:seq_len + 1,
                :
            ]

            if aa_hidden.shape[0] == 0:

                raise ValueError(
                    "Empty amino-acid embedding: "
                    f"{seq[:50]}"
                )

            vector = (
                aa_hidden.mean(
                    dim=0
                )
            )

            batch_vectors.append(
                vector
            )

        batch_vectors = torch.stack(
            batch_vectors
        )

        batch_vectors = (
            batch_vectors
            .float()
            .cpu()
            .numpy()
        )

        vectors.append(
            batch_vectors
        )

        processed = min(
            start + len(batch),
            total
        )

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total - processed
        )

        eta_seconds = (
            remaining / speed
            if speed > 0
            else 0
        )

        progress.set_postfix(
            processed=f"{processed:,}/{total:,}",
            speed=f"{speed:.2f}/s",
            ETA=time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    eta_seconds
                )
            )
        )

        del encoded
        del outputs
        del hidden
        del batch_vectors

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(
        vectors,
        axis=0
    ).astype(
        EMBEDDING_DTYPE,
        copy=False
    )


# ============================================================
# Load CSV
# ============================================================

def load_dataset(path):

    print_separator(
        f"LOAD DATASET\n{path}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        "Rows    :",
        len(df)
    )

    print(
        "Columns :",
        len(df.columns)
    )

    required_columns = [
        "smiles",
        "uniprot_id",
        "target_name",
        "affinity_type",
        "affinity_value_mean",
        "paffinity",
        "protein_name",
        "protein_sequence",
        "sequence_length"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # protein sequence QC
    # --------------------------------------------------------

    missing_sequence = (
        df["protein_sequence"]
        .isna()
        .sum()
    )

    print(
        "Missing protein_sequence:",
        missing_sequence
    )

    if missing_sequence > 0:

        df = df[
            df["protein_sequence"].notna()
        ].copy()

    # --------------------------------------------------------
    # String cleanup
    # --------------------------------------------------------

    df["smiles"] = (
        df["smiles"]
        .astype(str)
        .str.strip()
    )

    df["uniprot_id"] = (
        df["uniprot_id"]
        .astype(str)
        .str.strip()
    )

    df["protein_sequence"] = (
        df["protein_sequence"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # affinity_type normalize
    #
    # ic50 -> IC50
    # ki   -> KI
    # kd   -> KD
    # --------------------------------------------------------

    df["affinity_type"] = (
        df["affinity_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Empty checks
    # --------------------------------------------------------

    empty_smiles = (
        df["smiles"].eq("")
    ).sum()

    empty_uniprot = (
        df["uniprot_id"].eq("")
    ).sum()

    empty_sequence = (
        df["protein_sequence"].eq("")
    ).sum()

    print(
        "Empty SMILES :",
        empty_smiles
    )

    print(
        "Empty UniProt:",
        empty_uniprot
    )

    print(
        "Empty sequence:",
        empty_sequence
    )

    if empty_smiles > 0:

        df = df[
            df["smiles"] != ""
        ].copy()

    if empty_uniprot > 0:

        df = df[
            df["uniprot_id"] != ""
        ].copy()

    if empty_sequence > 0:

        df = df[
            df["protein_sequence"] != ""
        ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    print(
        "Rows after QC:",
        len(df)
    )

    print(
        "Affinity types:",
        df["affinity_type"]
        .value_counts()
        .to_dict()
    )

    return df


# ============================================================
# Build unique drugs
# ============================================================

def build_unique_drugs(df):

    print_separator(
        "BUILD UNIQUE DRUGS"
    )

    unique_smiles = (
        df["smiles"]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    unique_drugs = pd.DataFrame({
        "smiles": unique_smiles
    })

    unique_drugs["drug_id"] = np.arange(
        len(unique_drugs),
        dtype=np.int64
    )

    print(
        "Total interaction rows:",
        f"{len(df):,}"
    )

    print(
        "Unique SMILES:",
        f"{len(unique_drugs):,}"
    )

    print(
        "Drug duplication ratio:",
        f"{len(df) / max(len(unique_drugs), 1):.2f}"
    )

    return unique_drugs


# ============================================================
# Build unique proteins
# ============================================================

def build_unique_proteins(df):

    print_separator(
        "BUILD UNIQUE PROTEINS"
    )

    sequence_counts = (
        df.groupby(
            "uniprot_id"
        )["protein_sequence"]
        .nunique()
    )

    inconsistent = (
        sequence_counts[
            sequence_counts > 1
        ]
    )

    if len(inconsistent) > 0:

        print(
            "ERROR: UniProt IDs with multiple sequences:",
            len(inconsistent)
        )

        print(
            inconsistent.head(20)
        )

        raise ValueError(
            "A UniProt ID maps to multiple "
            "protein sequences."
        )

    unique_proteins = (
        df[
            [
                "uniprot_id",
                "protein_sequence",
                "protein_name"
            ]
        ]
        .drop_duplicates(
            subset=["uniprot_id"]
        )
        .reset_index(drop=True)
    )

    unique_proteins["protein_id"] = np.arange(
        len(unique_proteins),
        dtype=np.int64
    )

    unique_proteins = unique_proteins[
        [
            "protein_id",
            "uniprot_id",
            "protein_sequence",
            "protein_name"
        ]
    ]

    print(
        "Unique UniProt:",
        f"{len(unique_proteins):,}"
    )

    lengths = (
        unique_proteins[
            "protein_sequence"
        ].str.len()
    )

    print(
        "Sequence length min:",
        lengths.min()
    )

    print(
        "Sequence length max:",
        lengths.max()
    )

    over_1024 = (
        lengths > 1024
    ).sum()

    print(
        "Sequence >1024:",
        over_1024
    )

    if over_1024 > 0:

        raise ValueError(
            "Protein sequence >1024 detected. "
            "Check ESM-2 preprocessing."
        )

    return unique_proteins


# ============================================================
# Create mapping
# ============================================================

def create_interaction_mapping(
    df,
    unique_drugs,
    unique_proteins
):

    print_separator(
        "CREATE INTERACTION MAPPING"
    )

    drug_map = dict(
        zip(
            unique_drugs["smiles"],
            unique_drugs["drug_id"]
        )
    )

    protein_map = dict(
        zip(
            unique_proteins["uniprot_id"],
            unique_proteins["protein_id"]
        )
    )

    drug_ids = (
        df["smiles"]
        .map(drug_map)
        .to_numpy(
            dtype=np.int64
        )
    )

    protein_ids = (
        df["uniprot_id"]
        .map(protein_map)
        .to_numpy(
            dtype=np.int64
        )
    )

    if np.any(drug_ids < 0):

        raise ValueError(
            "Invalid drug_id detected."
        )

    if np.any(protein_ids < 0):

        raise ValueError(
            "Invalid protein_id detected."
        )

    return drug_ids, protein_ids


# ============================================================
# HDF5 string writer
# ============================================================

def create_string_dataset(
    group,
    name,
    data,
    compression=H5_COMPRESSION,
    compression_opts=H5_COMPRESSION_LEVEL
):

    data = [
        safe_str(x)
        for x in data
    ]

    max_length = get_max_string_length(
        pd.Series(data)
    )

    dtype = string_dtype(
        max_length
    )

    ds = group.create_dataset(
        name,
        shape=(len(data),),
        dtype=dtype,
        chunks=True,
        compression=compression,
        compression_opts=compression_opts
    )

    ds[:] = np.asarray(
        data,
        dtype=f"S{max_length}"
    )

    return ds


# ============================================================
# HDF5 metadata
# ============================================================

def write_metadata(
    h5,
    df
):

    print(
        "[HDF5] Writing metadata..."
    )

    metadata = h5.create_group(
        "metadata"
    )

    string_columns = [
        "smiles",
        "uniprot_id",
        "target_name",
        "affinity_type",
        "protein_name"
    ]

    numeric_columns = [
        "affinity_value_mean",
        "paffinity",
        "sequence_length"
    ]

    for col in string_columns:

        create_string_dataset(
            metadata,
            col,
            df[col].tolist()
        )

    for col in numeric_columns:

        arr = df[col].to_numpy(
            dtype=np.float32
        )

        metadata.create_dataset(
            col,
            data=arr,
            chunks=True,
            compression=H5_COMPRESSION,
            compression_opts=H5_COMPRESSION_LEVEL
        )


# ============================================================
# Write unique drugs
# ============================================================

def write_unique_drugs(
    h5,
    unique_drugs,
    drug_vectors
):

    print(
        "[HDF5] Writing unique_drugs..."
    )

    group = h5.create_group(
        "unique_drugs"
    )

    create_string_dataset(
        group,
        "smiles",
        unique_drugs[
            "smiles"
        ].tolist()
    )

    group.create_dataset(
        "vector",
        data=drug_vectors,
        chunks=(
            min(
                1024,
                len(drug_vectors)
            ),
            drug_vectors.shape[1]
        ),
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    group.attrs[
        "count"
    ] = len(unique_drugs)

    group.attrs[
        "embedding_dimension"
    ] = drug_vectors.shape[1]


# ============================================================
# Write unique proteins
# ============================================================

def write_unique_proteins(
    h5,
    unique_proteins,
    protein_vectors
):

    print(
        "[HDF5] Writing unique_proteins..."
    )

    group = h5.create_group(
        "unique_proteins"
    )

    create_string_dataset(
        group,
        "uniprot_id",
        unique_proteins[
            "uniprot_id"
        ].tolist()
    )

    create_string_dataset(
        group,
        "protein_sequence",
        unique_proteins[
            "protein_sequence"
        ].tolist()
    )

    create_string_dataset(
        group,
        "protein_name",
        unique_proteins[
            "protein_name"
        ].tolist()
    )

    group.create_dataset(
        "vector",
        data=protein_vectors,
        chunks=(
            min(
                256,
                len(protein_vectors)
            ),
            protein_vectors.shape[1]
        ),
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    group.attrs[
        "count"
    ] = len(unique_proteins)

    group.attrs[
        "embedding_dimension"
    ] = protein_vectors.shape[1]


# ============================================================
# Create interaction HDF5 datasets
# ============================================================

def create_interaction_datasets(
    h5,
    n_rows,
    drug_dim,
    protein_dim
):

    group = h5.create_group(
        "interactions"
    )

    drug_vector_ds = group.create_dataset(
        "drug_vector",
        shape=(n_rows, drug_dim),
        dtype=EMBEDDING_DTYPE,
        chunks=(
            INTERACTION_CHUNK_SIZE,
            drug_dim
        ),
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    protein_vector_ds = group.create_dataset(
        "protein_vector",
        shape=(n_rows, protein_dim),
        dtype=EMBEDDING_DTYPE,
        chunks=(
            INTERACTION_CHUNK_SIZE,
            protein_dim
        ),
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    drug_id_ds = group.create_dataset(
        "drug_id",
        shape=(n_rows,),
        dtype=np.int64,
        chunks=True,
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    protein_id_ds = group.create_dataset(
        "protein_id",
        shape=(n_rows,),
        dtype=np.int64,
        chunks=True,
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    paffinity_ds = group.create_dataset(
        "paffinity",
        shape=(n_rows,),
        dtype=np.float32,
        chunks=True,
        compression=H5_COMPRESSION,
        compression_opts=H5_COMPRESSION_LEVEL
    )

    return (
        drug_vector_ds,
        protein_vector_ds,
        drug_id_ds,
        protein_id_ds,
        paffinity_ds
    )


# ============================================================
# Write interaction vectors
# ============================================================

def write_interactions(
    h5,
    df,
    drug_ids,
    protein_ids,
    drug_vectors,
    protein_vectors
):

    print_separator(
        "WRITE INTERACTION DATA"
    )

    n_rows = len(df)

    drug_dim = drug_vectors.shape[1]
    protein_dim = protein_vectors.shape[1]

    (
        drug_vector_ds,
        protein_vector_ds,
        drug_id_ds,
        protein_id_ds,
        paffinity_ds
    ) = create_interaction_datasets(
        h5,
        n_rows,
        drug_dim,
        protein_dim
    )

    start_time = time.time()

    progress = tqdm(
        range(
            0,
            n_rows,
            INTERACTION_CHUNK_SIZE
        ),
        desc="HDF5 interaction writing",
        unit="chunk",
        dynamic_ncols=True
    )

    for start in progress:

        end = min(
            start + INTERACTION_CHUNK_SIZE,
            n_rows
        )

        chunk_drug_ids = (
            drug_ids[start:end]
        )

        chunk_protein_ids = (
            protein_ids[start:end]
        )

        chunk_drug_vectors = (
            drug_vectors[
                chunk_drug_ids
            ]
        )

        chunk_protein_vectors = (
            protein_vectors[
                chunk_protein_ids
            ]
        )

        chunk_paffinity = (
            df["paffinity"]
            .iloc[start:end]
            .to_numpy(
                dtype=np.float32
            )
        )

        drug_vector_ds[
            start:end
        ] = chunk_drug_vectors

        protein_vector_ds[
            start:end
        ] = chunk_protein_vectors

        drug_id_ds[
            start:end
        ] = chunk_drug_ids

        protein_id_ds[
            start:end
        ] = chunk_protein_ids

        paffinity_ds[
            start:end
        ] = chunk_paffinity

        processed = end

        elapsed = (
            time.time()
            - start_time
        )

        speed = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            n_rows - processed
        )

        eta_seconds = (
            remaining / speed
            if speed > 0
            else 0
        )

        progress.set_postfix(
            processed=f"{processed:,}/{n_rows:,}",
            speed=f"{speed:,.0f} rows/s",
            ETA=time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    eta_seconds
                )
            )
        )

    group = h5[
        "interactions"
    ]

    group.attrs[
        "count"
    ] = n_rows

    group.attrs[
        "drug_embedding_dimension"
    ] = drug_dim

    group.attrs[
        "protein_embedding_dimension"
    ] = protein_dim


# ============================================================
# HDF5 attributes
# ============================================================

def write_global_attributes(
    h5,
    affinity_type,
    n_rows,
    n_drugs,
    n_proteins,
    drug_dim,
    protein_dim
):

    h5.attrs[
        "database_type"
    ] = "DTI_vectorDB"

    h5.attrs[
        "affinity_type"
    ] = affinity_type

    h5.attrs[
        "num_interactions"
    ] = n_rows

    h5.attrs[
        "num_unique_drugs"
    ] = n_drugs

    h5.attrs[
        "num_unique_proteins"
    ] = n_proteins

    h5.attrs[
        "drug_encoder"
    ] = MOLFORMER_MODEL_NAME

    h5.attrs[
        "protein_encoder"
    ] = ESM_MODEL_NAME

    h5.attrs[
        "drug_embedding_dimension"
    ] = drug_dim

    h5.attrs[
        "protein_embedding_dimension"
    ] = protein_dim

    h5.attrs[
        "embedding_dtype"
    ] = str(
        np.dtype(
            EMBEDDING_DTYPE
        )
    )

    h5.attrs[
        "hdf5_compression"
    ] = H5_COMPRESSION


# ============================================================
# HDF5 validation
# ============================================================

def validate_h5(
    output_path,
    expected_rows,
    expected_drugs,
    expected_proteins,
    drug_dim,
    protein_dim
):

    print_separator(
        "HDF5 VALIDATION"
    )

    with h5py.File(
        output_path,
        "r"
    ) as h5:

        print(
            "HDF5 file:",
            output_path
        )

        print(
            "File size: %.3f GB"
            % get_file_size_gb(
                output_path
            )
        )

        required_groups = [
            "unique_drugs",
            "unique_proteins",
            "interactions",
            "metadata"
        ]

        for group_name in required_groups:

            if group_name not in h5:

                raise ValueError(
                    f"Missing HDF5 group: "
                    f"{group_name}"
                )

        # ----------------------------------------------------
        # Unique drug
        # ----------------------------------------------------

        drug_vectors = h5[
            "unique_drugs/vector"
        ]

        if drug_vectors.shape != (
            expected_drugs,
            drug_dim
        ):

            raise ValueError(
                f"Unique drug vector shape mismatch: "
                f"{drug_vectors.shape}"
            )

        # ----------------------------------------------------
        # Unique protein
        # ----------------------------------------------------

        protein_vectors = h5[
            "unique_proteins/vector"
        ]

        if protein_vectors.shape != (
            expected_proteins,
            protein_dim
        ):

            raise ValueError(
                f"Unique protein vector shape mismatch: "
                f"{protein_vectors.shape}"
            )

        # ----------------------------------------------------
        # Interaction
        # ----------------------------------------------------

        interaction_drug = h5[
            "interactions/drug_vector"
        ]

        interaction_protein = h5[
            "interactions/protein_vector"
        ]

        drug_ids = h5[
            "interactions/drug_id"
        ]

        protein_ids = h5[
            "interactions/protein_id"
        ]

        paffinity = h5[
            "interactions/paffinity"
        ]

        if interaction_drug.shape != (
            expected_rows,
            drug_dim
        ):

            raise ValueError(
                "Interaction drug vector "
                "shape mismatch."
            )

        if interaction_protein.shape != (
            expected_rows,
            protein_dim
        ):

            raise ValueError(
                "Interaction protein vector "
                "shape mismatch."
            )

        if drug_ids.shape != (
            expected_rows,
        ):

            raise ValueError(
                "drug_id shape mismatch."
            )

        if protein_ids.shape != (
            expected_rows,
        ):

            raise ValueError(
                "protein_id shape mismatch."
            )

        if paffinity.shape != (
            expected_rows,
        ):

            raise ValueError(
                "pAffinity shape mismatch."
            )

        # ----------------------------------------------------
        # Full ID range check
        # ----------------------------------------------------

        print(
            "Checking ID ranges..."
        )

        min_drug_id = np.min(
            drug_ids[:]
        )

        max_drug_id = np.max(
            drug_ids[:]
        )

        min_protein_id = np.min(
            protein_ids[:]
        )

        max_protein_id = np.max(
            protein_ids[:]
        )

        if (
            min_drug_id < 0
            or
            max_drug_id >= expected_drugs
        ):

            raise ValueError(
                "Invalid drug_id range."
            )

        if (
            min_protein_id < 0
            or
            max_protein_id >= expected_proteins
        ):

            raise ValueError(
                "Invalid protein_id range."
            )

        # ----------------------------------------------------
        # Sample NaN check
        # ----------------------------------------------------

        print(
            "Checking NaN..."
        )

        sample_n = min(
            expected_rows,
            10000
        )

        drug_sample = interaction_drug[
            :sample_n
        ]

        protein_sample = interaction_protein[
            :sample_n
        ]

        paffinity_sample = paffinity[
            :sample_n
        ]

        if np.isnan(
            drug_sample
        ).any():

            raise ValueError(
                "NaN found in drug vectors."
            )

        if np.isnan(
            protein_sample
        ).any():

            raise ValueError(
                "NaN found in protein vectors."
            )

        if np.isnan(
            paffinity_sample
        ).any():

            raise ValueError(
                "NaN found in pAffinity."
            )

        metadata_rows = h5[
            "metadata/smiles"
        ].shape[0]

        if metadata_rows != expected_rows:

            raise ValueError(
                "Metadata row count mismatch."
            )

        print()
        print("Validation PASSED")
        print()
        print(
            "Interactions      :",
            f"{expected_rows:,}"
        )
        print(
            "Unique drugs      :",
            f"{expected_drugs:,}"
        )
        print(
            "Unique proteins   :",
            f"{expected_proteins:,}"
        )
        print(
            "Drug dimension    :",
            drug_dim
        )
        print(
            "Protein dimension :",
            protein_dim
        )


# ============================================================
# One affinity type
# ============================================================

def process_affinity_type(
    affinity_type,
    input_path,
    output_path,
    mol_tokenizer,
    mol_model,
    mol_dim,
    esm_tokenizer,
    esm_model,
    esm_dim
):

    print_separator(
        f"START {affinity_type}"
    )

    print(
        "Input :",
        input_path
    )

    print(
        "Output:",
        output_path
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Existing output
    # --------------------------------------------------------

    if os.path.exists(
        output_path
    ):

        print()
        print(
            "WARNING: output HDF5 already exists."
        )

        print(
            "It will be overwritten."
        )

        os.remove(
            output_path
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset(
        input_path
    )

    # --------------------------------------------------------
    # Verify affinity type
    # --------------------------------------------------------

    normalized_affinity = (
        affinity_type
        .upper()
    )

    affinity_types = (
        df["affinity_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    print(
        "Affinity types:",
        affinity_types
    )

    if normalized_affinity not in affinity_types:

        raise ValueError(
            f"{normalized_affinity} not found "
            f"in affinity_type column."
        )

    # --------------------------------------------------------
    # Build unique drugs
    # --------------------------------------------------------

    unique_drugs = build_unique_drugs(
        df
    )

    # --------------------------------------------------------
    # Build unique proteins
    # --------------------------------------------------------

    unique_proteins = build_unique_proteins(
        df
    )

    # --------------------------------------------------------
    # MoLFormer
    # --------------------------------------------------------

    print_separator(
        f"{affinity_type} - MoLFormer EMBEDDING"
    )

    drug_vectors = molformer_encode(
        unique_drugs[
            "smiles"
        ].tolist(),
        mol_tokenizer,
        mol_model,
        MOLFORMER_BATCH_SIZE
    )

    print(
        "Drug vector shape:",
        drug_vectors.shape
    )

    # --------------------------------------------------------
    # ESM-2
    # --------------------------------------------------------

    print_separator(
        f"{affinity_type} - ESM-2 EMBEDDING"
    )

    protein_vectors = esm2_encode(
        unique_proteins[
            "protein_sequence"
        ].tolist(),
        esm_tokenizer,
        esm_model,
        ESM_BATCH_SIZE
    )

    print(
        "Protein vector shape:",
        protein_vectors.shape
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if drug_vectors.shape[0] != len(
        unique_drugs
    ):

        raise ValueError(
            "Drug embedding count mismatch."
        )

    if protein_vectors.shape[0] != len(
        unique_proteins
    ):

        raise ValueError(
            "Protein embedding count mismatch."
        )

    if drug_vectors.shape[1] != mol_dim:

        raise ValueError(
            "Unexpected MoLFormer dimension."
        )

    if protein_vectors.shape[1] != esm_dim:

        raise ValueError(
            "Unexpected ESM-2 dimension."
        )

    # --------------------------------------------------------
    # Interaction mapping
    # --------------------------------------------------------

    drug_ids, protein_ids = (
        create_interaction_mapping(
            df,
            unique_drugs,
            unique_proteins
        )
    )

    # --------------------------------------------------------
    # HDF5
    # --------------------------------------------------------

    print_separator(
        f"CREATE HDF5 - {affinity_type}"
    )

    with h5py.File(
        output_path,
        "w"
    ) as h5:

        write_global_attributes(
            h5,
            affinity_type,
            len(df),
            len(unique_drugs),
            len(unique_proteins),
            mol_dim,
            esm_dim
        )

        write_unique_drugs(
            h5,
            unique_drugs,
            drug_vectors
        )

        write_unique_proteins(
            h5,
            unique_proteins,
            protein_vectors
        )

        write_interactions(
            h5,
            df,
            drug_ids,
            protein_ids,
            drug_vectors,
            protein_vectors
        )

        write_metadata(
            h5,
            df
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_h5(
        output_path,
        len(df),
        len(unique_drugs),
        len(unique_proteins),
        mol_dim,
        esm_dim
    )

    elapsed = (
        time.time()
        - start_time
    )

    print_separator(
        f"{affinity_type} COMPLETE"
    )

    print(
        "Output:",
        output_path
    )

    print(
        "File size: %.3f GB"
        % get_file_size_gb(
            output_path
        )
    )

    print(
        "Rows:",
        f"{len(df):,}"
    )

    print(
        "Unique drugs:",
        f"{len(unique_drugs):,}"
    )

    print(
        "Unique proteins:",
        f"{len(unique_proteins):,}"
    )

    print(
        "MoLFormer dimension:",
        mol_dim
    )

    print(
        "ESM-2 dimension:",
        esm_dim
    )

    print(
        "Elapsed:",
        time.strftime(
            "%H:%M:%S",
            time.gmtime(
                elapsed
            )
        )
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del df
    del unique_drugs
    del unique_proteins
    del drug_vectors
    del protein_vectors
    del drug_ids
    del protein_ids

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return elapsed


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print_separator(
        "DTI VECTOR DATABASE GENERATION"
    )

    print(
        "Base directory:"
    )

    print(
        BASE_DIR
    )

    print()

    print(
        "SMILES encoder:"
    )

    print(
        MOLFORMER_MODEL_NAME
    )

    print()

    print(
        "Protein encoder:"
    )

    print(
        ESM_MODEL_NAME
    )

    print_device_info()

    # --------------------------------------------------------
    # Input file check
    # --------------------------------------------------------

    print_separator(
        "INPUT FILE CHECK"
    )

    for affinity_type, path in INPUT_FILES.items():

        exists = os.path.exists(
            path
        )

        print(
            f"{affinity_type:5s}: "
            f"{path} "
            f"[{'OK' if exists else 'MISSING'}]"
        )

        if not exists:

            raise FileNotFoundError(
                f"Missing input file:\n{path}"
            )

    # --------------------------------------------------------
    # Load models ONCE
    # --------------------------------------------------------

    mol_tokenizer, mol_model, mol_dim = (
        load_molformer()
    )

    esm_tokenizer, esm_model, esm_dim = (
        load_esm2()
    )

    # --------------------------------------------------------
    # Overall progress
    # --------------------------------------------------------

    affinity_times = []

    overall_progress = tqdm(
        total=3,
        desc="OVERALL IC50/Ki/Kd",
        unit="dataset",
        dynamic_ncols=True
    )

    # ========================================================
    # IC50
    # ========================================================

    elapsed = process_affinity_type(
        affinity_type="IC50",
        input_path=INPUT_FILES["IC50"],
        output_path=OUTPUT_FILES["IC50"],
        mol_tokenizer=mol_tokenizer,
        mol_model=mol_model,
        mol_dim=mol_dim,
        esm_tokenizer=esm_tokenizer,
        esm_model=esm_model,
        esm_dim=esm_dim
    )

    affinity_times.append(
        elapsed
    )

    overall_progress.update(1)

    if len(affinity_times) > 0:

        avg_time = np.mean(
            affinity_times
        )

        remaining = (
            3
            - len(affinity_times)
        )

        overall_progress.set_postfix(
            completed="IC50",
            ETA=time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    avg_time * remaining
                )
            )
        )

    # ========================================================
    # Ki
    # ========================================================

    elapsed = process_affinity_type(
        affinity_type="Ki",
        input_path=INPUT_FILES["Ki"],
        output_path=OUTPUT_FILES["Ki"],
        mol_tokenizer=mol_tokenizer,
        mol_model=mol_model,
        mol_dim=mol_dim,
        esm_tokenizer=esm_tokenizer,
        esm_model=esm_model,
        esm_dim=esm_dim
    )

    affinity_times.append(
        elapsed
    )

    overall_progress.update(1)

    avg_time = np.mean(
        affinity_times
    )

    remaining = (
        3
        - len(affinity_times)
    )

    overall_progress.set_postfix(
        completed="IC50 + Ki",
        ETA=time.strftime(
            "%H:%M:%S",
            time.gmtime(
                max(
                    avg_time * remaining,
                    0
                )
            )
        )
    )

    # ========================================================
    # Kd
    # ========================================================

    elapsed = process_affinity_type(
        affinity_type="Kd",
        input_path=INPUT_FILES["Kd"],
        output_path=OUTPUT_FILES["Kd"],
        mol_tokenizer=mol_tokenizer,
        mol_model=mol_model,
        mol_dim=mol_dim,
        esm_tokenizer=esm_tokenizer,
        esm_model=esm_model,
        esm_dim=esm_dim
    )

    affinity_times.append(
        elapsed
    )

    overall_progress.update(1)

    overall_progress.set_postfix(
        completed="IC50 + Ki + Kd",
        ETA="00:00:00"
    )

    overall_progress.close()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    total_elapsed = (
        time.time()
        - total_start
    )

    print_separator(
        "ALL VECTOR DATABASES COMPLETE"
    )

    for affinity_type, path in OUTPUT_FILES.items():

        print(
            f"{affinity_type}:"
        )

        print(
            path
        )

        print(
            "Size: %.3f GB"
            % get_file_size_gb(
                path
            )
        )

        print()

    print(
        "Total elapsed:",
        time.strftime(
            "%H:%M:%S",
            time.gmtime(
                total_elapsed
            )
        )
    )

    print()
    print(
        "ALL DONE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()