import os
import json
import hashlib
import h5py
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1. Original CSV
# ------------------------------------------------------------
CSV_PATH = (
    r"C:\workspace\python\project_personal\DTI_project\data"
    r"\esm2_human_whole_protein_vecterDB\homo_protein_seq_fasta"
    r"\esm2_ready_dataset_cropped_final.csv"
)

# ------------------------------------------------------------
# 2. Current H5
# ------------------------------------------------------------
H5_PATH = (
    r"C:\workspace\python\project_personal\DTI_project\data"
    r"\esm2_human_whole_protein_vecterDB"
    r"\whole_human_protein_vector_esm2_t33_650M_1280d_sequence_v2.H5"
)

# ------------------------------------------------------------
# 3. Output directory
# ------------------------------------------------------------
OUTPUT_DIR = (
    r"C:\workspace\python\project_personal\DTI_project\data"
    r"\esm2_human_whole_protein_vecterDB"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

CSV_H5_MAPPING = os.path.join(
    OUTPUT_DIR,
    "QC_CSV_vs_H5_mapping.csv"
)

CSV_ID_STATUS = os.path.join(
    OUTPUT_DIR,
    "QC_CSV_id_H5_status.csv"
)

CSV_UNIPROT_MULTIPLE_ID = os.path.join(
    OUTPUT_DIR,
    "QC_Uniprot_multiple_ID_analysis.csv"
)

CSV_H5_METADATA_FIELDS = os.path.join(
    OUTPUT_DIR,
    "QC_H5_metadata_fields.csv"
)

CSV_HASH_MATCH = os.path.join(
    OUTPUT_DIR,
    "QC_sequence_hash_matching.csv"
)

TXT_REPORT = os.path.join(
    OUTPUT_DIR,
    "QC_CSV_vs_H5_report.txt"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_string(x):
    """
    Normalize metadata values to Python string.
    """
    if x is None:
        return None

    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)

    if isinstance(x, np.bytes_):
        try:
            return x.tobytes().decode("utf-8")
        except Exception:
            return str(x)

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return normalize_string(x.item())

        return json.dumps(
            x.tolist(),
            ensure_ascii=False
        )

    return str(x)


def try_parse_json(value):
    """
    Try to parse JSON metadata.
    """
    value = normalize_string(value)

    if value is None:
        return None

    try:
        return json.loads(value)
    except Exception:
        return None


def recursive_find_id_fields(obj, path="", results=None):
    """
    Recursively search metadata for fields whose name contains 'id'.
    """
    if results is None:
        results = []

    if isinstance(obj, dict):
        for key, value in obj.items():

            current_path = (
                f"{path}/{key}"
                if path
                else str(key)
            )

            key_lower = str(key).lower()

            if "id" in key_lower:
                results.append(
                    (
                        current_path,
                        value
                    )
                )

            recursive_find_id_fields(
                value,
                current_path,
                results
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            current_path = f"{path}[{i}]"

            recursive_find_id_fields(
                value,
                current_path,
                results
            )

    return results


def sha256_sequence(sequence):
    if sequence is None:
        return None

    sequence = str(sequence).strip().upper()

    return hashlib.sha256(
        sequence.encode("utf-8")
    ).hexdigest()


def md5_sequence(sequence):
    if sequence is None:
        return None

    sequence = str(sequence).strip().upper()

    return hashlib.md5(
        sequence.encode("utf-8")
    ).hexdigest()


def collect_h5_structure(h5_file):

    rows = []

    def visitor(name, obj):

        if isinstance(obj, h5py.Group):

            rows.append({
                "path": name,
                "type": "GROUP",
                "shape": "",
                "dtype": "",
                "attrs": json.dumps(
                    {
                        str(k): normalize_string(v)
                        for k, v in obj.attrs.items()
                    },
                    ensure_ascii=False
                )
            })

        elif isinstance(obj, h5py.Dataset):

            rows.append({
                "path": name,
                "type": "DATASET",
                "shape": str(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": json.dumps(
                    {
                        str(k): normalize_string(v)
                        for k, v in obj.attrs.items()
                    },
                    ensure_ascii=False
                )
            })

    h5_file.visititems(visitor)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("CSV ↔ H5 METADATA 1:1 QC")
    print("=" * 100)

    print()
    print("CSV:")
    print(CSV_PATH)

    print()
    print("H5:")
    print(H5_PATH)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    if not os.path.exists(H5_PATH):
        raise FileNotFoundError(
            f"H5 not found:\n{H5_PATH}"
        )

    # ========================================================
    # 1. LOAD CSV
    # ========================================================

    print()
    print("=" * 100)
    print("1. LOAD ORIGINAL CSV")
    print("=" * 100)

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False
    )

    print(f"Rows : {len(df):,}")
    print(f"Cols : {len(df.columns)}")

    print()
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    required_columns = [
        "id",
        "sequence",
        "length",
        "type",
        "raw_header",
        "uniprot_id"
    ]

    print()
    print("Required column check:")

    for col in required_columns:

        if col in df.columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [MISSING] {col}")

    # ========================================================
    # 2. CSV ID QC
    # ========================================================

    print()
    print("=" * 100)
    print("2. ORIGINAL CSV ID QC")
    print("=" * 100)

    csv_id_count = df["id"].notna().sum()

    csv_unique_id = df["id"].nunique(
        dropna=True
    )

    csv_duplicate_id = (
        df["id"].duplicated(
            keep=False
        ).sum()
    )

    print(f"Total rows             : {len(df):,}")
    print(f"Non-null id            : {csv_id_count:,}")
    print(f"Unique id              : {csv_unique_id:,}")
    print(f"Duplicated id rows     : {csv_duplicate_id:,}")

    print()
    print("Sample IDs:")

    print(
        df[
            [
                "id",
                "uniprot_id",
                "type",
                "length"
            ]
        ].head(10).to_string(index=False)
    )

    # ========================================================
    # 3. ADD SEQUENCE HASH TO CSV
    # ========================================================

    print()
    print("=" * 100)
    print("3. CALCULATE CSV SEQUENCE HASH")
    print("=" * 100)

    df["csv_sha256"] = df["sequence"].apply(
        sha256_sequence
    )

    df["csv_md5"] = df["sequence"].apply(
        md5_sequence
    )

    print(
        f"Unique CSV SHA256 : "
        f"{df['csv_sha256'].nunique():,}"
    )

    print(
        f"Unique CSV MD5    : "
        f"{df['csv_md5'].nunique():,}"
    )

    # ========================================================
    # 4. OPEN H5
    # ========================================================

    print()
    print("=" * 100)
    print("4. OPEN H5")
    print("=" * 100)

    with h5py.File(
        H5_PATH,
        "r"
    ) as h5:

        print("Root groups/datasets:")

        for key in h5.keys():
            obj = h5[key]

            print(
                f"  {key} -> "
                f"{type(obj).__name__}"
            )

        print()
        print("Root attributes:")

        for key, value in h5.attrs.items():

            print(
                f"  {key} = "
                f"{normalize_string(value)}"
            )

        # ====================================================
        # 5. FULL H5 STRUCTURE
        # ====================================================

        print()
        print("=" * 100)
        print("5. FULL H5 STRUCTURE")
        print("=" * 100)

        structure_df = collect_h5_structure(h5)

        print(
            structure_df.head(100).to_string(
                index=False
            )
        )

        structure_df.to_csv(
            CSV_H5_METADATA_FIELDS,
            index=False,
            encoding="utf-8-sig"
        )

        print()
        print(
            f"Saved structure CSV:\n"
            f"{CSV_H5_METADATA_FIELDS}"
        )

        # ====================================================
        # 6. FIND VECTORS
        # ====================================================

        print()
        print("=" * 100)
        print("6. VECTOR GROUP")
        print("=" * 100)

        if "vectors" not in h5:

            print(
                "[WARNING] H5 has no 'vectors' group."
            )

            vector_keys = []

        else:

            vector_group = h5["vectors"]

            vector_keys = list(
                vector_group.keys()
            )

            print(
                f"Number of vectors : "
                f"{len(vector_keys):,}"
            )

            if vector_keys:

                first_key = vector_keys[0]

                ds = vector_group[first_key]

                print(
                    f"First vector key : {first_key}"
                )

                print(
                    f"Shape            : {ds.shape}"
                )

                print(
                    f"Dtype            : {ds.dtype}"
                )

        # ====================================================
        # 7. READ H5 METADATA
        # ====================================================

        print()
        print("=" * 100)
        print("7. READ H5 METADATA")
        print("=" * 100)

        if "metadata" not in h5:

            print(
                "[ERROR] H5 has no 'metadata' group."
            )

            metadata_keys = []

        else:

            metadata_group = h5["metadata"]

            metadata_keys = list(
                metadata_group.keys()
            )

            print(
                f"Metadata entries : "
                f"{len(metadata_keys):,}"
            )

        # ====================================================
        # 8. EXTRACT H5 METADATA
        # ====================================================

        h5_records = []

        h5_id_locations = []

        for idx, key in enumerate(
            metadata_keys
        ):

            group = metadata_group[key]

            record = {
                "sequence_hash": key
            }

            # ----------------------------------------------
            # group attributes
            # ----------------------------------------------

            for attr_key, attr_value in group.attrs.items():

                record[
                    f"attr__{attr_key}"
                ] = normalize_string(
                    attr_value
                )

            # ----------------------------------------------
            # child datasets
            # ----------------------------------------------

            for child_name in group.keys():

                obj = group[child_name]

                if isinstance(
                    obj,
                    h5py.Dataset
                ):

                    try:

                        value = obj[()]

                        value = normalize_string(
                            value
                        )

                    except Exception as e:

                        value = (
                            f"<READ_ERROR: {e}>"
                        )

                    record[
                        child_name
                    ] = value

                    # Search any ID-like field
                    if "id" in child_name.lower():

                        h5_id_locations.append({
                            "sequence_hash": key,
                            "field": child_name,
                            "value": value
                        })

                elif isinstance(
                    obj,
                    h5py.Group
                ):

                    # Store nested group name
                    record[
                        f"group__{child_name}"
                    ] = "<GROUP>"

            h5_records.append(record)

        h5_meta_df = pd.DataFrame(
            h5_records
        )

        print()
        print("Detected H5 metadata columns:")

        for col in h5_meta_df.columns:

            print(f"  - {col}")

        # ====================================================
        # 9. CHECK WHETHER 'id' EXISTS
        # ====================================================

        print()
        print("=" * 100)
        print("8. DOES H5 METADATA CONTAIN 'id'?")
        print("=" * 100)

        id_columns = [
            col
            for col in h5_meta_df.columns
            if col.lower() == "id"
        ]

        id_like_columns = [
            col
            for col in h5_meta_df.columns
            if "id" in col.lower()
        ]

        if id_columns:

            print(
                "[FOUND] Exact 'id' field exists."
            )

            print(
                f"Columns: {id_columns}"
            )

        else:

            print(
                "[NOT FOUND] Exact 'id' field "
                "does NOT exist in H5 metadata."
            )

        print()
        print("All ID-like fields:")

        for col in id_like_columns:

            print(
                f"  - {col}"
            )

        # ====================================================
        # 10. SEARCH RAW H5 METADATA FOR CSV IDs
        # ====================================================

        print()
        print("=" * 100)
        print("9. SEARCH CSV 'id' VALUES INSIDE H5")
        print("=" * 100)

        csv_ids = set(
            df["id"]
            .dropna()
            .astype(str)
        )

        h5_string_values = set()

        for col in h5_meta_df.columns:

            if col == "sequence_hash":
                continue

            for value in h5_meta_df[col].dropna():

                value = normalize_string(
                    value
                )

                if value is not None:

                    h5_string_values.add(
                        value
                    )

        found_ids = []
        missing_ids = []

        for csv_id in csv_ids:

            found = False

            for value in h5_string_values:

                if csv_id == value:

                    found = True
                    break

                # JSON encoded list
                if csv_id in value:

                    found = True
                    break

            if found:
                found_ids.append(csv_id)
            else:
                missing_ids.append(csv_id)

        print(
            f"CSV unique IDs        : "
            f"{len(csv_ids):,}"
        )

        print(
            f"Found somewhere in H5 : "
            f"{len(found_ids):,}"
        )

        print(
            f"Not found in H5       : "
            f"{len(missing_ids):,}"
        )

        if len(csv_ids) > 0:

            print(
                f"ID presence rate      : "
                f"{len(found_ids) / len(csv_ids) * 100:.4f}%"
            )

        # ====================================================
        # 11. EXTRACT H5 UNIPROT IDS
        # ====================================================

        print()
        print("=" * 100)
        print("10. H5 UNIPROT_ID MAPPING")
        print("=" * 100)

        h5_uniprot_mapping = {}

        if "uniprot_ids" in h5_meta_df.columns:

            for _, row in h5_meta_df.iterrows():

                sequence_hash = row[
                    "sequence_hash"
                ]

                raw_value = row[
                    "uniprot_ids"
                ]

                parsed = try_parse_json(
                    raw_value
                )

                if isinstance(
                    parsed,
                    list
                ):

                    ids = [
                        str(x)
                        for x in parsed
                    ]

                elif parsed is not None:

                    ids = [
                        str(parsed)
                    ]

                else:

                    ids = [
                        str(raw_value)
                    ]

                h5_uniprot_mapping[
                    sequence_hash
                ] = ids

        else:

            print(
                "[WARNING] 'uniprot_ids' "
                "field not found."
            )

        print(
            f"H5 sequence hashes with UniProt mapping: "
            f"{len(h5_uniprot_mapping):,}"
        )

        # ====================================================
        # 12. BUILD REVERSE H5 MAPPING
        # ====================================================

        uniprot_to_hashes = defaultdict(list)

        for sequence_hash, ids in (
            h5_uniprot_mapping.items()
        ):

            for uid in ids:

                uniprot_to_hashes[
                    uid
                ].append(sequence_hash)

        print()
        print(
            "Top UniProt IDs by number of sequence hashes:"
        )

        top_uniprot = sorted(
            uniprot_to_hashes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:20]

        for uid, hashes in top_uniprot:

            print(
                f"  {uid}: "
                f"{len(hashes):,} sequence hashes"
            )

        # ====================================================
        # 13. CSV UNIPROT -> MULTIPLE ID
        # ====================================================

        print()
        print("=" * 100)
        print("11. CSV UniProt -> MULTIPLE id ANALYSIS")
        print("=" * 100)

        csv_uniprot_to_ids = (
            df.groupby(
                "uniprot_id"
            )["id"]
            .apply(
                lambda x: sorted(
                    set(
                        x.dropna()
                        .astype(str)
                    )
                )
            )
            .to_dict()
        )

        multi_id_rows = []

        for uid, ids in csv_uniprot_to_ids.items():

            if len(ids) > 1:

                multi_id_rows.append({
                    "uniprot_id": uid,
                    "num_ids": len(ids),
                    "ids": json.dumps(
                        ids,
                        ensure_ascii=False
                    )
                })

        multi_id_df = pd.DataFrame(
            multi_id_rows
        )

        print(
            f"UniProt IDs with multiple CSV ids: "
            f"{len(multi_id_df):,}"
        )

        if len(multi_id_df) > 0:

            print()
            print(
                multi_id_df
                .sort_values(
                    "num_ids",
                    ascending=False
                )
                .head(20)
                .to_string(
                    index=False
                )
            )

        multi_id_df.to_csv(
            CSV_UNIPROT_MULTIPLE_ID,
            index=False,
            encoding="utf-8-sig"
        )

        # ====================================================
        # 14. 1:1 CSV ROW -> H5 HASH MATCH
        # ====================================================

        print()
        print("=" * 100)
        print("12. CSV ROW ↔ H5 SEQUENCE HASH MATCH")
        print("=" * 100)

        h5_hash_set = set(
            metadata_keys
        )

        # Candidate hash algorithms
        sha256_set = set(
            df["csv_sha256"]
            .dropna()
        )

        md5_set = set(
            df["csv_md5"]
            .dropna()
        )

        sha256_direct_match = (
            len(
                sha256_set.intersection(
                    h5_hash_set
                )
            )
        )

        md5_direct_match = (
            len(
                md5_set.intersection(
                    h5_hash_set
                )
            )
        )

        print(
            f"CSV SHA256 unique hashes : "
            f"{len(sha256_set):,}"
        )

        print(
            f"H5 sequence hashes       : "
            f"{len(h5_hash_set):,}"
        )

        print()
        print(
            f"SHA256 direct H5 matches : "
            f"{sha256_direct_match:,}"
        )

        print(
            f"MD5 direct H5 matches    : "
            f"{md5_direct_match:,}"
        )

        # ====================================================
        # 15. BUILD CSV ↔ H5 MAPPING
        # ====================================================

        print()
        print("=" * 100)
        print("13. BUILD CSV ↔ H5 MAPPING TABLE")
        print("=" * 100)

        mapping_rows = []

        # Map sequence hash by SHA256
        h5_by_sha256 = {}

        for h in h5_hash_set:

            h5_by_sha256[h] = h

        for _, row in df.iterrows():

            csv_id = str(
                row["id"]
            )

            csv_uid = str(
                row["uniprot_id"]
            )

            sequence = str(
                row["sequence"]
            )

            sha = row["csv_sha256"]

            md5 = row["csv_md5"]

            # --------------------------------------------
            # Direct SHA256
            # --------------------------------------------

            h5_hash = None

            if sha in h5_hash_set:

                h5_hash = sha

            # --------------------------------------------
            # Direct MD5
            # --------------------------------------------

            elif md5 in h5_hash_set:

                h5_hash = md5

            # --------------------------------------------
            # If hash does not match, try sequence
            # through H5 metadata if available.
            # --------------------------------------------

            status = "NO_HASH_MATCH"

            h5_uniprots = []

            if h5_hash is not None:

                status = "HASH_MATCH"

                h5_uniprots = (
                    h5_uniprot_mapping
                    .get(
                        h5_hash,
                        []
                    )
                )

            mapping_rows.append({

                "csv_id": csv_id,

                "csv_uniprot_id": csv_uid,

                "csv_type": row.get(
                    "type",
                    ""
                ),

                "csv_length": row.get(
                    "length",
                    ""
                ),

                "csv_sha256": sha,

                "csv_md5": md5,

                "h5_sequence_hash": (
                    h5_hash
                ),

                "h5_uniprot_ids": json.dumps(
                    h5_uniprots,
                    ensure_ascii=False
                ),

                "uniprot_match": (
                    csv_uid in h5_uniprots
                ),

                "mapping_status": status

            })

        mapping_df = pd.DataFrame(
            mapping_rows
        )

        mapping_df.to_csv(
            CSV_H5_MAPPING,
            index=False,
            encoding="utf-8-sig"
        )

        # ====================================================
        # 16. MAPPING STATISTICS
        # ====================================================

        print()
        print("=" * 100)
        print("14. MAPPING STATISTICS")
        print("=" * 100)

        total_rows = len(mapping_df)

        hash_match_count = (
            mapping_df[
                "mapping_status"
            ]
            .eq("HASH_MATCH")
            .sum()
        )

        uniprot_match_count = (
            mapping_df[
                "uniprot_match"
            ]
            .eq(True)
            .sum()
        )

        no_match_count = (
            mapping_df[
                "mapping_status"
            ]
            .eq("NO_HASH_MATCH")
            .sum()
        )

        print(
            f"CSV rows                    : "
            f"{total_rows:,}"
        )

        print(
            f"Sequence hash matched       : "
            f"{hash_match_count:,}"
        )

        print(
            f"Sequence hash match rate    : "
            f"{hash_match_count / total_rows * 100:.4f}%"
        )

        print(
            f"UniProt also matched        : "
            f"{uniprot_match_count:,}"
        )

        print(
            f"UniProt match rate          : "
            f"{uniprot_match_count / total_rows * 100:.4f}%"
        )

        print(
            f"No sequence hash match      : "
            f"{no_match_count:,}"
        )

        # ====================================================
        # 17. ID STATUS
        # ====================================================

        print()
        print("=" * 100)
        print("15. ID LOSS DIAGNOSIS")
        print("=" * 100)

        id_status_rows = []

        # H5 exact id field?
        has_exact_id_field = (
            len(id_columns) > 0
        )

        # CSV IDs found anywhere?
        found_id_set = set(
            found_ids
        )

        for _, row in df.iterrows():

            csv_id = str(
                row["id"]
            )

            if has_exact_id_field:

                status = (
                    "H5_HAS_EXACT_ID_FIELD"
                )

            elif csv_id in found_id_set:

                status = (
                    "ID_EXISTS_SOMEWHERE_IN_H5"
                )

            else:

                status = (
                    "ID_NOT_FOUND_IN_H5"
                )

            id_status_rows.append({

                "id": csv_id,

                "uniprot_id": str(
                    row["uniprot_id"]
                ),

                "type": str(
                    row.get(
                        "type",
                        ""
                    )
                ),

                "length": row.get(
                    "length",
                    ""
                ),

                "id_status": status

            })

        id_status_df = pd.DataFrame(
            id_status_rows
        )

        id_status_df.to_csv(
            CSV_ID_STATUS,
            index=False,
            encoding="utf-8-sig"
        )

        # ====================================================
        # 18. FINAL DIAGNOSIS
        # ====================================================

        print()
        print("=" * 100)
        print("16. FINAL DIAGNOSIS")
        print("=" * 100)

        if has_exact_id_field:

            diagnosis = (
                "H5 metadata contains an exact 'id' field."
            )

        elif len(found_ids) > 0:

            diagnosis = (
                "The CSV 'id' values exist somewhere "
                "inside H5 metadata, but not as an exact "
                "'id' field."
            )

        else:

            diagnosis = (
                "The CSV 'id' values were not found "
                "inside H5 metadata."
            )

        print()
        print(diagnosis)

        print()

        if not has_exact_id_field:

            print(
                "Therefore, if the H5 was generated directly "
                "from this CSV, the original 'id' column was "
                "not preserved as a dedicated H5 metadata field."
            )

        print()
        print(
            "Important:"
        )

        print(
            "The H5 can still preserve the protein sequence "
            "through sequence_hash even if the original CSV "
            "'id' was discarded."
        )

        print(
            "Therefore, 'id loss' and 'sequence loss' are "
            "separate issues."
        )

        # ====================================================
        # 19. SAVE REPORT
        # ====================================================

        report_lines = []

        report_lines.append(
            "CSV ↔ H5 METADATA QC REPORT"
        )

        report_lines.append(
            "=" * 80
        )

        report_lines.append(
            f"CSV: {CSV_PATH}"
        )

        report_lines.append(
            f"H5 : {H5_PATH}"
        )

        report_lines.append("")

        report_lines.append(
            f"CSV rows: {len(df):,}"
        )

        report_lines.append(
            f"CSV unique IDs: {csv_unique_id:,}"
        )

        report_lines.append(
            f"H5 metadata entries: {len(metadata_keys):,}"
        )

        report_lines.append("")

        report_lines.append(
            f"H5 exact 'id' field: "
            f"{has_exact_id_field}"
        )

        report_lines.append(
            f"CSV IDs found somewhere in H5: "
            f"{len(found_ids):,}"
        )

        report_lines.append(
            f"CSV IDs missing from H5: "
            f"{len(missing_ids):,}"
        )

        report_lines.append("")

        report_lines.append(
            f"Sequence hash match: "
            f"{hash_match_count:,} / {total_rows:,}"
        )

        report_lines.append(
            f"UniProt match: "
            f"{uniprot_match_count:,} / {total_rows:,}"
        )

        report_lines.append("")

        report_lines.append(
            "FINAL DIAGNOSIS"
        )

        report_lines.append(
            diagnosis
        )

        report_lines.append("")

        report_lines.append(
            "OUTPUT FILES"
        )

        report_lines.append(
            CSV_H5_MAPPING
        )

        report_lines.append(
            CSV_ID_STATUS
        )

        report_lines.append(
            CSV_UNIPROT_MULTIPLE_ID
        )

        report_lines.append(
            CSV_H5_METADATA_FIELDS
        )

        report_lines.append(
            CSV_HASH_MATCH
        )

        with open(
            TXT_REPORT,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(
                    report_lines
                )
            )

        print()
        print("=" * 100)
        print("17. OUTPUT")
        print("=" * 100)

        print(
            f"[1] CSV ↔ H5 mapping:"
        )
        print(
            f"    {CSV_H5_MAPPING}"
        )

        print(
            f"[2] ID status:"
        )
        print(
            f"    {CSV_ID_STATUS}"
        )

        print(
            f"[3] UniProt multiple-ID:"
        )
        print(
            f"    {CSV_UNIPROT_MULTIPLE_ID}"
        )

        print(
            f"[4] H5 metadata structure:"
        )
        print(
            f"    {CSV_H5_METADATA_FIELDS}"
        )

        print(
            f"[5] Report:"
        )
        print(
            f"    {TXT_REPORT}"
        )

        print()
        print("=" * 100)
        print("QC COMPLETE")
        print("=" * 100)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()