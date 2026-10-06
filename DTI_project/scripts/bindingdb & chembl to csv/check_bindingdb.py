import pandas as pd

BINDINGDB_TSV = "../data/bindingdb/BindingDB_All.tsv"

print("BindingDB 컬럼 확인 중...")

df = pd.read_csv(
    BINDINGDB_TSV,
    sep="\t",
    nrows=5,
    on_bad_lines="skip",
    low_memory=False
)

print("\n--- BindingDB 컬럼 목록 ---")
for idx, col in enumerate(df.columns):
    print(f"{idx}: {col}")