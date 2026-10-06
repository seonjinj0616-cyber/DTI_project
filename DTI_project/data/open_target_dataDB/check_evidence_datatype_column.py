"""
evidence parquet 의 실제 컬럼 이름을 확인하는 용도 (이미 받아두신 폴더를 그대로 씀, 새로 받을 필요 없음).
open_target_dataDB 폴더 안에서 실행하세요.
"""

import duckdb

PARQUET_DIR = "./opentargets"  # load_open_targets.py 의 PARQUET_DIR 과 동일하게 맞추세요

path = f"{PARQUET_DIR}/evidence/**/*.parquet"

con = duckdb.connect()

cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}', hive_partitioning=false)").fetchdf()

print("=== 컬럼 목록 ===")
print(cols[["column_name", "column_type"]].to_string(index=False))

# datatype/datasource 로 보이는 컬럼만 따로 추려서 보여줌
guess = cols[cols["column_name"].str.contains("datatype|datasource|type", case=False)]
if len(guess):
    print("\n=== 'type' 이 들어간 컬럼(후보) ===")
    print(guess.to_string(index=False))

print("\n=== 샘플: targetId, diseaseId, score 와 함께 있는 종류 컬럼 값 몇 개 ===")
sample = con.execute(f"SELECT * FROM read_parquet('{path}', hive_partitioning=false) LIMIT 5").fetchdf()
print(sample.to_string())
