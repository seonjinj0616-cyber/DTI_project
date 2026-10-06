import sqlite3

DB_PATH = r"C:\workspace\python\project_personal\DTI_project\data\chembl\chembl_37_sqlite\chembl_37\chembl_37_sqlite\chembl_37.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1. 테이블 목록
cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
""")

tables = cur.fetchall()

print("=== ChEMBL 37 Tables ===")
for table in tables:
    print(table[0])

# 2. 우리가 사용할 주요 테이블 구조
target_tables = [
    "molecule_dictionary",
    "compound_structures",
    "activities",
    "assays",
    "target_dictionary",
    "target_components",
    "component_sequences"
]

print("\n=== Important Table Schemas ===")

for table in target_tables:
    print(f"\n[{table}]")

    cur.execute(f"PRAGMA table_info({table})")
    columns = cur.fetchall()

    for col in columns:
        print(f"  {col[1]} | {col[2]}")

conn.close()