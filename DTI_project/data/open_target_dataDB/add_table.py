import sqlite3

con = sqlite3.connect(r"C:\workspace\python\project_personal\DTI_project\data\open_target_dataDB\cdss_integrated.db")
con.executescript(open("add_evidence_detail_table.sql", encoding="utf-8").read())
con.commit()
con.close()

print("완료")