import sqlite3

def init_cdss_db(db_path: str = "cdss_integrated.db"):
    """CDSS 통합 SQLite 데이터베이스 및 테이블 스키마 초기화 함수"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 위에서 작성한 DDL 스크립트 실행
    with open("schema.sql", "r", encoding="utf-8") as f:
        sql_script = f.read()

    cursor.executescript(sql_script)
    conn.commit()
    conn.close()
    print(f"✅ CDSS 통합 SQLite DB 스키마 구축 완료: '{db_path}'")


if __name__ == "__main__":
    init_cdss_db()