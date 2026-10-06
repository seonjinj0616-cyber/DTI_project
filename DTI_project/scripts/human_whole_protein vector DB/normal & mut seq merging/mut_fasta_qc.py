import os
from Bio import SeqIO

# 지정된 경로 설정
BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
FILES = ["human_protein_ori", "human_mutants"]


def validate_fasta(file_dir, filename):
    # 확장자가 붙어있거나 없는 경우 모두 대응
    file_path = None
    for ext in ["", ".fasta", ".fa", ".txt"]:
        target = os.path.join(file_dir, filename + ext)
        if os.path.exists(target):
            file_path = target
            break

    if not file_path:
        print(f"❌ [파일 없음] 경로에 {filename} 파일이 존재하지 않습니다.")
        return

    print(f"\n==========================================")
    print(f"🔍 검증 대상: {os.path.basename(file_path)}")
    print(f"==========================================")

    try:
        # FASTA 파싱 시도
        records = list(SeqIO.parse(file_path, "fasta"))

        if not records:
            print("⚠️ [경고] '>' 로 시작하는 헤더를 찾을 수 없거나 파일이 비어 있습니다.")
            return

        # 내부 서열 이상 유무 확인 (공백, 개행 문자 등)
        invalid_count = 0
        for rec in records:
            seq_str = str(rec.seq)
            if any(char in seq_str for char in [" ", "\t", "\n", "\r"]):
                invalid_count += 1

        print(f"✅ [성공] 정상적인 FASTA 형식입니다.")
        print(f"   • 총 서열 개수: {len(records):,}개")
        print(f"   • 첫 번째 서열 ID: {records[0].id}")
        print(f"   • 첫 번째 서열 길이: {len(records[0].seq)} aa")

        if invalid_count > 0:
            print(f"⚠️ [주의] 서열 내 불필요한 공백/개행이 포함된 항목이 {invalid_count}개 있습니다.")

    except Exception as e:
        print(f"❌ [형식 에러] FASTA 파싱 중 오류가 발생했습니다.\n   상세 내용: {e}")


# 실행
if __name__ == "__main__":
    for f in FILES:
        validate_fasta(BASE_DIR, f)

# 파이썬으로 헤더, 서열 내용 표(DataFrame)로 출력하기

import pandas as pd

# Pandas 출력 설정 (텍스트 잘림 방지)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
FILES = ["human_protein_ori", "human_mutants"]


def peek_fasta(filename, num_records=3):
    # 파일 확장자 탐색
    file_path = None
    for ext in ["", ".fasta", ".fa", ".txt"]:
        target = os.path.join(BASE_DIR, filename + ext)
        if os.path.exists(target):
            file_path = target
            break

    if not file_path:
        print(f"❌ 경로에 {filename} 파일이 없습니다.")
        return

    data = []
    for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
        if i >= num_records:
            break

        # 헤더와 서열 분해
        seq_str = str(record.seq)
        data.append({
            "Header ID": record.id,
            "Description": record.description,
            "Length": len(seq_str),
            "Sequence Preview": seq_str[:25] + "..." if len(seq_str) > 25 else seq_str
        })

    df = pd.DataFrame(data)
    print(f"\n📋 [{filename}] 상위 {len(df)}개 내용 확인")
    print(df.to_string(index=False))


# 실행 (상위 3개씩 확인)
for f in FILES:
    peek_fasta(f, num_records=3)