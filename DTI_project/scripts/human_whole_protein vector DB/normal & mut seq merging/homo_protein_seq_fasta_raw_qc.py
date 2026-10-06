"""
parse_uniprot_fasta.py

목적:
    UniProt에서 받은 canonical FASTA(UP000005640_9606.fasta.gz)와
    isoform FASTA(UP000005640_9606_additional.fasta.gz)를 파싱해서
    구조화된 CSV 테이블(uniprot_id, is_isoform, sequence, length 등)로 변환한다.

    Biopython 의존성 없이 표준 라이브러리(gzip)만으로 FASTA를 직접 파싱한다.

입력 폴더 규칙:
    INPUT_DIR 안에서 파일명이 FASTA_PREFIX로 시작하는 모든
    .fasta / .fasta.gz 파일을 자동으로 찾아서 전부 파싱한다.
    (예: UP000005640_9606.fasta.gz, UP000005640_9606_additional.fasta.gz)

출력:
    OUTPUT_CSV : 파싱된 전체 서열 테이블
    콘솔        : QC 요약 (총 서열 수, canonical/isoform 비율, 길이 분포,
                  ESM-2 최대 길이 초과 서열 수, 중복 ID 등)

사용법:
    python parse_uniprot_fasta.py
    (경로는 아래 상수에서 본인 환경에 맞게 수정)
"""

import glob
import gzip
import os

import pandas as pd

# =============================================================================
# 경로 설정 (본인 환경에 맞게 수정)
# =============================================================================
INPUT_DIR = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"
FASTA_PREFIX = "UP000005640_9606"  # 이 이름으로 시작하는 파일을 전부 찾아서 파싱
OUTPUT_CSV = os.path.join(INPUT_DIR, "wildtype_protein_sequences.csv")

# ESM-2 모델별 최대 토큰 길이 참고용 (특수 토큰 포함 실제 한도보다 약간 여유있게 잡음)
ESM2_MAX_LENGTH = 1022


def find_fasta_files(input_dir: str, prefix: str) -> list[str]:
    """
    input_dir 안에서 prefix로 시작하는 .fasta / .fasta.gz 파일을 모두 찾는다.

    같은 base 이름에 대해 .fasta와 .fasta.gz가 둘 다 존재하는 경우
    (예: gzip -k로 압축 해제하고 원본을 안 지운 경우) 같은 내용을
    두 번 파싱하는 걸 방지하기 위해, .fasta.gz를 우선 사용하고
    동일 base의 .fasta는 건너뛴다.
    """
    gz_files = glob.glob(os.path.join(input_dir, f"{prefix}*.fasta.gz"))
    plain_files = glob.glob(os.path.join(input_dir, f"{prefix}*.fasta"))

    gz_basenames = {os.path.basename(f)[: -len(".gz")] for f in gz_files}
    filtered_plain_files = [
        f for f in plain_files if os.path.basename(f) not in gz_basenames
    ]

    skipped = set(plain_files) - set(filtered_plain_files)
    if skipped:
        print("[안내] 아래 .fasta 파일은 동일 이름의 .fasta.gz가 이미 있어 건너뜁니다 (중복 방지):")
        for f in skipped:
            print(f"  - {f}")

    files = sorted(set(gz_files) | set(filtered_plain_files))
    return files


def open_fasta(path: str):
    """gzip 여부에 따라 적절히 파일을 연다."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_fasta_file(path: str) -> list[dict]:
    """
    UniProt FASTA 한 파일을 파싱한다.
    헤더 형식 예: >sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
                 >sp|P04637-2|P53_HUMAN Isoform 2 of ...
    """
    records = []
    current_header = None
    current_seq_lines: list[str] = []

    def flush():
        if current_header is None:
            return
        parts = current_header.split("|")
        # UniProt FASTA 표준: >db|accession|entry_name description...
        if len(parts) >= 3:
            db = parts[0].lstrip(">")
            accession = parts[1]
            rest = parts[2].split(" ", 1)
            entry_name = rest[0]
            description = rest[1] if len(rest) > 1 else ""
        else:
            # 예외적인 헤더 형식 대비 (그대로 accession에 저장)
            db = ""
            accession = current_header.lstrip(">").split(" ")[0]
            entry_name = ""
            description = current_header

        is_isoform = "-" in accession  # 예: P04637-2 형태면 isoform
        base_uniprot_id = accession.split("-")[0]
        sequence = "".join(current_seq_lines)

        records.append(
            {
                "uniprot_id": accession,
                "base_uniprot_id": base_uniprot_id,
                "is_isoform": is_isoform,
                "db": db,
                "entry_name": entry_name,
                "description": description,
                "sequence": sequence,
                "length": len(sequence),
                "source_file": os.path.basename(path),
            }
        )

    with open_fasta(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line
                current_seq_lines = []
            else:
                current_seq_lines.append(line)
        flush()  # 마지막 레코드 처리

    return records


def run_qc(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("UniProt FASTA 파싱 결과 QC")
    print("=" * 80)

    print(f"\n[1] 총 서열 수: {len(df):,}")
    print(f"  ├─ Canonical : {(~df['is_isoform']).sum():,}")
    print(f"  └─ Isoform   : {df['is_isoform'].sum():,}")

    print(f"\n[2] 소스 파일별 분포:")
    print(df["source_file"].value_counts().to_string())

    print(f"\n[3] Unique base_uniprot_id 수: {df['base_uniprot_id'].nunique():,}")
    print(f"[4] Unique uniprot_id(accession, isoform 포함) 수: {df['uniprot_id'].nunique():,}")

    dup_count = len(df) - df["uniprot_id"].nunique()
    if dup_count > 0:
        print(f"\n[경고] uniprot_id 기준 중복 {dup_count}건 발견")
        dup_ids = df[df.duplicated("uniprot_id", keep=False)]["uniprot_id"].unique()
        print(f"  중복 ID 예시: {list(dup_ids[:10])}")
    else:
        print("\n[PASS] uniprot_id 기준 완전 중복 없음")

    print(f"\n[5] 서열 길이 통계:")
    print(df["length"].describe().to_string())

    over_limit = df[df["length"] > ESM2_MAX_LENGTH]
    print(
        f"\n[6] ESM-2 최대 길이({ESM2_MAX_LENGTH}) 초과 서열: "
        f"{len(over_limit):,}건 ({len(over_limit)/len(df)*100:.2f}%)"
    )
    if len(over_limit) > 0:
        print("  → 이후 단계에서 truncation 또는 windowing 전략 필요")
        print(f"  최대 길이 Top 5:\n{over_limit.nlargest(5, 'length')[['uniprot_id', 'length']].to_string(index=False)}")

    empty_seq = df[df["length"] == 0]
    if len(empty_seq) > 0:
        print(f"\n[경고] 서열 길이 0인 레코드 {len(empty_seq)}건 발견 (파싱 오류 가능성)")
    else:
        print("\n[PASS] 빈 서열 없음")

    print("\n" + "=" * 80)
    print("QC 완료")
    print("=" * 80)


def main() -> None:
    fasta_files = find_fasta_files(INPUT_DIR, FASTA_PREFIX)
    if not fasta_files:
        raise FileNotFoundError(
            f"'{INPUT_DIR}' 안에서 '{FASTA_PREFIX}*' 패턴의 fasta/fasta.gz 파일을 찾지 못했습니다."
        )

    print(f"발견된 파일 ({len(fasta_files)}개):")
    for f in fasta_files:
        print(f"  - {f}")

    all_records = []


if __name__ == "__main__":
    main()

import os
from pathlib import Path

# 1. 파일 및 폴더 경로 설정 (Windows 경로 이스케이프 방지를 위해 raw string 'r' 사용)
DATA_DIR = Path(r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta")

# 입력 파일명 (확장자가 .fasta 또는 .fa 인 경우에 대응하도록 처리)
input_filename = "UP000005640_9606"
input_path = DATA_DIR / f"{input_filename}.fasta"

# .fasta가 아닐 경우 .fa 확장자 검사
if not input_path.exists():
    input_path = DATA_DIR / f"{input_filename}.fa"
    if not input_path.exists():
        input_path = DATA_DIR / input_filename  # 확장자가 없는 경우

# 필터링된 결과물 저장 경로
output_path = DATA_DIR / f"{input_filename}_swissprot_filtered.fasta"

print(f"입력 파일 로드 중: {input_path}")

# 2. Swiss-Prot (>sp|) 데이터 추출
saved_count = 0
write_flag = False


# Unreviewed(TrEMBL)를 전부 포함한 "대표 프로테옴(representative proteome)" 파일 filtering
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if line.startswith(">"):
            # >sp| 로 시작하는 Swiss-Prot (Canonical + Isoforms) 헤더만 필터링
            if line.startswith(">sp|"):
                write_flag = True
                saved_count += 1
                outfile.write(line)
            else:
                write_flag = False
        else:
            # 헤더 아래 서열 데이터 기록
            if write_flag:
                outfile.write(line)

print("\n--- 필터링 작업 완료 ---")
print(f"저장된 파일 위치: {output_path}")
print(f"추출된 단백질 개수 (Canonical + Isoforms): {saved_count:,} 개")