import os
import re
from pathlib import Path

# 3글자 아미노산 약어 -> 1글자 매핑 맵
AA_MAP = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
    'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'Sec': 'U', 'Pyl': 'O'
}


def parse_fasta(fasta_path):
    """Swiss-Prot FASTA 파싱: {Accession: (Entry_Name, Sequence)}"""
    fasta_dict = {}
    current_acc = None
    current_entry = None
    current_seq = []

    with open(fasta_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_acc:
                    fasta_dict[current_acc] = (current_entry, "".join(current_seq))

                parts = line[1:].split()[0].split('|')
                if len(parts) >= 2:
                    current_acc = parts[1]
                    current_entry = parts[2] if len(parts) >= 3 else current_acc
                else:
                    current_acc = line[1:].split()[0]
                    current_entry = current_acc
                current_seq = []
            else:
                current_seq.append(line)

        if current_acc:
            fasta_dict[current_acc] = (current_entry, "".join(current_seq))

    return fasta_dict


def process_humsavar(fasta_path, humsavar_path, output_path):
    print(f"1. 참조 FASTA 데이터 로드 중: {fasta_path.name}")
    fasta_dict = parse_fasta(fasta_path)
    print(f"   -> {len(fasta_dict):,} 개의 정상 단백질 서열 확보 완료.")

    # p.His52Arg 형태 파싱용 정규식
    mut_pattern = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')

    written_count = 0
    skipped_no_acc = 0
    skipped_mismatch = 0
    skipped_bounds = 0
    seen_mutants = set()

    print(f"\n2. humsavar.txt 변이 적용 및 human_mutants.fasta 생성 시작...")

    with open(humsavar_path, 'r', encoding='utf-8', errors='ignore') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            # 헤더 및 주석 선(____ 등) 제외
            if not line or line.startswith('#') or line.startswith('_') or line.startswith('Main'):
                continue

            parts = line.split()
            # 최소 컬럼 개수 확인 (gene, AC, FTId, AA change 최소 4개 이상)
            if len(parts) < 4:
                continue

            acc = parts[1]  # 예: P04217
            change_str = parts[3]  # 예: p.His52Arg

            match = mut_pattern.match(change_str)
            if not match:
                continue

            wt_3, pos_str, mut_3 = match.groups()

            wt_aa = AA_MAP.get(wt_3)
            mut_aa = AA_MAP.get(mut_3)
            pos = int(pos_str)

            if not wt_aa or not mut_aa:
                continue

            # 참조 FASTA에 해당 UniProt AC가 존재하는지 검사
            if acc not in fasta_dict:
                skipped_no_acc += 1
                continue

            entry_name, wt_seq = fasta_dict[acc]
            idx = pos - 1  # 1-based -> 0-based

            if idx < 0 or idx >= len(wt_seq):
                skipped_bounds += 1
                continue

            # 원본 단백질 서열의 아미노산과 humsavar의 기존 아미노산(WT) 일치 여부 검증
            if wt_seq[idx] != wt_aa:
                skipped_mismatch += 1
                continue

            # 변이 ID 생성 (예: P04217_MUT_H52R)
            mut_id = f"{acc}_MUT_{wt_aa}{pos}{mut_aa}"
            if mut_id in seen_mutants:
                continue
            seen_mutants.add(mut_id)

            # 아미노산 1개 치환 (Mutant Sequence 생성)
            mut_seq = wt_seq[:idx] + mut_aa + wt_seq[idx + 1:]

            # FASTA 파일 작성
            header = f">sp|{mut_id}|{entry_name}_{wt_aa}{pos}{mut_aa}"
            outfile.write(f"{header}\n")

            for i in range(0, len(mut_seq), 60):
                outfile.write(f"{mut_seq[i:i + 60]}\n")

            written_count += 1

    print("\n--- 처리 완료 ---")
    print(f"생성된 파일 위치 : {output_path}")
    print(f"생성된 변이 단백질 : {written_count:,} 개")
    print(f"제외 (FASTA 미존재) : {skipped_no_acc:,} 개")
    print(f"제외 (서열 범위 초과): {skipped_bounds:,} 개")
    print(f"제외 (기존 AA 불일치): {skipped_mismatch:,} 개")


if __name__ == "__main__":
    DATA_DIR = Path(r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta")

    fasta_file = DATA_DIR / "UP000005640_9606_swissprot_filtered.fasta"
    humsavar_file = DATA_DIR / "humsavar_mutant_protein_seq.txt"
    output_file = DATA_DIR / "human_mutants.fasta"

    process_humsavar(fasta_file, humsavar_file, output_file)