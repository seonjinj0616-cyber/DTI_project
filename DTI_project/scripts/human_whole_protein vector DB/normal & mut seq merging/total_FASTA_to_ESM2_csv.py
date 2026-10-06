import os
import pandas as pd
from Bio import SeqIO

BASE_DIR = r"C:\workspace\python\project_personal\DTI_project\data\homo_protein_seq_fasta"


def get_clean_id(header_id, seq_type):
    """라벨링 시 즉시 알아볼 수 있도록 깔끔한 ID로 변환"""
    parts = header_id.split('|')
    acc_id = parts[1] if len(parts) >= 2 else header_id

    if seq_type == "mutant":
        return acc_id.replace("_MUT_", "_")  # 예: P04217_H52R
    else:
        return f"WT_{acc_id}"  # 예: WT_A0A0A0MRZ8


def prepare_for_esm2():
    esm_data = []

    files_info = [
        ("human_protein_ori", "wildtype"),
        ("human_mutants", "mutant")
    ]

    for filename, seq_type in files_info:
        file_path = None
        for ext in ["", ".fasta", ".fa", ".txt"]:
            target = os.path.join(BASE_DIR, filename + ext)
            if os.path.exists(target):
                file_path = target
                break

        if not file_path:
            print(f"⚠️ [경고] 파일을 찾을 수 없습니다: {filename}")
            continue

        for record in SeqIO.parse(file_path, "fasta"):
            clean_id = get_clean_id(record.id, seq_type)
            clean_seq = str(record.seq).replace(" ", "").replace("\r", "").replace("\n", "").upper()

            esm_data.append({
                "id": clean_id,
                "sequence": clean_seq,
                "length": len(clean_seq),
                "type": seq_type,
                "raw_header": record.description
            })

    df = pd.DataFrame(esm_data)

    output_path = os.path.join(BASE_DIR, "esm2_ready_dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 전처리 파일 생성 완료: {output_path}")
    return df


def run_quality_control(df):
    """ESM-2 입력 전 ID 및 서열 데이터 정합성 QC 검사"""
    print("\n==========================================")
    print("🔍 [QC 검증 시작] ID 및 데이터 정합성 점검")
    print("==========================================")

    passed = True

    # 1. ID 중복 검사 (중복 시 Dict 저장 시 벡터 덮어쓰기 발생)
    duplicates = df[df.duplicated(subset=['id'], keep=False)]
    if not duplicates.empty:
        print(f"❌ [FAIL] 중복된 ID가 {len(duplicates)}개 발견되었습니다!")
        print(duplicates[['id', 'type']])
        passed = False
    else:
        print("✅ [PASS] 모든 ID가 중복 없이 고유(Unique)합니다.")

    # 2. 결측치 검사
    if df[['id', 'sequence']].isnull().any().any():
        print("❌ [FAIL] ID 또는 서열 컬럼에 결측치(NaN/None)가 존재합니다.")
        passed = False
    else:
        print("✅ [PASS] 결측치가 존재하지 않습니다.")

    # 3. ID 유형별 변환 건수 확인
    wt_count = df['id'].str.startswith('WT_').sum()
    mut_count = len(df) - wt_count
    print(f"ℹ️ [INFO] ID 구성: Wildtype(WT_) {wt_count}개 / Mutant {mut_count}개")

    # 4. 아미노산 서열 불법 문자 검사 (표준 20개 아미노산 외 문자 존재 여부)
    invalid_seqs = df[df['sequence'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True)]
    if not invalid_seqs.empty:
        print(f"⚠️ [WARN] 비표준 아미노산 기호(X, B, Z 등) 포함 항목: {len(invalid_seqs)}개")
    else:
        print("✅ [PASS] 모든 서열이 표준 아미노산 기호로 정제되었습니다.")

    # 5. ESM-2 권장 길이(1024 aa) 초과 검사
    over_len = df[df['length'] > 1024]
    if not over_len.empty:
        print(f"⚠️ [WARN] ESM-2 기본 제한(1024 aa) 초과 서열: {len(over_len)}개 (추후 잘라내기 필요)")
    else:
        print("✅ [PASS] 모든 서열이 1024 aa 이하입니다.")

    print("------------------------------------------")
    if passed:
        print("🎉 [QC 최종 통과] ESM-2 모델 입력 준비가 완벽히 완료되었습니다.")
    else:
        print("🚨 [QC 실패] 출력된 오류 메시지를 확인 후 수정해 주세요.")


# 실행
if __name__ == "__main__":
    df_esm = prepare_for_esm2()
    if not df_esm.empty:
        run_quality_control(df_esm)