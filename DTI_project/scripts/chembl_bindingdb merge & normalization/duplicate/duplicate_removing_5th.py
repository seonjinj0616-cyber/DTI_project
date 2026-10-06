import os
import pandas as pd


# ============================================================
# DEDUP5
# ============================================================
#
# 목적
# ------------------------------------------------------------
# merged_dti_DEDUP4.csv 전체를 대상으로
#
# 1. ChEMBL ↔ BindingDB 공통 그룹 탐색
# 2. 1:1 그룹은 그대로 유지
# 3. 1:N / N:1 / N:N 그룹만 추출
# 4. 해당 그룹 내부에서 source별 중복 제거
# 5. 중복 제거된 결과를 전체 DEDUP4 데이터와 재결합
# 6. merged_dti_DEDUP5.csv 생성
#
# 중요
# ------------------------------------------------------------
# - 전체 데이터에 일괄 drop_duplicates 하지 않음
# - 1:1 그룹은 삭제하지 않음
# - ChEMBL ↔ BindingDB 다중매칭 그룹만 처리
# - source별로만 중복 제거
#
# ============================================================


# ============================================================
# 0. 경로 설정
# ============================================================

INPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_DEDUP4.csv"
)

OUTPUT_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\merged_dti_DEDUP5.csv"
)

QC_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\dedup5_qc.csv"
)

GROUP_QC_FILE = (
    r"C:\workspace\python\project_personal\DTI_project"
    r"\data\output\dedup5_group_qc.csv"
)


# ============================================================
# 1. 설정
# ============================================================

CHUNK_SIZE = 200_000


# ============================================================
# 2. ChEMBL ↔ BindingDB 공통 그룹 확인 기준
# ============================================================
#
# 기존 DEDUP5/QC 기준
#
# target_name       제외
# publication_id    제외
# affinity_relation 제외
# assay_id          제외
# smiles            제외
#
# affinity_value는 소수점 둘째 자리까지 반올림하여 비교
#
# ============================================================

OVERLAP_KEY_COLS = [
    "chembl_id",
    "uniprot_id",
    "affinity_type",
    "affinity_value_rounded",
    "affinity_unit",
]


# ============================================================
# 3. 다중 매칭 그룹 내부 중복 제거 기준
# ============================================================
#
# 사용자가 지정한 기준
#
# 같은 source 내부에서 아래 6개 값이 모두 같으면
# 중복으로 판단
#
# ============================================================

DUPLICATE_KEYS = [
    "chembl_id",
    "smiles",
    "uniprot_id",
    "affinity_type",
    "affinity_relation",
    "affinity_value",
]


# ============================================================
# 4. 필수 컬럼
# ============================================================

REQUIRED_COLUMNS = [
    "source",
    "chembl_id",
    "smiles",
    "uniprot_id",
    "affinity_type",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
]


# ============================================================
# 5. 파일 존재 확인
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )


# ============================================================
# 6. 입력 컬럼 확인
# ============================================================

header = pd.read_csv(
    INPUT_FILE,
    nrows=0
)

missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in header.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다:\n{missing_columns}"
    )


# ============================================================
# 7. 전체 DEDUP4 읽기
# ============================================================

print("=" * 100)
print("DEDUP5")
print("merged_dti_DEDUP4 전체 데이터에서 다중 매칭 그룹 내부 중복 제거")
print("=" * 100)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"\n전체 DEDUP4 행 수: {len(df):,}")


# ============================================================
# 8. 원본 행 번호 보존
# ============================================================
#
# 매우 중요
#
# 이후 merge()를 여러 번 수행하기 때문에
# DataFrame의 기본 index를 삭제 대상 식별자로 사용하면 안 됨.
#
# 따라서 처음부터 원본 행 번호를 별도 컬럼으로 보존한다.
#
# ============================================================

df["_original_index"] = range(len(df))


# ============================================================
# 9. source 표준화
# ============================================================

df["source"] = (
    df["source"]
    .astype("string")
    .str.strip()
)


# ============================================================
# 10. affinity_value 숫자 변환
# ============================================================

df["affinity_value_numeric"] = pd.to_numeric(
    df["affinity_value"],
    errors="coerce"
)


# ============================================================
# 11. 공통 그룹 비교용 affinity_value 반올림
# ============================================================
#
# 원본 affinity_value는 변경하지 않음.
#
# 오직 공통 그룹 탐색용으로만 사용.
#
# ============================================================

df["affinity_value_rounded"] = (
    df["affinity_value_numeric"].round(2)
)


# ============================================================
# 12. ChEMBL / BindingDB 분리
# ============================================================

chembl = df[
    df["source"] == "ChEMBL"
].copy()

bindingdb = df[
    df["source"] == "BindingDB"
].copy()

print(f"ChEMBL rows   : {len(chembl):,}")
print(f"BindingDB rows: {len(bindingdb):,}")


# ============================================================
# 13. 각 source의 공통 그룹별 개수 계산
# ============================================================

chembl_group_count = (
    chembl
    .groupby(
        OVERLAP_KEY_COLS,
        dropna=False
    )
    .size()
    .reset_index(
        name="chembl_count"
    )
)


bindingdb_group_count = (
    bindingdb
    .groupby(
        OVERLAP_KEY_COLS,
        dropna=False
    )
    .size()
    .reset_index(
        name="bindingdb_count"
    )
)


# ============================================================
# 14. ChEMBL ↔ BindingDB 공통 그룹 탐색
# ============================================================

overlap_groups = pd.merge(
    chembl_group_count,
    bindingdb_group_count,
    on=OVERLAP_KEY_COLS,
    how="inner"
)

print(
    f"\nChEMBL ↔ BindingDB 공통 그룹: "
    f"{len(overlap_groups):,}"
)


# ============================================================
# 15. 공통 그룹 관계 유형
# ============================================================

overlap_groups["overlap_type"] = (
    overlap_groups["chembl_count"].astype(str)
    + ":"
    + overlap_groups["bindingdb_count"].astype(str)
)


# ============================================================
# 16. 1:1 / 다중매칭 그룹 분리
# ============================================================
#
# 1:1
# → 절대 건드리지 않음
#
# 1:N / N:1 / N:N
# → DEDUP5 내부 중복 제거 대상
#
# ============================================================

one_to_one_groups = overlap_groups[
    (overlap_groups["chembl_count"] == 1)
    &
    (overlap_groups["bindingdb_count"] == 1)
].copy()


multi_groups = overlap_groups[
    ~(
        (overlap_groups["chembl_count"] == 1)
        &
        (overlap_groups["bindingdb_count"] == 1)
    )
].copy()


print(
    f"1:1 그룹                  : "
    f"{len(one_to_one_groups):,}"
)

print(
    f"1:N / N:1 / N:N 그룹     : "
    f"{len(multi_groups):,}"
)


# ============================================================
# 17. 다중 매칭 그룹 ID 생성
# ============================================================

multi_groups = (
    multi_groups
    .reset_index(drop=True)
)

multi_groups["matched_group_id"] = (
    "MULTI_"
    +
    (multi_groups.index + 1)
    .astype(str)
    .str.zfill(8)
)


# ============================================================
# 18. 다중 매칭 그룹 정보 생성
# ============================================================

group_info = multi_groups[
    OVERLAP_KEY_COLS
    +
    [
        "chembl_count",
        "bindingdb_count",
        "overlap_type",
        "matched_group_id",
    ]
].copy()


# ============================================================
# 19. ChEMBL 다중 매칭 실제 행 추출
# ============================================================

chembl_multi = pd.merge(
    chembl,
    group_info,
    on=OVERLAP_KEY_COLS,
    how="inner"
)

chembl_multi["matched_source"] = "ChEMBL"


# ============================================================
# 20. BindingDB 다중 매칭 실제 행 추출
# ============================================================

bindingdb_multi = pd.merge(
    bindingdb,
    group_info,
    on=OVERLAP_KEY_COLS,
    how="inner"
)

bindingdb_multi["matched_source"] = "BindingDB"


# ============================================================
# 21. 다중 매칭 실제 행 합치기
# ============================================================

multi_rows = pd.concat(
    [
        chembl_multi,
        bindingdb_multi
    ],
    ignore_index=True
)


print(
    f"\n다중 매칭 그룹에 포함된 실제 행 수: "
    f"{len(multi_rows):,}"
)


# ============================================================
# 22. 다중 매칭 그룹 내부 중복 확인
# ============================================================
#
# 중요
#
# 전체 데이터에 drop_duplicates 하지 않는다.
#
# 오직
#
# matched_group_id
# +
# matched_source
# +
# DUPLICATE_KEYS
#
# 가 동일한 경우만 중복으로 판단한다.
#
# ============================================================

DEDUP_KEYS = [
    "matched_group_id",
    "matched_source",
] + DUPLICATE_KEYS


multi_rows["_is_duplicate"] = (
    multi_rows
    .duplicated(
        subset=DEDUP_KEYS,
        keep="first"
    )
)


# ============================================================
# 23. 실제 삭제 대상 추출
# ============================================================

multi_duplicate_rows = multi_rows[
    multi_rows["_is_duplicate"]
].copy()


multi_keep_rows = multi_rows[
    ~multi_rows["_is_duplicate"]
].copy()


print(
    f"다중 매칭 내부 중복 삭제 대상: "
    f"{len(multi_duplicate_rows):,}"
)

print(
    f"다중 매칭 내부 중복 제거 후: "
    f"{len(multi_keep_rows):,}"
)


# ============================================================
# 24. 삭제 대상 원본 index 추출
# ============================================================
#
# merge 후에도 _original_index가 유지되므로
# 실제 DEDUP4 원본 행을 정확하게 식별할 수 있다.
#
# ============================================================

delete_indices = set(
    multi_duplicate_rows["_original_index"]
)


print(
    f"\n실제 DEDUP4에서 제거할 행 수: "
    f"{len(delete_indices):,}"
)


# ============================================================
# 25. Source별 삭제 통계
# ============================================================

print("\n" + "=" * 100)
print("다중 매칭 그룹 내부 중복 삭제 통계")
print("=" * 100)

qc_rows = []

for source in ["ChEMBL", "BindingDB"]:

    before = int(
        (
            multi_rows["matched_source"]
            == source
        ).sum()
    )

    deleted = int(
        (
            multi_duplicate_rows["matched_source"]
            == source
        ).sum()
    )

    after = before - deleted

    print(
        f"{source:12s} : "
        f"{before:,} → {after:,} "
        f"(삭제 {deleted:,})"
    )

    qc_rows.append({
        "source": source,
        "multi_match_rows_before": before,
        "multi_match_rows_after": after,
        "deleted_rows": deleted,
    })


# ============================================================
# 26. 다중 매칭 그룹별 BEFORE 관계
# ============================================================

before_group = (
    multi_rows
    .groupby(
        [
            "matched_group_id",
            "matched_source"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="before_count"
    )
)


# ============================================================
# 27. 다중 매칭 그룹별 AFTER 관계
# ============================================================

after_group = (
    multi_keep_rows
    .groupby(
        [
            "matched_group_id",
            "matched_source"
        ],
        dropna=False
    )
    .size()
    .reset_index(
        name="after_count"
    )
)


# ============================================================
# 28. BEFORE / AFTER 그룹 QC
# ============================================================

group_qc = pd.merge(
    before_group,
    after_group,
    on=[
        "matched_group_id",
        "matched_source"
    ],
    how="left"
)

group_qc["after_count"] = (
    group_qc["after_count"]
    .fillna(0)
    .astype(int)
)

group_qc["deleted_count"] = (
    group_qc["before_count"]
    -
    group_qc["after_count"]
)


# ============================================================
# 29. 최종 그룹 관계 계산
# ============================================================

group_relation = (
    group_qc
    .pivot(
        index="matched_group_id",
        columns="matched_source",
        values="after_count"
    )
    .reset_index()
)

group_relation = group_relation.rename_axis(
    None,
    axis=1
)


if "ChEMBL" not in group_relation.columns:
    group_relation["ChEMBL"] = 0


if "BindingDB" not in group_relation.columns:
    group_relation["BindingDB"] = 0


group_relation["ChEMBL"] = (
    group_relation["ChEMBL"]
    .fillna(0)
    .astype(int)
)


group_relation["BindingDB"] = (
    group_relation["BindingDB"]
    .fillna(0)
    .astype(int)
)


group_relation["final_overlap_type"] = (
    group_relation["ChEMBL"].astype(str)
    + ":"
    + group_relation["BindingDB"].astype(str)
)


# ============================================================
# 30. Group QC에 최종 관계 추가
# ============================================================

group_qc = pd.merge(
    group_qc,
    group_relation[
        [
            "matched_group_id",
            "final_overlap_type"
        ]
    ],
    on="matched_group_id",
    how="left"
)


# ============================================================
# 31. Group QC 저장
# ============================================================

group_qc.to_csv(
    GROUP_QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 32. 최종 그룹 관계 출력
# ============================================================

print("\n" + "=" * 100)
print("다중 매칭 그룹 최종 관계")
print("=" * 100)

print(
    group_relation[
        "final_overlap_type"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 33. 핵심
# ============================================================
#
# DEDUP4 전체 데이터에서
#
# "다중 매칭 그룹 내부에서 실제로 중복이라고 판단된 행"
#
# 만 제거한다.
#
# 나머지는 전부 그대로 유지한다.
#
# ============================================================

print("\n" + "=" * 100)
print("전체 DEDUP4 데이터에 삭제 대상 적용")
print("=" * 100)


df_result = df[
    ~df["_original_index"].isin(delete_indices)
].copy()


# ============================================================
# 34. 최종 행 수 확인
# ============================================================

print(
    f"DEDUP4 원본 행 수 : "
    f"{len(df):,}"
)

print(
    f"삭제 행 수        : "
    f"{len(delete_indices):,}"
)

print(
    f"DEDUP5 최종 행 수 : "
    f"{len(df_result):,}"
)


# ============================================================
# 35. 삭제 검증
# ============================================================

expected_rows = (
    len(df)
    -
    len(delete_indices)
)

if len(df_result) != expected_rows:
    raise RuntimeError(
        "최종 행 수 검증 실패"
    )


# ============================================================
# 36. Source별 최종 행 수
# ============================================================

print("\n[최종 source 분포]")

print(
    df_result["source"]
    .value_counts()
)


# ============================================================
# 37. DEDUP5 QC 저장
# ============================================================

qc_df = pd.DataFrame(qc_rows)

qc_df["total_input_rows"] = len(df)

qc_df["total_deleted_rows"] = (
    len(delete_indices)
)

qc_df["total_output_rows"] = (
    len(df_result)
)

qc_df.to_csv(
    QC_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 38. 보조 QC 컬럼 제거
# ============================================================
#
# 아래 컬럼들은 DEDUP5 작업용이므로
# 최종 데이터에는 넣지 않는다.
#
# 원본 DEDUP4의 원래 컬럼 구조를 유지한다.
#
# ============================================================

DROP_WORK_COLUMNS = [
    "_original_index",
    "affinity_value_numeric",
    "affinity_value_rounded",
]

df_result = df_result.drop(
    columns=[
        col
        for col in DROP_WORK_COLUMNS
        if col in df_result.columns
    ]
)


# ============================================================
# 39. 최종 CSV 저장
# ============================================================

df_result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 40. 최종 결과 검증
# ============================================================

print("\n" + "=" * 100)
print("DEDUP5 완료")
print("=" * 100)

print(
    f"입력 파일 : {INPUT_FILE}"
)

print(
    f"출력 파일 : {OUTPUT_FILE}"
)

print(
    f"DEDUP4 행 수 : {len(df):,}"
)

print(
    f"삭제 행 수   : {len(delete_indices):,}"
)

print(
    f"DEDUP5 행 수 : {len(df_result):,}"
)

print(
    f"\nQC 파일      : {QC_FILE}"
)

print(
    f"Group QC     : {GROUP_QC_FILE}"
)

print("\n종료 코드 0(으)로 완료된 프로세스")