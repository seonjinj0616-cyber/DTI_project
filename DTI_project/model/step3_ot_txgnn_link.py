"""
Step 3: 최종 단백질(Top5) -> Open Targets DB 질병 후보 -> PrimeKG ID 매핑 -> TxGNN (약물, 질병) 점수 연결

흐름
    [prepare]  이 파일 (일반 파이썬 환경)
        1. final5_proteins.csv 의 UniProt accession -> Ensembl Gene ID (ENSG)
               (UniProt REST 교차참조 -> 실패하면 유전자 기호로 ref_target_pathway 조회)
        2. cdss_integrated.db 의 ref_target_disease_evidence 에서 타깃별 질병 근거(overall_score) 조회
        3. 타깃별 상위 질병을 모아 질병 후보군 구성 (여러 타깃이 공통으로 가리키는 질병 우선)
        4. Open Targets 질병 ID(EFO_/MONDO_/Orphanet_/HP_ ...) -> PrimeKG 질병 노드 매핑
        5. 처음 입력한 약물(SMILES)이 PrimeKG 약물 노드(DrugBank ID)로 존재하는지 확인
        -> TxGNN 입력 파일(txgnn_drugs.csv, txgnn_diseases.csv) 생성
    [TxGNN]    step3_txgnn_score.py (TxGNN 이 설치된 환경) : 약물 x 질병 점수 계산 -> txgnn_scores.csv
    [finalize] 이 파일: 점수를 후보군에 합쳐 최종 표 생성, (선택) cdss_integrated.db 에 저장

꼭 알아둘 점
    * TxGNN 은 PrimeKG 에 "이미 있는" 약물 노드(DrugBank ID)만 점수를 낼 수 있다.
      SMILES 가 알려진 약물이면 PubChem 동의어로 DrugBank ID 를 찾아 연결하지만,
      신규 화합물이면 PrimeKG 에 없어서 점수를 계산할 수 없다.
      이때는 --proxy-from-targets 로 "Top5 타깃을 표적으로 하는 PrimeKG 약물"을 대리(proxy)로 쓸 수 있으며,
      이 점수는 입력 화합물 자체의 점수가 아니라는 표시(txgnn_score_source = proxy)가 결과에 남는다.
    * ref_target_disease_evidence.overall_score 는 evidence 데이터셋의 개별 증거 점수 중 최댓값(>= 0.4)이다.
      Open Targets 웹의 "overall association score"(증거를 종합한 값)와는 다르다.

필요한 자료
    - cdss_integrated.db (Open Targets 참조 테이블이 적재된 SQLite)
    - PrimeKG kg.csv  (Harvard Dataverse, 질병/약물 노드와 약물-단백질 연결에 사용)
    - (권장) Open Targets disease parquet 폴더: dbXRefs 로 EFO/Orphanet -> MONDO 매핑률을 높인다
      (DB 적재 때 쓴 ./opentargets/disease 폴더가 남아 있으면 --ot-disease-dir 로 지정)
    - 인터넷: UniProt(ENSG 매핑), PubChem(SMILES -> 약물 이름/DrugBank ID)  (결과는 캐시)
"""

import os
import re
import sys
import glob
import json
import time
import uuid
import sqlite3
import hashlib
import argparse
import pathlib
import subprocess

import pandas as pd


# ============================================================
# 0. 설정
# ============================================================

def _find_project_root(marker_name="DTI_project"):
    """
    이 스크립트 파일 위치에서 위로 올라가며 이름이 marker_name 인 폴더를 찾는다.
    DTI_project 폴더 전체를 다른 컴퓨터/다른 드라이브/다른 사용자 계정으로 옮겨도,
    폴더 이름만 같으면(하위 몇 단계 안에 이 스크립트가 있든) 경로를 자동으로 다시 찾는다.
    """

    path = os.path.dirname(os.path.abspath(__file__))

    while True:

        if os.path.basename(path) == marker_name:
            return path

        parent = os.path.dirname(path)

        if parent == path:  # 드라이브/파일시스템 루트까지 올라갔는데 못 찾음
            raise RuntimeError(
                "'%s' 폴더를 찾을 수 없습니다. 이 스크립트가 %s 폴더 안(하위 폴더 포함)에 있는지 확인하세요."
                % (marker_name, marker_name)
            )

        path = parent


PROJECT_DIR = _find_project_root()


def _find_env_python(env_name):
    """
    현재 실행 중인 파이썬(sys.executable)이 conda 환경 안에 있다고 가정하고,
    같은 위치(...\\envs\\<이 환경>\\python.exe)의 다른 환경(env_name)의 python.exe 경로를 추정한다.
    conda 환경은 보통 <conda_root>\\envs\\<환경이름>\\python.exe 구조라, 그 폴더 이름만 바꾸면 된다.
    없으면(경로 추정 실패 또는 실제로 그런 환경이 없으면) None 을 반환한다.
    """

    cur = os.path.abspath(sys.executable)
    parts = cur.split(os.sep)

    if "envs" not in parts:
        return None

    envs_idx = parts.index("envs")

    if envs_idx + 1 >= len(parts):
        return None

    parts = list(parts)
    parts[envs_idx + 1] = env_name
    candidate = os.sep.join(parts)

    return candidate if os.path.exists(candidate) else None

OT_DB_DIR = os.path.join(PROJECT_DIR, "data", "open_target_dataDB")     # C:\workspace\python\project_personal\DTI_project\data\open_target_dataDB
OT_DB_PATH = os.path.join(OT_DB_DIR, "cdss_integrated.db")
OT_DISEASE_PARQUET_DIR = os.path.join(OT_DB_DIR, "opentargets", "disease")   # 없으면 --ot-disease-dir 로 지정

PRIMEKG_CSV = os.path.join(PROJECT_DIR, "data", "primekg", "kg.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
UNIPROT_CACHE_DIR = os.path.join(OUTPUT_DIR, "uniprot_ensembl_cache")
PUBCHEM_CACHE_DIR = os.path.join(OUTPUT_DIR, "pubchem_cache")

MIN_EVIDENCE_SCORE = 0.4          # DB 에는 이미 0.4 이상만 있음 (더 높이려면 여기서)
PER_TARGET_TOP_K = 50             # 타깃당 상위 몇 개 질병까지 후보로 볼지 (0 = 전부)
MAX_CANDIDATES = 300              # TxGNN 으로 점수를 낼 질병 후보 최대 개수 (0 = 제한 없음)

DRUGBANK_RE = re.compile(r"^DB\d{5}$")


# ============================================================
# 1. UniProt accession -> Ensembl gene (ENSG)
# ============================================================

def canonical_accession(acc):
    return str(acc).split("-")[0]


def parse_ensembl_entry(entry):
    """UniProt REST JSON 에서 유전자 기호 / ENSG(버전 제거) / 단백질 이름을 뽑는다."""

    symbol = None
    genes = entry.get("genes") or []

    if genes:
        symbol = (genes[0].get("geneName") or {}).get("value")

    ensg = []

    for xref in entry.get("uniProtKBCrossReferences") or []:

        if xref.get("database") != "Ensembl":
            continue

        candidates = [str(p.get("value", "")) for p in (xref.get("properties") or []) if p.get("key") == "GeneId"]
        candidates.append(str(xref.get("id", "")))

        for c in candidates:
            gid = c.split(".")[0]
            if gid.startswith("ENSG") and gid not in ensg:
                ensg.append(gid)

    description = entry.get("proteinDescription") or {}
    name_block = description.get("recommendedName")

    if not name_block:
        name_block = (description.get("submissionNames") or [{}])[0]

    protein_name = ((name_block or {}).get("fullName") or {}).get("value")

    return {"gene_symbol": symbol, "ensembl_gene_ids": ensg, "protein_name": protein_name}


def fetch_uniprot_ensembl(accession, cache_dir=None, timeout=15, retries=2):
    """반환: (dict 또는 None, status).  status: cache / fetched / not_found / http_<코드> / connection_failed / requests_missing"""

    accession = canonical_accession(accession)
    path = os.path.join(cache_dir, accession + ".json") if cache_dir else None

    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), "cache"
        except (OSError, ValueError):
            pass

    try:
        import requests
    except ImportError:
        return None, "requests_missing"

    url = "https://rest.uniprot.org/uniprotkb/%s.json" % accession
    status = "connection_failed"

    for attempt in range(retries + 1):

        try:
            r = requests.get(url, timeout=timeout)
        except Exception:
            time.sleep(0.5 * (attempt + 1))
            continue

        if r.status_code == 200:

            info = parse_ensembl_entry(r.json())

            if path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(info, f, ensure_ascii=False)

            time.sleep(0.1)

            return info, "fetched"

        if r.status_code == 404:
            return None, "not_found"

        status = "http_%d" % r.status_code

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.0 * (attempt + 1))
            continue

        break

    return None, status


# ============================================================
# 2. Open Targets SQLite DB
# ============================================================

def open_ot_db(path, read_only=True):

    if not os.path.exists(path):
        raise FileNotFoundError("Open Targets DB 를 찾을 수 없습니다: %s" % path)

    if read_only:
        uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
    else:
        con = sqlite3.connect(path)

    con.execute("PRAGMA foreign_keys = ON")

    return con


def check_ot_schema(con, need_step3_tables=False):

    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    required = {"ref_target_disease_evidence", "ref_target_pathway"}

    if need_step3_tables:
        required |= {"analysis_history", "step3_indication_results"}

    missing = sorted(required - tables)

    if missing:
        raise RuntimeError(
            "DB 에 필요한 테이블이 없습니다: %s\n  -> db_create_schema.sql 로 스키마를 만들고 "
            "load_open_targets.py 로 적재했는지 확인하세요." % missing
        )

    counts = {t: con.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
              for t in ("ref_target_disease_evidence", "ref_target_pathway")}

    return counts


def _in_clause(values):
    return ",".join("?" for _ in values)


def targets_with_evidence(con, target_ids):
    """주어진 ENSG 중 질병 근거 행이 있는 것만 반환."""

    ids = sorted(set(target_ids))

    if not ids:
        return set()

    rows = con.execute(
        "SELECT DISTINCT target_id FROM ref_target_disease_evidence WHERE target_id IN (%s)" % _in_clause(ids), ids
    ).fetchall()

    return {r[0] for r in rows}


def targets_by_symbol(con, symbol):
    """유전자 기호로 ENSG 를 찾는다 (ref_target_pathway 에 기호가 저장되어 있음)."""

    if not symbol:
        return []

    rows = con.execute(
        "SELECT DISTINCT target_id FROM ref_target_pathway WHERE UPPER(gene_symbol) = UPPER(?)", (symbol,)
    ).fetchall()

    return [r[0] for r in rows]


def target_exists_anywhere(con, target_id):
    """근거는 없어도 DB 의 어느 참조 테이블에든 등장하는 타깃인지."""

    for table in ("ref_target_pathway", "ref_target_moa", "ref_target_phenotypes"):
        try:
            if con.execute("SELECT 1 FROM %s WHERE target_id = ? LIMIT 1" % table, (target_id,)).fetchone():
                return True
        except sqlite3.OperationalError:
            continue
    return False


def resolve_targets(final_df, fetcher, con):
    """
    최종 단백질 -> Open Targets 타깃(ENSG).
    반환: (protein 별 매핑 상태 표, (id, uniprot_id, target_id, gene_symbol) 행 표)
    """

    map_rows, target_rows = [], []

    for _, row in final_df.iterrows():

        pid, acc = str(row["id"]), str(row["uniprot_id"])

        info, fetch_status = fetcher(acc)

        record = {"id": pid, "uniprot_id": acc, "gene_symbol": None, "protein_name": None,
                  "ensembl_gene_ids": None, "target_ids_used": None, "status": None}

        symbol = None
        ensg_list = []

        if info is None:
            record["status"] = "uniprot_unavailable(%s)" % fetch_status
        else:
            symbol = info.get("gene_symbol")
            ensg_list = info.get("ensembl_gene_ids", [])
            record["gene_symbol"] = symbol
            record["protein_name"] = info.get("protein_name")
            record["ensembl_gene_ids"] = ",".join(ensg_list)

        chosen, method = [], None

        if ensg_list:
            with_ev = targets_with_evidence(con, ensg_list)
            chosen = [g for g in ensg_list if g in with_ev]
            method = "matched_ensembl" if chosen else None

        if not chosen and symbol:
            by_symbol = targets_by_symbol(con, symbol)
            with_ev = targets_with_evidence(con, by_symbol)
            chosen = [g for g in by_symbol if g in with_ev]
            method = "matched_symbol" if chosen else None

            known = by_symbol or []
            if not chosen and (known or any(target_exists_anywhere(con, g) for g in ensg_list)):
                record["status"] = "target_found_no_evidence"

        if chosen:
            record["status"] = method
            record["target_ids_used"] = ",".join(chosen)
            for g in chosen:
                target_rows.append({"id": pid, "uniprot_id": acc, "target_id": g,
                                    "gene_symbol": symbol or g})
        elif record["status"] is None:
            record["status"] = "not_in_db" if info is not None else record["status"]

        map_rows.append(record)

    return pd.DataFrame(map_rows), pd.DataFrame(
        target_rows, columns=["id", "uniprot_id", "target_id", "gene_symbol"])


def fetch_evidence(con, target_ids, min_score=MIN_EVIDENCE_SCORE):

    ids = sorted(set(target_ids))

    if not ids:
        return pd.DataFrame(columns=["target_id", "disease_id", "disease_name", "overall_score"])

    query = (
        "SELECT target_id, disease_id, disease_name, overall_score FROM ref_target_disease_evidence "
        "WHERE target_id IN (%s) AND overall_score >= ?" % _in_clause(ids)
    )

    return pd.read_sql_query(query, con, params=ids + [float(min_score)])


def fetch_target_phenotypes(con, target_ids):
    """
    ref_target_phenotypes 조회 (load_open_targets.py 가 채운 두 출처):
      evidence_source 가 'safetyLiabilities:...' 로 시작 -> target.safetyLiabilities (문헌 등으로 이미 알려진 위험)
      evidence_source 가 'openFDA_significant_adverse_target_reactions'         -> 실사용 보고 기반 통계적 신호
    """

    ids = sorted(set(target_ids))

    if not ids:
        return pd.DataFrame(columns=["target_id", "phenotype_id", "phenotype_label", "evidence_source"])

    query = ("SELECT target_id, phenotype_id, phenotype_label, evidence_source FROM ref_target_phenotypes "
             "WHERE target_id IN (%s)" % _in_clause(ids))

    return pd.read_sql_query(query, con, params=ids)


def target_safety_sentence(gene_symbol, known, possible, max_items=8):
    """
    known(target.safetyLiabilities)과 possible(openFDA 통계적 신호) 목록으로 안내 문구를 만든다.
    요청하신 형식: "알려진 부작용은 '~'이고(target.safetyLiabilities 참고), '~' 이러한 부작용이
    일어날 가능성이 있습니다(openfda_significant_adverse_target_reactions 참고)."
    두 출처 중 하나 또는 둘 다 정보가 없으면 그 사실을 그대로 알린다.
    """

    def fmt(items):
        items = sorted(items)
        text = ", ".join(items[:max_items])
        return text + (" 등" if len(items) > max_items else "")

    has_known, has_possible = bool(known), bool(possible)

    if has_known and has_possible:
        return ("%s: 알려진 부작용은 '%s'이고(target.safetyLiabilities 참고), "
                "'%s' 이러한 부작용이 일어날 가능성이 있습니다(openfda_significant_adverse_target_reactions 참고)."
                % (gene_symbol, fmt(known), fmt(possible)))

    if has_known:
        return ("%s: 알려진 부작용은 '%s'이고(target.safetyLiabilities 참고), "
                "그 외 통계적으로 유의미한 부작용 신호는 없습니다(openfda_significant_adverse_target_reactions 참고)."
                % (gene_symbol, fmt(known)))

    if has_possible:
        return ("%s: 알려진 부작용 정보는 없으나(target.safetyLiabilities 참고), "
                "'%s' 이러한 부작용이 일어날 가능성이 있습니다(openfda_significant_adverse_target_reactions 참고)."
                % (gene_symbol, fmt(possible)))

    return ("%s: 이 타깃에 대해 알려진 안전성 정보가 없습니다 "
            "(target.safetyLiabilities / openfda_significant_adverse_target_reactions 모두 참고)." % gene_symbol)


def summarize_target_safety(target_ids, gene_symbol_of, phenotypes):
    """타깃별로 known/possible 목록과 안내 문장을 만든다. 반환: DataFrame."""

    rows = []

    for tid in sorted(set(target_ids)):

        sub = phenotypes[phenotypes["target_id"] == tid]
        known = set(sub.loc[sub["evidence_source"].str.startswith("safetyLiabilities"), "phenotype_label"])
        possible = set(sub.loc[sub["evidence_source"] == "openFDA_significant_adverse_target_reactions", "phenotype_label"])
        symbol = gene_symbol_of.get(tid, tid)

        rows.append({"target_id": tid, "gene_symbol": symbol,
                     "known_side_effects": "; ".join(sorted(known)),
                     "possible_side_effects": "; ".join(sorted(possible)),
                     "summary": target_safety_sentence(symbol, known, possible)})

    return pd.DataFrame(rows, columns=["target_id", "gene_symbol", "known_side_effects",
                                       "possible_side_effects", "summary"])


def fetch_target_pathways(con, target_ids):
    """ref_target_pathway 조회 (Reactome). 화합물의 약리 작용과 무관하게, 타깃 자체가 어떤 생물학적
    경로에 관여하는지를 보여주는 배경 정보 -> 그 경로 이상이 질병과 어떻게 이어지는지 맥락을 제공한다."""

    ids = sorted(set(target_ids))

    if not ids:
        return pd.DataFrame(columns=["target_id", "gene_symbol", "pathway_id", "pathway_name"])

    query = ("SELECT target_id, gene_symbol, pathway_id, pathway_name FROM ref_target_pathway "
             "WHERE target_id IN (%s)" % _in_clause(ids))

    return pd.read_sql_query(query, con, params=ids)


def target_pathway_sentence(gene_symbol, pathways, max_items=8):

    if not pathways:
        return "%s: 알려진 Reactome 경로 정보가 없습니다." % gene_symbol

    items = sorted(pathways)
    text = ", ".join(items[:max_items]) + (" 등" if len(items) > max_items else "")

    return "%s: %d개 경로에 관여 -> %s (Reactome 참고)" % (gene_symbol, len(pathways), text)


def summarize_target_pathways(target_ids, gene_symbol_of, pathways_df):
    """타깃별 관여 경로 목록과 안내 문장을 만든다."""

    rows = []

    for tid in sorted(set(target_ids)):

        names = sorted(set(pathways_df.loc[pathways_df["target_id"] == tid, "pathway_name"]))
        symbol = gene_symbol_of.get(tid, tid)

        rows.append({"target_id": tid, "gene_symbol": symbol, "n_pathways": len(names),
                     "pathways": "; ".join(names), "summary": target_pathway_sentence(symbol, names)})

    return pd.DataFrame(rows, columns=["target_id", "gene_symbol", "n_pathways", "pathways", "summary"])


def annotate_candidates_with_pathways(candidates, pairs, pathways_df, max_items=6):
    """
    질병 후보(candidates)에, 그 질병을 지지하는 타깃들이 관여하는 경로를 참고 정보로 덧붙인다.
    주의: "이 경로가 이 질병을 일으킨다"는 인과 주장이 아니라, 그 타깃들이 어떤 경로의 구성원인지
    보여주는 배경 맥락(context)이다. 화합물이 억제제인지 활성제인지와는 무관하게 성립하는 정보다.
    """

    pw_of_target = pathways_df.groupby("target_id")["pathway_name"].apply(lambda s: sorted(set(s))).to_dict()

    out = candidates.copy()
    texts = []

    for _, row in out.iterrows():

        sub = pairs[pairs["disease_id"] == row["disease_id"]]
        names = set()

        for tid in sub["target_id"].unique():
            names.update(pw_of_target.get(tid, []))

        names = sorted(names)
        text = "; ".join(names[:max_items]) + (" 등" if len(names) > max_items else "")

        texts.append(text)

    out["supporting_pathways"] = texts

    return out


def fetch_evidence_detail(con, target_ids):
    """
    ref_target_disease_evidence_detail 조회 (증거 종류별 점수: genetic_association, known_drug, literature 등).
    이 테이블은 선택 사항(add_evidence_detail_table.sql 로 만들고 load_open_targets.py 로 채워야 함)이라,
    테이블이 없으면 오류 대신 빈 결과 + 안내만 출력한다.
    """

    ids = sorted(set(target_ids))
    empty = pd.DataFrame(columns=["target_id", "disease_id", "datatype", "max_score"])

    if not ids:
        return empty

    try:
        query = ("SELECT target_id, disease_id, datatype, max_score FROM ref_target_disease_evidence_detail "
                 "WHERE target_id IN (%s)" % _in_clause(ids))
        return pd.read_sql_query(query, con, params=ids)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        print("[참고] ref_target_disease_evidence_detail 테이블이 없어 증거 종류별 내역은 생략합니다.")
        print("       -> add_evidence_detail_table.sql 실행 + load_open_targets.py 재실행하면 채워집니다.")
        return empty


def annotate_pairs_with_evidence_types(pairs, detail, max_items=6):
    """(타깃, 질병) 쌍마다 'genetic_association(0.90); known_drug(0.60)' 형태의 근거 내역을 붙인다."""

    out = pairs.copy()

    if detail.empty:
        out["evidence_types"] = ""
        return out

    d = detail.sort_values("max_score", ascending=False).copy()
    d["item"] = d["datatype"] + "(" + d["max_score"].round(2).astype(str) + ")"

    grouped = d.groupby(["target_id", "disease_id"])["item"].apply(
        lambda s: "; ".join(s.head(max_items))).reset_index(name="evidence_types")

    out = out.merge(grouped, on=["target_id", "disease_id"], how="left")
    out["evidence_types"] = out["evidence_types"].fillna("")

    return out


def annotate_candidates_with_evidence_types(candidates, pairs, detail, max_items=6):
    """
    질병 후보마다, 그 질병을 지지하는 타깃들 전체에 걸쳐 어떤 "종류"의 증거가 있었는지
    (예: genetic_association, known_drug, literature) 를 모아 참고 정보로 붙인다.
    """

    out = candidates.copy()

    if detail.empty:
        out["evidence_types"] = ""
        return out

    types_by_pair = detail.groupby(["target_id", "disease_id"])["datatype"].apply(set).to_dict()
    texts = []

    for _, row in out.iterrows():

        sub = pairs[pairs["disease_id"] == row["disease_id"]]
        names = set()

        for tid in sub["target_id"].unique():
            names.update(types_by_pair.get((tid, row["disease_id"]), set()))

        names = sorted(names)
        texts.append(", ".join(names[:max_items]) + (" 등" if len(names) > max_items else ""))

    out["evidence_types"] = texts

    return out


def build_candidates(evidence_df, targets_df, per_target_top_k=PER_TARGET_TOP_K, max_candidates=MAX_CANDIDATES):
    """
    타깃별 상위 질병을 모아 질병 후보군을 만든다.
    정렬: 그 질병을 가리키는 타깃 수(많을수록) -> 최고 증거 점수.
    반환: (후보 질병 표, 후보에 포함된 (타깃, 질병) 쌍 표)
    """

    ev = evidence_df.merge(
        targets_df[["target_id", "gene_symbol"]].drop_duplicates("target_id"), on="target_id", how="left"
    )
    ev["gene_symbol"] = ev["gene_symbol"].fillna(ev["target_id"])

    ev = ev.sort_values(["target_id", "overall_score"], ascending=[True, False], kind="stable")

    top = ev.groupby("target_id", sort=False).head(per_target_top_k) if per_target_top_k > 0 else ev

    agg = top.groupby("disease_id", sort=False).agg(
        disease_name=("disease_name", "first"),
        n_targets=("target_id", "nunique"),
        max_score=("overall_score", "max"),
        mean_score=("overall_score", "mean"),
        targets=("gene_symbol", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()

    agg = agg.sort_values(["n_targets", "max_score"], ascending=[False, False], kind="stable").reset_index(drop=True)

    if max_candidates and max_candidates > 0:
        agg = agg.head(max_candidates)

    pairs = top[top["disease_id"].isin(set(agg["disease_id"]))].copy()

    return agg, pairs


# ============================================================
# 3. PrimeKG 노드 / 약물-단백질 연결
# ============================================================

def _extract_pairs(chunk, relation, type_a, type_b, cols_a, cols_b):
    """relation 으로 연결된 두 노드를 양방향(x->y, y->x) 모두 표준 컬럼명으로 뽑는다."""

    rel = chunk[chunk["relation"] == relation]

    if not len(rel):
        return None

    fwd = rel[(rel["x_type"] == type_a) & (rel["y_type"] == type_b)]
    rev = rel[(rel["x_type"] == type_b) & (rel["y_type"] == type_a)]

    a = fwd[["x_id", "x_name", "y_id", "y_name"]]
    a.columns = cols_a + cols_b
    b = rev[["y_id", "y_name", "x_id", "x_name"]]
    b.columns = cols_a + cols_b

    return pd.concat([a, b]).drop_duplicates()


def load_primekg_nodes(path, chunksize=500000):
    """
    kg.csv 에서 질병/약물/유전자/부작용 노드와 drug_protein, drug_effect 연결을 chunk 로 읽어 모은다.
    반환 dict: disease, drug, effect (columns: index, id, name)
              drug_protein (drug_id, drug_name, gene_id, gene_name)
              drug_effect  (drug_id, drug_name, effect_id, effect_name)   -- PrimeKG 에 등록된 "알려진" 부작용(SIDER 기반)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            "PrimeKG kg.csv 를 찾을 수 없습니다: %s\n  -> Harvard Dataverse(doi:10.7910/DVN/IXA7BM)의 kg.csv 를 이 경로에 두세요." % path
        )

    usecols = ["relation", "x_index", "x_id", "x_type", "x_name", "y_index", "y_id", "y_type", "y_name"]

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in usecols if c not in header]

    if missing:
        raise RuntimeError("kg.csv 에 필요한 컬럼이 없습니다: %s\n실제 컬럼: %s" % (missing, header))

    node_parts = {"disease": [], "drug": [], "effect/phenotype": []}
    dp_parts, de_parts = [], []
    total = 0

    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, chunksize=chunksize):

        total += len(chunk)

        for side in ("x", "y"):
            for kind in node_parts:
                sub = chunk[chunk[side + "_type"] == kind][[side + "_index", side + "_id", side + "_name"]]
                sub.columns = ["index", "id", "name"]
                node_parts[kind].append(sub.drop_duplicates())

        dp = _extract_pairs(chunk, "drug_protein", "drug", "gene/protein", ["drug_id", "drug_name"], ["gene_id", "gene_name"])
        if dp is not None:
            dp_parts.append(dp)

        de = _extract_pairs(chunk, "drug_effect", "drug", "effect/phenotype", ["drug_id", "drug_name"], ["effect_id", "effect_name"])
        if de is not None:
            de_parts.append(de)

    nodes = {k: pd.concat(v, ignore_index=True).drop_duplicates("id").reset_index(drop=True) for k, v in node_parts.items()}

    drug_protein = pd.concat(dp_parts, ignore_index=True).drop_duplicates() if dp_parts \
        else pd.DataFrame(columns=["drug_id", "drug_name", "gene_id", "gene_name"])
    drug_effect = pd.concat(de_parts, ignore_index=True).drop_duplicates() if de_parts \
        else pd.DataFrame(columns=["drug_id", "drug_name", "effect_id", "effect_name"])

    print("PrimeKG 로드: %s행 -> 질병 %d, 약물 %d, 부작용 %d, 약물-단백질 %d, 약물-부작용 %d"
          % (format(total, ","), len(nodes["disease"]), len(nodes["drug"]), len(nodes["effect/phenotype"]),
             len(drug_protein), len(drug_effect)))

    return {"disease": nodes["disease"], "drug": nodes["drug"], "effect": nodes["effect/phenotype"],
            "drug_protein": drug_protein, "drug_effect": drug_effect}


# ============================================================
# 4. Open Targets 질병 -> PrimeKG 질병 매핑
# ============================================================

_RE_OT_MONDO = re.compile(r"(?i)^MONDO[_:]0*(\d+)$")


def normalize_name(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def ot_mondo_number(identifier):
    """'MONDO_0005147' / 'MONDO:0005147' -> '5147' (아니면 None)."""
    m = _RE_OT_MONDO.match(str(identifier).strip())
    return str(int(m.group(1))) if m else None


def primekg_mondo_numbers(node_id):
    """PrimeKG 질병 id(숫자 / 'MONDO:0005147' / 묶음 '5044_5039')에서 MONDO 숫자들을 뽑는다."""
    return {str(int(t)) for t in re.findall(r"\d+", str(node_id))}


def build_disease_indexes(prime_disease):

    by_mondo, by_name = {}, {}

    for i, (nid, name) in enumerate(zip(prime_disease["id"], prime_disease["name"])):
        for n in primekg_mondo_numbers(nid):
            by_mondo.setdefault(n, []).append(i)
        by_name.setdefault(normalize_name(name), []).append(i)

    return by_mondo, by_name


def meta_from_frames(disease_df, synonym_df=None):
    """
    Open Targets disease 표(id, name, dbXRefs, ancestors[, exact synonyms]) -> {id: {...}} 로 변환.
    """

    synonyms = {}

    if synonym_df is not None:
        for i, s in zip(synonym_df["id"], synonym_df["exact"]):
            synonyms[i] = [x for x in list(s)] if s is not None else []

    meta = {}

    for _, r in disease_df.iterrows():
        meta[r["id"]] = {
            "name": r["name"],
            "xrefs": [str(x) for x in list(r["dbXRefs"])] if r["dbXRefs"] is not None else [],
            "ancestors": [str(x) for x in list(r["ancestors"])] if r["ancestors"] is not None else [],
            "synonyms": synonyms.get(r["id"], []),
        }

    return meta


def load_ot_disease_meta(parquet_dir):
    """Open Targets disease parquet 를 읽어 dbXRefs / ancestors / 동의어를 얻는다 (없으면 None)."""

    if not parquet_dir or not os.path.isdir(parquet_dir):
        print("[참고] Open Targets disease parquet 폴더가 없어 dbXRefs 기반 매핑은 생략합니다: %s" % parquet_dir)
        return None

    pattern = os.path.join(parquet_dir, "**", "*.parquet").replace("\\", "/").replace("'", "''")

    try:
        import duckdb
    except ImportError:
        print("[참고] duckdb 가 없어 disease parquet 를 읽지 못했습니다 (pip install duckdb).")
        return None

    con = duckdb.connect()

    try:
        disease_df = con.execute(
            "SELECT id, name, dbXRefs, ancestors FROM read_parquet('%s', hive_partitioning=false)" % pattern
        ).fetchdf()
    except Exception as e:
        print("[참고] disease parquet 읽기 실패 (%s) - dbXRefs 기반 매핑은 생략합니다." % e)
        return None

    synonym_df = None

    try:
        synonym_df = con.execute(
            "SELECT id, synonyms.hasExactSynonym AS exact FROM read_parquet('%s', hive_partitioning=false)" % pattern
        ).fetchdf()
    except Exception:
        pass

    con.close()

    meta = meta_from_frames(disease_df, synonym_df)

    print("Open Targets disease 메타 로드: %d개 (dbXRefs/ancestors%s)"
          % (len(meta), "/동의어" if synonym_df is not None else ""))

    return meta


def map_ot_diseases(candidates, prime_disease, ot_meta=None, allow_ancestor=False):
    """
    후보 질병(disease_id, disease_name) 각각을 PrimeKG 질병 노드에 매핑한다. 우선순위와 표시:
        mondo_direct  : Open Targets id 가 MONDO_xxxx 이고 PrimeKG 에 같은 MONDO 가 있음
        mondo_xref    : OT 의 dbXRefs 에 MONDO:xxxx 가 있고 PrimeKG 에 있음
        name_exact    : 이름(또는 OT 정확 동의어)이 PrimeKG 질병 이름과 정규화 후 동일
        ancestor_mondo: (옵션) 매핑되는 상위 질병 중 가장 깊은 것 -> is_broader=True (더 넓은 질병이므로 주의)
        unmapped
    """

    by_mondo, by_name = build_disease_indexes(prime_disease)

    def rows_for_mondo(nums):
        for n in nums:
            if n in by_mondo:
                return by_mondo[n], n
        return None, None

    def xref_numbers(meta):
        nums = []
        for x in (meta or {}).get("xrefs", []):
            n = ot_mondo_number(str(x).replace(":", "_", 1))
            if n:
                nums.append(n)
        return nums

    def pick(rows, name):
        if len(rows) == 1:
            return rows[0]
        target = normalize_name(name)
        for r in rows:
            if normalize_name(prime_disease.loc[r, "name"]) == target:
                return r
        return rows[0]

    results = []

    for did, dname in zip(candidates["disease_id"], candidates["disease_name"]):

        meta = ot_meta.get(did) if ot_meta else None
        method, via, rows, broader = "unmapped", None, None, False

        n = ot_mondo_number(did)
        if n and n in by_mondo:
            method, via, rows = "mondo_direct", did, by_mondo[n]

        if rows is None and meta:
            r, n = rows_for_mondo(xref_numbers(meta))
            if r:
                method, via, rows = "mondo_xref", "MONDO:%s" % n, r

        if rows is None:
            names = [dname] + list((meta or {}).get("synonyms", []))
            for cand in names:
                r = by_name.get(normalize_name(cand))
                if r:
                    method, via, rows = "name_exact", cand, r
                    break

        if rows is None and allow_ancestor and meta and ot_meta:
            best = None
            for anc in meta["ancestors"]:
                anc_meta = ot_meta.get(anc)
                if not anc_meta:
                    continue
                nums = ([ot_mondo_number(anc)] if ot_mondo_number(anc) else []) + xref_numbers(anc_meta)
                r, n = rows_for_mondo([x for x in nums if x])
                if r:
                    depth = len(anc_meta["ancestors"])          # 조상이 많을수록 더 깊은(구체적인) 상위 질병
                    if best is None or depth > best[0]:
                        best = (depth, anc, r)
            if best:
                method, via, rows, broader = "ancestor_mondo", best[1], best[2], True

        out = {"disease_id": did, "map_method": method, "map_via": via, "is_broader": broader,
               "primekg_x_id": None, "primekg_x_index": None, "primekg_name": None, "n_matches": 0}

        if rows:
            r = pick(rows, dname)
            out.update({"primekg_x_id": prime_disease.loc[r, "id"],
                        "primekg_x_index": prime_disease.loc[r, "index"],
                        "primekg_name": prime_disease.loc[r, "name"],
                        "n_matches": len(rows)})

        results.append(out)

    return pd.DataFrame(results)


# ============================================================
# 5. 처음 입력한 약물 -> PrimeKG 약물 노드
# ============================================================

def fetch_pubchem_synonyms(smiles, cache_dir=None, timeout=20):
    """SMILES 로 PubChem 동의어 목록을 얻는다. 반환: (list 또는 None, status)."""

    key = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache_dir, key + ".json") if cache_dir else None

    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), "cache"
        except (OSError, ValueError):
            pass

    try:
        import requests
    except ImportError:
        return None, "requests_missing"

    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/synonyms/JSON"

    try:
        r = requests.post(url, data={"smiles": smiles}, timeout=timeout)
    except Exception:
        return None, "connection_failed"

    if r.status_code == 404:
        return None, "not_found"

    if r.status_code != 200:
        return None, "http_%d" % r.status_code

    synonyms = []

    for info in (r.json().get("InformationList", {}).get("Information", [])):
        synonyms.extend(info.get("Synonym", []))

    synonyms = synonyms[:1000]

    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False)

    return synonyms, "fetched"


def resolve_query_drug(prime_drug, smiles=None, drug_id=None, drug_name=None, synonym_fetcher=None):
    """
    처음 입력한 약물을 PrimeKG 약물 노드(DrugBank ID)에 연결한다.
    우선순위: --drug-id > --drug-name > SMILES(PubChem 동의어의 DrugBank ID / 이름)
    """

    by_id = {i: k for k, i in enumerate(prime_drug["id"])}
    by_name = {}

    for k, n in enumerate(prime_drug["name"]):
        by_name.setdefault(str(n).lower(), k)

    def result(k, status, via):
        return {"found": True, "drug_id": prime_drug.loc[k, "id"], "drug_name": prime_drug.loc[k, "name"],
                "primekg_index": prime_drug.loc[k, "index"], "status": status, "matched_via": via}

    if drug_id:
        if drug_id in by_id:
            return result(by_id[drug_id], "given_id", drug_id)
        return {"found": False, "status": "given_id_not_in_primekg", "matched_via": drug_id}

    if drug_name and drug_name.lower() in by_name:
        return result(by_name[drug_name.lower()], "given_name", drug_name)

    if smiles and synonym_fetcher:

        synonyms, fetch_status = synonym_fetcher(smiles)

        if synonyms:
            for s in synonyms:
                if DRUGBANK_RE.match(s) and s in by_id:
                    return result(by_id[s], "pubchem_drugbank_id", s)
            for s in synonyms:
                if s.lower() in by_name:
                    return result(by_name[s.lower()], "pubchem_name", s)
            return {"found": False, "status": "pubchem_synonyms_not_in_primekg", "matched_via": None}

        return {"found": False, "status": "pubchem_unavailable(%s)" % fetch_status, "matched_via": None}

    return {"found": False, "status": "no_identifier", "matched_via": None}


def proxy_drugs_for_targets(drug_protein, prime_drug, symbols, max_drugs=20):
    """Top5 타깃을 표적으로 하는 PrimeKG 약물 (여러 타깃을 동시에 표적으로 하는 약물 우선)."""

    wanted = {str(s).upper() for s in symbols if s}

    sub = drug_protein[drug_protein["gene_name"].str.upper().isin(wanted)]

    if not len(sub):
        return pd.DataFrame(columns=["drug_id", "drug_name", "primekg_index", "n_targets", "targets"])

    agg = sub.groupby(["drug_id", "drug_name"]).agg(
        n_targets=("gene_name", "nunique"),
        targets=("gene_name", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()

    agg = agg.sort_values(["n_targets", "drug_name"], ascending=[False, True], kind="stable").head(max_drugs)

    index_of = dict(zip(prime_drug["id"], prime_drug["index"]))
    agg["primekg_index"] = agg["drug_id"].map(index_of)

    return agg[["drug_id", "drug_name", "primekg_index", "n_targets", "targets"]].reset_index(drop=True)


def known_side_effects(drug_effect, drug_id):
    """
    PrimeKG(drug_effect, SIDER 기반)에 등록된 이 약물의 알려진 부작용을 조회한다.
    반환: DataFrame(effect_id, effect_name) - 없으면 빈 DataFrame (오류가 아니라 "정보가 없다"는 뜻).
    """

    rows = drug_effect[drug_effect["drug_id"] == drug_id][["effect_id", "effect_name"]].drop_duplicates()

    return rows.sort_values("effect_name").reset_index(drop=True)


# ============================================================
# 6. prepare
# ============================================================

def find_latest(results_dir, pattern):
    found = sorted(glob.glob(os.path.join(results_dir, pattern)), key=os.path.getmtime)
    return found[-1] if found else None


def find_final_csv(results_dir):
    """
    최종 단백질 CSV 를 자동으로 찾는다 (가장 최근 파일).
      1) results\\stage2_*\\final*_proteins.csv   (Step2 가 만든 결과)
      2) results\\final*_proteins.csv             (직접 넣어둔 파일)
    """

    found = []

    for pattern in (os.path.join("stage2_*", "final*_proteins.csv"), "final*_proteins.csv"):
        found.extend(glob.glob(os.path.join(results_dir, pattern)))

    return max(found, key=os.path.getmtime) if found else None


def read_step2_smiles(final_csv):
    """Step2 결과 폴더의 run_config.json 에서 입력 SMILES 를 읽는다."""

    cfg = os.path.join(os.path.dirname(os.path.abspath(final_csv)), "run_config.json")

    if os.path.exists(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                return json.load(f).get("smiles")
        except (OSError, ValueError):
            return None

    return None


def run_step2_pipeline(smiles, out_dir, log=print):
    """
    final5_proteins.csv 가 아직 없을 때, Step 1+2 통합 스크립트(step2_Top20_regression.py, 그
    안에서 Step 1 도 자동 실행됨)를 같은 환경(p_proj)에서 subprocess 로 돌려서 만들어낸다.
    반환: 새로 생긴 final5_proteins.csv 경로. 실패하면 예외를 던진다.
    """

    step2_script = os.path.join(PROJECT_DIR, "model", "step2_Top20_regression.py")

    if not os.path.exists(step2_script):
        raise FileNotFoundError(
            "%s 를 찾을 수 없습니다. step2_Top20_regression.py 위치를 확인하세요." % step2_script
        )

    # sys.executable(지금 이 스크립트를 실행 중인 파이썬)을 그냥 쓰면, p_proj 를 활성화하지 않고
    # 이 스크립트를 실행했을 때(다른/기본 환경) torch 등이 없어 Step 2 가 바로 실패한다.
    # 그래서 "p_proj" 환경을 명시적으로 찾아서 그 파이썬을 쓴다 (없으면 sys.executable 로 대체).
    p_proj_python = _find_env_python("p_proj") or sys.executable

    if p_proj_python == sys.executable:
        log("[참고] p_proj 환경을 찾지 못해 지금 파이썬(%s)으로 실행합니다. "
            "torch 관련 오류가 나면 'conda activate p_proj' 후 다시 시도하세요." % sys.executable)

    log("\n[자동 실행] final5_proteins.csv 가 없어 Step 1+2 파이프라인을 먼저 실행합니다.")
    log("  (SMILES=%s, 몇 분 정도 걸릴 수 있습니다)" % smiles)
    log("  python(p_proj) = %s" % p_proj_python)

    result = subprocess.run(
        [p_proj_python, step2_script, "--smiles", smiles],
        cwd=os.path.dirname(step2_script)
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Step 1+2 실행이 실패했습니다 (종료 코드 %d). 위쪽 출력을 확인하세요." % result.returncode
        )

    final_csv = find_final_csv(out_dir)

    if not final_csv:
        raise RuntimeError("Step 1+2 는 끝났는데 final5_proteins.csv 를 찾지 못했습니다.")

    log("[자동 실행] Step 1+2 완료: %s" % final_csv)

    return final_csv


def run_txgnn_score(run_dir, log=print):
    """
    DTI_txgnn conda 환경의 python 으로 step3_txgnn_score.py 를 subprocess 로 실행한다.
    (dgl 버전 충돌 때문에 이 스크립트와 같은 프로세스/환경에서는 돌릴 수 없다.)
    반환: (성공 여부, txgnn_scores.csv 경로 또는 None)
    """

    dti_python = _find_env_python("DTI_txgnn")

    if not dti_python:
        log("\n[실패] DTI_txgnn conda 환경을 찾지 못했습니다 (…\\envs\\DTI_txgnn\\python.exe 없음).")
        log("       TxGNN 점수 계산을 건너뜁니다. 수동으로 실행하려면:")
        log("       conda activate DTI_txgnn && python step3_txgnn_score.py --run-dir \"%s\"" % run_dir)
        return False, None

    score_script = os.path.join(PROJECT_DIR, "model", "DTI_txgnn_conda_python_3_11", "step3_txgnn_score.py")

    if not os.path.exists(score_script):
        log("\n[실패] %s 를 찾을 수 없습니다." % score_script)
        return False, None

    log("\n[자동 실행] TxGNN 점수 계산 (DTI_txgnn 환경) 시작 -- 몇 분에서 수십 분 걸릴 수 있습니다.")
    log("  python(DTI_txgnn) = %s" % dti_python)

    result = subprocess.run(
        [dti_python, score_script, "--run-dir", run_dir],
        cwd=os.path.dirname(score_script)
    )

    scores_path = os.path.join(run_dir, "txgnn_scores.csv")

    if result.returncode != 0 or not os.path.exists(scores_path):
        log("\n[실패] TxGNN 점수 계산이 실패했거나 결과 파일이 없습니다 (종료 코드 %d)." % result.returncode)
        log("       수동으로 다시 시도하려면:")
        log("       conda activate DTI_txgnn && python step3_txgnn_score.py --run-dir \"%s\"" % run_dir)
        return False, None

    log("\n[자동 실행] TxGNN 점수 계산 완료: %s" % scores_path)

    return True, scores_path


def cmd_prepare(args, fetch_ensembl=None, fetch_synonyms=None):

    final_csv = args.final_csv or find_final_csv(args.out_dir)

    if (not final_csv or not os.path.exists(final_csv)) and args.auto_chain and args.smiles:
        final_csv = run_step2_pipeline(args.smiles, args.out_dir)

    if not final_csv or not os.path.exists(final_csv):
        raise FileNotFoundError(
            "최종 단백질 CSV 를 찾을 수 없습니다.\n"
            "  찾아본 위치: %s\\stage2_*\\final*_proteins.csv  /  %s\\final*_proteins.csv\n"
            "  -> 파일을 위 위치에 두거나, --final-csv \"전체경로\" 로 직접 지정하거나,\n"
            "     --auto-chain --smiles \"...\" 로 Step 1+2 부터 자동 실행하세요."
            % (args.out_dir, args.out_dir)
        )

    final_df = pd.read_csv(final_csv, dtype={"id": str, "uniprot_id": str}, keep_default_na=False, encoding="utf-8-sig")

    for col in ("id", "uniprot_id"):
        if col not in final_df.columns:
            raise RuntimeError("최종 단백질 CSV 에 %s 컬럼이 없습니다. 컬럼: %s" % (col, list(final_df.columns)))

    smiles = args.smiles or read_step2_smiles(final_csv)

    analysis_id = str(uuid.uuid4())

    print("=" * 80)
    print("Step 3 prepare: 최종 단백질 -> Open Targets 질병 후보 -> PrimeKG 매핑")
    print("=" * 80)
    print("최종 단백질: %s (%d개)" % (final_csv, len(final_df)))
    print("입력 SMILES: %s" % (smiles or "(없음 - --smiles 또는 --drug-id/--drug-name 필요)"))
    print("analysis_id: %s" % analysis_id)

    fetch_ensembl = fetch_ensembl or (lambda acc: fetch_uniprot_ensembl(acc, cache_dir=args.uniprot_cache_dir))
    fetch_synonyms = fetch_synonyms or (lambda s: fetch_pubchem_synonyms(s, cache_dir=args.pubchem_cache_dir))

    # ---- 1) 타깃 매핑 + 근거 ----
    con = open_ot_db(args.ot_db, read_only=True)

    counts = check_ot_schema(con)
    print("\nOpen Targets DB: 근거 %s행, pathway %s행" % (format(counts["ref_target_disease_evidence"], ","),
                                                        format(counts["ref_target_pathway"], ",")))

    target_map, targets_df = resolve_targets(final_df, fetch_ensembl, con)

    print("\n[타깃 매핑]")
    print(target_map[["id", "uniprot_id", "gene_symbol", "target_ids_used", "status"]].to_string(index=False, na_rep="-"))

    if targets_df.empty:
        con.close()
        raise RuntimeError("Open Targets DB 에서 근거가 있는 타깃을 하나도 찾지 못했습니다. 위 status 를 확인하세요.")

    evidence = fetch_evidence(con, targets_df["target_id"], args.min_score)
    phenotypes = fetch_target_phenotypes(con, targets_df["target_id"])
    pathways = fetch_target_pathways(con, targets_df["target_id"])
    evidence_detail = fetch_evidence_detail(con, targets_df["target_id"])
    con.close()

    print("\n타깃별 질병 근거: %s행" % format(len(evidence), ","))

    candidates, pairs = build_candidates(evidence, targets_df, args.per_target_top_k, args.max_candidates)

    gene_symbol_of = dict(zip(targets_df["target_id"], targets_df["gene_symbol"].fillna(targets_df["target_id"])))
    target_safety = summarize_target_safety(targets_df["target_id"], gene_symbol_of, phenotypes)

    target_pathways = summarize_target_pathways(targets_df["target_id"], gene_symbol_of, pathways)

    print("\n[타깃 관련 생물학적 경로 (Reactome)] -- 화합물의 억제/활성 여부와 무관하게 성립하는 배경 정보입니다")
    for line in target_pathways["summary"]:
        print("  " + line)

    print("\n[타깃 안전성 정보 (Open Targets)]")
    for line in target_safety["summary"]:
        print("  " + line)

    print("질병 후보군: %d개 (타깃당 상위 %s개, 최대 %s개)" % (len(candidates), args.per_target_top_k or "전부", args.max_candidates or "제한없음"))
    print("  2개 이상 타깃이 공통으로 가리키는 질병: %d개" % int((candidates["n_targets"] >= 2).sum()))

    # ---- 2) PrimeKG 매핑 ----
    prime = load_primekg_nodes(args.primekg)

    ot_meta = load_ot_disease_meta(args.ot_disease_dir)

    mapping = map_ot_diseases(candidates, prime["disease"], ot_meta, allow_ancestor=args.allow_ancestor)

    pairs = annotate_pairs_with_evidence_types(pairs, evidence_detail)

    cand = candidates.merge(mapping, on="disease_id", how="left")
    cand = annotate_candidates_with_pathways(cand, pairs, pathways)
    cand = annotate_candidates_with_evidence_types(cand, pairs, evidence_detail)

    print("\n[질병 ID 매핑: Open Targets -> PrimeKG]")
    print(cand["map_method"].value_counts().to_string())
    mapped = cand[cand["map_method"] != "unmapped"]
    print("매핑됨 %d / %d (%.0f%%)" % (len(mapped), len(cand), 100.0 * len(mapped) / max(len(cand), 1)))

    if len(cand) - len(mapped):
        print("  매핑 안 된 질병 예시:", "; ".join(cand[cand["map_method"] == "unmapped"]["disease_name"].head(5)))

    # ---- 3) 약물 ----
    drug = resolve_query_drug(prime["drug"], smiles=smiles, drug_id=args.drug_id, drug_name=args.drug_name,
                              synonym_fetcher=fetch_synonyms)

    drug_rows = []
    side_effects = None

    if drug["found"]:
        print("\n[입력 약물] PrimeKG 약물 노드 확인: %s (%s) via %s [%s]"
              % (drug["drug_id"], drug["drug_name"], drug["matched_via"], drug["status"]))
        drug_rows.append({"role": "query", "drug_id": drug["drug_id"], "drug_name": drug["drug_name"],
                          "primekg_index": drug["primekg_index"], "n_targets": None, "targets": None,
                          "matched_via": drug["matched_via"]})

        side_effects = known_side_effects(prime["drug_effect"], drug["drug_id"])

        print("\n[알려진 부작용] PrimeKG (SIDER 기반) 조회 결과:")
        if len(side_effects):
            print("  %d개 -> %s%s" % (len(side_effects), "; ".join(side_effects["effect_name"].head(10)),
                                     " ..." if len(side_effects) > 10 else ""))
        else:
            print("  PrimeKG 에 이 약물의 알려진 부작용 정보가 없습니다.")
    else:
        print("\n[입력 약물] PrimeKG 에서 찾지 못함 (%s)" % drug["status"])
        print("  TxGNN 은 PrimeKG 약물 노드만 점수를 낼 수 있습니다. 알려진 약물이면 --drug-id DBxxxxx 또는 --drug-name 을 지정하세요.")
        print("\n[알려진 부작용] 약물을 PrimeKG 에서 찾지 못해 조회할 수 없습니다 (알려진 정보가 없습니다).")

    symbols = targets_df["gene_symbol"].dropna().unique().tolist()

    if args.proxy_from_targets or (not drug["found"] and args.proxy_if_missing):
        proxies = proxy_drugs_for_targets(prime["drug_protein"], prime["drug"], symbols, args.max_proxy_drugs)
        print("  대리(proxy) 약물: Top 타깃을 표적으로 하는 PrimeKG 약물 %d개 (입력 화합물 자체의 점수가 아님)" % len(proxies))
        for _, r in proxies.iterrows():
            drug_rows.append({"role": "proxy", "drug_id": r["drug_id"], "drug_name": r["drug_name"],
                              "primekg_index": r["primekg_index"], "n_targets": r["n_targets"],
                              "targets": r["targets"], "matched_via": "drug_protein"})

    drugs_df = pd.DataFrame(drug_rows, columns=["role", "drug_id", "drug_name", "primekg_index",
                                                "n_targets", "targets", "matched_via"])

    # ---- 저장 ----
    # 같은 초에 여러 번 실행해도 폴더가 겹치지 않도록 analysis_id 앞부분을 붙인다
    run_dir = os.path.join(args.out_dir, "step3_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), analysis_id[:6]))
    os.makedirs(run_dir, exist_ok=True)

    enc = "utf-8-sig"

    target_map.to_csv(os.path.join(run_dir, "ot_target_map.csv"), index=False, encoding=enc)
    pairs.to_csv(os.path.join(run_dir, "ot_target_disease_pairs.csv"), index=False, encoding=enc)
    cand.to_csv(os.path.join(run_dir, "ot_disease_candidates.csv"), index=False, encoding=enc)

    diseases_out = mapped.drop_duplicates("primekg_x_id")[
        ["disease_id", "disease_name", "primekg_x_id", "primekg_x_index", "primekg_name", "map_method", "is_broader"]
    ].rename(columns={"disease_id": "ot_disease_id", "disease_name": "ot_disease_name"})

    diseases_out.to_csv(os.path.join(run_dir, "txgnn_diseases.csv"), index=False, encoding=enc)
    drugs_df.to_csv(os.path.join(run_dir, "txgnn_drugs.csv"), index=False, encoding=enc)

    if side_effects is not None:
        side_effects.to_csv(os.path.join(run_dir, "known_side_effects.csv"), index=False, encoding=enc)

    target_safety.to_csv(os.path.join(run_dir, "target_safety_summary.csv"), index=False, encoding=enc)
    target_pathways.to_csv(os.path.join(run_dir, "target_pathways_summary.csv"), index=False, encoding=enc)

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"analysis_id": analysis_id, "smiles": smiles, "final_csv": final_csv,
                   "compound_name": args.compound_name or (drug.get("drug_name") if drug["found"] else None),
                   "query_drug_status": drug["status"], "query_drug_id": drug.get("drug_id"),
                   "ot_db": args.ot_db, "primekg": args.primekg,
                   "min_score": args.min_score, "per_target_top_k": args.per_target_top_k,
                   "max_candidates": args.max_candidates, "allow_ancestor": args.allow_ancestor,
                   "n_known_side_effects": (int(len(side_effects)) if side_effects is not None else None)},
                  f, indent=2, ensure_ascii=False)

    print("\n저장 완료: %s" % run_dir)

    if not args.auto_chain:
        print("  다음: TxGNN 환경에서  python step3_txgnn_score.py --run-dir \"%s\" ..." % run_dir)
        print("        점수가 나오면  python step3_ot_txgnn_link.py finalize --run-dir \"%s\" --txgnn-scores <txgnn_scores.csv>" % run_dir)
        return run_dir

    # ---------------- --auto-chain: TxGNN 점수 계산 -> finalize 까지 한 번에 -------------
    ok, scores_path = run_txgnn_score(run_dir)

    if not ok:
        print("\n[중단] TxGNN 점수 계산에 실패해 finalize 는 건너뜁니다. (prepare 결과 자체는 %s 에 저장되어 있습니다)" % run_dir)
        return run_dir

    print("\n" + "=" * 80)
    print("이어서 finalize 자동 실행")
    print("=" * 80)

    finalize_args = argparse.Namespace(
        run_dir=run_dir, out_dir=args.out_dir, ot_db=args.ot_db, txgnn_scores=scores_path,
        write_db=args.write_db, compound_name=args.compound_name,
        contra_as_side_effect=False, show=20, no_chart=False, chart_top_n=20
    )

    cmd_finalize(finalize_args)

    return run_dir


# ============================================================
# 7. finalize
# ============================================================

def merge_txgnn_scores(candidates, scores, drugs):
    """
    후보 질병 표에 TxGNN 순위를 합친다.
      - role=query 약물의 순위가 있으면 txgnn_rank (source=query)
      - 없으면 proxy 약물 중 최고 순위 (source=proxy, txgnn_best_drug 표시)
      - relation=contraindication 은 txgnn_contra_rank 로 별도 보관 (같은 방식: 낮을수록 금기 가능성이 높다는 뜻)
      - known_indication 컬럼이 있으면 txgnn_known_in_kg(yes/no: 이미 KG 에 알려진 적응증)도 채운다
      - score_raw 컬럼이 있으면 txgnn_score_raw(시그모이드 없는 원점수)도 참고용으로 채운다.
      - rank_percentile 컬럼이 있으면 txgnn_top_pct("상위 X%", 0=1등에 가까움)도 채운다.
        원점수(txgnn_score_raw)보다 이게 더 직관적이면서도 질병 간 비교가 가능한 지표다.

    질병 간 비교(정렬)는 반드시 txgnn_rank 로 해야 한다. txgnn_score_raw 는 참고용일 뿐 질병끼리
    직접 비교하면 안 된다 -- 질병마다 점수 분포의 스케일이 다를 수 있어서 왜곡된다.
    txgnn_rank는 "그 질병 안에서 전체 약물 몇 개 중 몇 등인가" 라서 질병이 달라도 같은 기준
    (전체 약물 수가 항상 같음)으로 비교할 수 있다.
    scores 컬럼: drug_id, primekg_x_id, score_raw, [relation, rank, known_indication]
    """

    out = candidates.copy()
    out["primekg_x_id"] = out["primekg_x_id"].astype("object")

    for col, default in (("txgnn_rank", float("nan")), ("txgnn_score_source", ""), ("txgnn_best_drug", ""),
                         ("txgnn_contra_rank", float("nan")), ("txgnn_known_in_kg", ""),
                         ("txgnn_score_raw", float("nan")), ("txgnn_top_pct", float("nan"))):
        out[col] = default

    if scores is None or len(scores) == 0:
        return out

    s = scores.copy()
    s["score_raw"] = pd.to_numeric(s["score_raw"], errors="coerce")
    s = s.dropna(subset=["score_raw"])

    if "relation" not in s.columns:
        s["relation"] = "indication"

    role_of = dict(zip(drugs["drug_id"], drugs["role"])) if len(drugs) else {}
    name_of = dict(zip(drugs["drug_id"], drugs["drug_name"])) if len(drugs) else {}

    s["role"] = s["drug_id"].map(role_of).fillna("query")

    def best_for(rel):

        sub = s[s["relation"] == rel].copy()

        if "rank" in sub.columns:
            sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")

        result = {}

        for key, grp in sub.groupby("primekg_x_id"):

            has_rank = "rank" in grp.columns and grp["rank"].notna().any()
            # rank: 낮을수록 좋음. rank 가 없으면(하위호환) score_raw 로 대신 고른다 -- 이것도 낮을수록 좋음(확정된 방향).
            sort_col, ascending = ("rank", True) if has_rank else ("score_raw", True)

            q = grp[grp["role"] == "query"]
            if len(q):
                r, source = q.sort_values(sort_col, ascending=ascending).iloc[0], "query"
            else:
                r, source = grp.sort_values(sort_col, ascending=ascending).iloc[0], "proxy"

            rank = pd.to_numeric(r.get("rank"), errors="coerce") if "rank" in grp.columns else float("nan")
            known = str(r.get("known_indication", "")).strip().lower() if "known_indication" in grp.columns else ""
            known = "yes" if known == "true" else ("no" if known == "false" else "")
            score_raw = pd.to_numeric(r.get("score_raw"), errors="coerce")
            top_pct = pd.to_numeric(r.get("rank_percentile"), errors="coerce") * 100 if "rank_percentile" in grp.columns else float("nan")

            result[key] = (rank, source, name_of.get(r["drug_id"], r["drug_id"]), known, score_raw, top_pct)

        return result

    ind = best_for("indication")
    con = best_for("contraindication")

    for i, key in enumerate(out["primekg_x_id"]):
        if key in ind:
            out.loc[out.index[i], ["txgnn_rank", "txgnn_score_source", "txgnn_best_drug"]] = list(ind[key][:3])
            out.loc[out.index[i], "txgnn_known_in_kg"] = ind[key][3]
            out.loc[out.index[i], "txgnn_score_raw"] = ind[key][4]
            out.loc[out.index[i], "txgnn_top_pct"] = ind[key][5]
        if key in con:
            con_rank, _, _, _, _, _ = con[key]
            out.loc[out.index[i], "txgnn_contra_rank"] = con_rank

    out["txgnn_rank"] = pd.to_numeric(out["txgnn_rank"], errors="coerce")
    out["txgnn_contra_rank"] = pd.to_numeric(out["txgnn_contra_rank"], errors="coerce")
    out["txgnn_score_raw"] = pd.to_numeric(out["txgnn_score_raw"], errors="coerce")
    out["txgnn_top_pct"] = pd.to_numeric(out["txgnn_top_pct"], errors="coerce")

    return out


def rank_final(df):
    """
    정렬 기준은 txgnn_rank (전체 약물 중 몇 등, 낮을수록 좋음) 이다.
    txgnn_score(확률) 가 아니라 rank 를 쓰는 이유: score 는 시그모이드로 변환된 값이라 원점수가
    어느 정도만 유력해도(예: 로짓 -5 이하) 순식간에 1.0 근처로 포화된다. 그래서 서로 다른 질병의
    score 를 그대로 비교하면 실제로는 다른데도 다 "1.000" 으로 보여 구분이 안 될 수 있다.
    반면 rank 는 "그 질병 안에서 전체 7,957개 약물 중 몇 등인가" 라서 질병이 달라도 같은 기준으로
    비교할 수 있다 -> 질병 간 비교에는 rank 가 더 적절하다.
    """

    t = df.copy()

    t = t.sort_values(["txgnn_rank", "n_targets", "max_score"], ascending=[True, False, False],
                      na_position="last", kind="stable").reset_index(drop=True)

    t.insert(0, "rank", range(1, len(t) + 1))

    return t


def write_results_to_db(db_path, analysis_id, smiles, compound_name, final_df, pairs_df,
                        contra_as_side_effect=False):
    """
    cdss_integrated.db 의 analysis_history / step3_indication_results 에 저장한다.
      EVIDENCE_DB : Open Targets 증거 (타깃, 질병) 쌍 - confidence_score = 증거 점수
      TXGNN_AI    : TxGNN 순위 - 질병을 가리키는 각 타깃 행마다 같은 순위 (confidence_score = txgnn_rank, 낮을수록 좋음. "확률"이 아님에 주의)
    같은 analysis_id 로 다시 실행하면 이전 행을 지우고 다시 넣는다(중복 방지).
    """

    con = open_ot_db(db_path, read_only=False)

    try:
        check_ot_schema(con, need_step3_tables=True)

        cur = con.cursor()

        cur.execute("INSERT OR IGNORE INTO analysis_history (analysis_id, input_smiles, compound_name) VALUES (?, ?, ?)",
                    (analysis_id, smiles or "", compound_name))

        cur.execute("DELETE FROM step3_indication_results WHERE analysis_id = ?", (analysis_id,))

        disease_ids = set(final_df["disease_id"])
        rows = []

        for _, r in pairs_df[pairs_df["disease_id"].isin(disease_ids)].iterrows():
            rows.append((analysis_id, r["target_id"], r["disease_id"], r["disease_name"],
                         "EVIDENCE_DB", float(r["overall_score"]), 0))

        support = {}
        for _, r in pairs_df.iterrows():
            support.setdefault(r["disease_id"], []).append(r["target_id"])

        for _, r in final_df.iterrows():

            targets = sorted(set(support.get(r["disease_id"], [])))

            if pd.notna(r.get("txgnn_rank")):
                for t in targets:
                    rows.append((analysis_id, t, r["disease_id"], r["disease_name"], "TXGNN_AI", float(r["txgnn_rank"]), 0))

            if contra_as_side_effect and pd.notna(r.get("txgnn_contra_rank")):
                for t in targets:
                    rows.append((analysis_id, t, r["disease_id"], r["disease_name"], "TXGNN_AI",
                                 float(r["txgnn_contra_rank"]), 1))

        cur.executemany(
            "INSERT INTO step3_indication_results "
            "(analysis_id, target_id, disease_id, disease_name, result_type, confidence_score, is_side_effect) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

        con.commit()

        n_ev = sum(1 for r in rows if r[4] == "EVIDENCE_DB")
        n_ai = len(rows) - n_ev

    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    return n_ev, n_ai


def _svg_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def build_candidate_chart_html(final_df, top_n=20):
    """
    상위 후보 막대그래프를 순수 SVG(외부 라이브러리·인터넷 없이)로 그려 완성된 HTML 문자열로 반환한다.
    txgnn_top_pct(상위 %) 가 있는 행만 대상으로 하고, 막대 길이는 "beats"(100 - top_pct, 즉 이
    화합물이 전체 후보 약물 중 이겨낸 비율)로 정해서 막대가 길수록 더 좋은 후보가 되게 한다.
    txgnn_top_pct 가 있는 행이 하나도 없으면(TxGNN 을 아직 안 돌렸으면) None 을 반환한다.
    """

    rows = final_df[final_df["txgnn_top_pct"].notna()].head(top_n).copy()

    if rows.empty:
        return None

    row_h, gap = 34, 10
    label_w, bar_x, bar_max_w = 260, 280, 320
    top_pad, bottom_pad = 50, 30
    width = bar_x + bar_max_w + 90
    height = top_pad + len(rows) * (row_h + gap) + bottom_pad

    bars = []

    for i, (_, r) in enumerate(rows.iterrows()):

        y = top_pad + i * (row_h + gap)
        beats = max(0.0, min(100.0, 100.0 - float(r["txgnn_top_pct"])))
        bar_w = bar_max_w * beats / 100.0

        name = str(r.get("disease_name") or r.get("disease_id"))
        name = name if len(name) <= 42 else name[:39] + "..."

        known = str(r.get("txgnn_known_in_kg", "")).strip().lower() == "yes"
        color = "#2563eb" if known else "#60a5fa"

        bars.append(
            '\n    <text x="%s" y="%s" text-anchor="end" dominant-baseline="middle" '
            'font-size="13" fill="#1f2937">%s</text>'
            '\n    <rect x="%s" y="%s" width="%s" height="%s" rx="4" fill="#eef2f7"/>'
            '\n    <rect x="%s" y="%s" width="%.1f" height="%s" rx="4" fill="%s">'
            '\n      <title>%s: 상위 %.2f%% (rank %d)</title>'
            '\n    </rect>'
            '\n    <text x="%s" y="%s" dominant-baseline="middle" '
            'font-size="13" fill="#1f2937">상위 %.1f%%</text>'
            % (label_w, y + row_h / 2, _svg_escape(name),
               bar_x, y, bar_max_w, row_h,
               bar_x, y, bar_w, row_h, color,
               _svg_escape(name), float(r["txgnn_top_pct"]), int(r["txgnn_rank"]),
               bar_x + bar_max_w + 12, y + row_h / 2, float(r["txgnn_top_pct"])))

    svg = (
        '<svg width="%s" height="%s" viewBox="0 0 %s %s" xmlns="http://www.w3.org/2000/svg">'
        '\n  <rect width="%s" height="%s" fill="#ffffff"/>'
        '\n  <text x="%s" y="24" text-anchor="end" font-size="15" font-weight="600" fill="#111827">'
        '\n    TxGNN 상위 후보 (막대가 길수록 좋음, "상위 %%"는 낮을수록 좋음)'
        '\n  </text>%s'
        '\n</svg>'
        % (width, height, width, height, width, height, label_w, "".join(bars)))

    legend = ('<span style="display:inline-block;width:12px;height:12px;background:#2563eb;'
              'border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
              'PrimeKG 에 이미 알려진 적응증&nbsp;&nbsp;&nbsp;'
              '<span style="display:inline-block;width:12px;height:12px;background:#60a5fa;'
              'border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
              'TxGNN 이 새로 제시한 후보')

    html = (
        '<!DOCTYPE html>\n<html lang="ko"><head><meta charset="utf-8"><title>TxGNN 후보 질병 순위</title></head>'
        '\n<body style="font-family:-apple-system,\'Malgun Gothic\',sans-serif;max-width:%spx;margin:24px auto;">'
        '\n  <div style="font-size:12px;color:#6b7280;margin-bottom:8px">%s</div>'
        '\n  %s'
        '\n  <p style="font-size:12px;color:#6b7280;margin-top:12px">'
        '\n    막대에 마우스를 올리면 정확한 순위와 상위 %% 가 표시됩니다. 원점수(score_raw)는 disease_candidates_final.csv 를 참고하세요.'
        '\n  </p>\n</body></html>'
        % (width + 60, legend, svg))

    return html


def cmd_finalize(args):

    run_dir = args.run_dir or find_latest(args.out_dir, "step3_*")

    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("prepare 결과 폴더를 찾을 수 없습니다. --run-dir 로 지정하세요.")

    def read(name):
        return pd.read_csv(os.path.join(run_dir, name), dtype=str, keep_default_na=False, encoding="utf-8-sig")

    with open(os.path.join(run_dir, "run_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cand = read("ot_disease_candidates.csv")
    pairs = read("ot_target_disease_pairs.csv")
    drugs = read("txgnn_drugs.csv")

    for col in ("n_targets", "max_score", "mean_score", "n_matches"):
        if col in cand.columns:
            cand[col] = pd.to_numeric(cand[col], errors="coerce")

    cand["primekg_x_id"] = cand["primekg_x_id"].replace("", pd.NA)
    pairs["overall_score"] = pd.to_numeric(pairs["overall_score"], errors="coerce")

    scores = None

    if args.txgnn_scores:
        scores = pd.read_csv(args.txgnn_scores, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for col in ("drug_id", "primekg_x_id", "score_raw"):
            if col not in scores.columns:
                raise RuntimeError("TxGNN 점수 파일에 %s 컬럼이 없습니다. 컬럼: %s" % (col, list(scores.columns)))
        print("TxGNN 점수: %d행 (%s)" % (len(scores), args.txgnn_scores))
    else:
        print("[참고] --txgnn-scores 가 없어 Open Targets 근거만으로 정리합니다.")

    merged = merge_txgnn_scores(cand, scores, drugs)
    final_df = rank_final(merged)

    final_df.to_csv(os.path.join(run_dir, "disease_candidates_final.csv"), index=False, encoding="utf-8-sig")

    n_scored = int(final_df["txgnn_rank"].notna().sum())

    print("\n최종 후보 %d개 (TxGNN 점수 있음 %d개)" % (len(final_df), n_scored))
    if n_scored and (final_df["txgnn_score_source"] == "proxy").any():
        print("  [주의] txgnn_score_source=proxy 인 행은 입력 화합물이 아니라 대리 약물의 점수입니다.")

    show_cols = ["rank", "disease_id", "disease_name", "n_targets", "max_score"]

    if final_df["txgnn_rank"].notna().any():
        show_cols += ["txgnn_rank", "txgnn_top_pct", "txgnn_known_in_kg"]

    show_cols += ["txgnn_score_source", "map_method"]

    show = final_df[show_cols].head(args.show)
    print(show.to_string(index=False, na_rep="-", float_format=lambda v: "%.3f" % v))

    if not args.no_chart:
        try:
            html = build_candidate_chart_html(final_df, top_n=args.chart_top_n)
            if html:
                chart_path = os.path.join(run_dir, "disease_candidates_chart.html")
                with open(chart_path, "w", encoding="utf-8") as f:
                    f.write(html)
                print("\n그래프 저장: %s (더블클릭하면 브라우저로 열립니다)" % chart_path)
            else:
                print("\n[참고] TxGNN 점수가 있는 질병이 없어 그래프는 생략합니다.")
        except Exception as e:
            print("\n[참고] 그래프 생성 중 문제가 있어 건너뜁니다: %s" % e)

    if args.write_db:

        n_ev, n_ai = write_results_to_db(
            args.ot_db, cfg["analysis_id"], cfg.get("smiles"),
            args.compound_name or cfg.get("compound_name"), final_df, pairs,
            contra_as_side_effect=args.contra_as_side_effect)

        print("\nDB 저장 완료 (%s): analysis_id=%s | EVIDENCE_DB %d행, TXGNN_AI %d행" % (args.ot_db, cfg["analysis_id"], n_ev, n_ai))
    else:
        print("\n(DB 에는 저장하지 않았습니다. 저장하려면 --write-db)")

    return final_df


# ============================================================
# 8. CLI
# ============================================================

def parse_args(argv=None):

    p = argparse.ArgumentParser(description="최종 단백질 -> Open Targets 질병 후보 -> PrimeKG 매핑 -> TxGNN 연결")

    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    common.add_argument("--ot-db", type=str, default=OT_DB_PATH, help="cdss_integrated.db 경로")

    pp = sub.add_parser("prepare", parents=[common], help="후보 질병 구성 + PrimeKG 매핑 + TxGNN 입력 생성")
    pp.add_argument("--final-csv", type=str, default=None, help="Step2 결과 final5_proteins.csv (기본: 가장 최근 것)")
    pp.add_argument("--smiles", type=str, default=None, help="입력 약물 SMILES (기본: Step2 run_config.json 의 값)")
    pp.add_argument("--drug-id", type=str, default=None, help="입력 약물의 DrugBank ID (알고 있으면 가장 확실)")
    pp.add_argument("--drug-name", type=str, default=None)
    pp.add_argument("--compound-name", type=str, default=None, help="DB 의 analysis_history.compound_name 용")
    pp.add_argument("--primekg", type=str, default=PRIMEKG_CSV)
    pp.add_argument("--ot-disease-dir", type=str, default=OT_DISEASE_PARQUET_DIR,
                    help="Open Targets disease parquet 폴더 (dbXRefs 로 매핑률 향상, 없어도 동작)")
    pp.add_argument("--min-score", type=float, default=MIN_EVIDENCE_SCORE)
    pp.add_argument("--per-target-top-k", type=int, default=PER_TARGET_TOP_K)
    pp.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    pp.add_argument("--allow-ancestor", action="store_true",
                    help="매핑 안 되는 질병을 매핑되는 상위 질병으로 대체 (is_broader=True 로 표시, 덜 구체적)")
    pp.add_argument("--proxy-from-targets", action="store_true",
                    help="Top 타깃을 표적으로 하는 PrimeKG 약물을 대리(proxy)로도 점수 계산")
    pp.add_argument("--proxy-if-missing", action="store_true",
                    help="입력 약물이 PrimeKG 에 없을 때만 proxy 약물 사용")
    pp.add_argument("--max-proxy-drugs", type=int, default=20)
    pp.add_argument("--uniprot-cache-dir", type=str, default=UNIPROT_CACHE_DIR)
    pp.add_argument("--pubchem-cache-dir", type=str, default=PUBCHEM_CACHE_DIR)
    pp.add_argument("--auto-chain", action="store_true",
                    help="원터치 실행: final5_proteins.csv 가 없으면 Step 1+2 부터 자동 실행하고(--smiles 필요), "
                         "prepare 가 끝나면 TxGNN 환경으로 자동 전환해 점수를 계산한 뒤 finalize 까지 이어서 실행")
    pp.add_argument("--write-db", action="store_true",
                    help="--auto-chain 으로 finalize 까지 자동 실행할 때, cdss_integrated.db 에도 결과를 저장")

    fp = sub.add_parser("finalize", parents=[common], help="TxGNN 점수를 합쳐 최종 표 생성 (+ DB 저장)")
    fp.add_argument("--run-dir", type=str, default=None, help="prepare 결과 폴더 (기본: 가장 최근 step3_*)")
    fp.add_argument("--txgnn-scores", type=str, default=None, help="step3_txgnn_score.py 가 만든 txgnn_scores.csv")
    fp.add_argument("--write-db", action="store_true", help="cdss_integrated.db 의 step3_indication_results 에 저장")
    fp.add_argument("--compound-name", type=str, default=None)
    fp.add_argument("--contra-as-side-effect", action="store_true",
                    help="TxGNN contraindication 점수를 is_side_effect=1 행으로도 저장 (금기 != 부작용이므로 기본 꺼짐)")
    fp.add_argument("--show", type=int, default=20)
    fp.add_argument("--no-chart", action="store_true", help="상위 후보 막대그래프 HTML 생성을 생략")
    fp.add_argument("--chart-top-n", type=int, default=20, help="그래프에 표시할 상위 후보 개수")

    return p.parse_args(argv)


def main(argv=None):

    # 인자 없이 실행하면(예: PyCharm 실행 버튼) 기본으로 prepare 를 실행한다.
    if argv is None and len(sys.argv) == 1:
        print("[안내] 실행할 작업이 지정되지 않아 'prepare' 로 실행합니다. (finalize 는 'finalize' 를 붙여 실행)")
        argv = ["prepare"]

    args = parse_args(argv)

    if args.command == "prepare":
        cmd_prepare(args)
    else:
        cmd_finalize(args)


if __name__ == "__main__":
    main()