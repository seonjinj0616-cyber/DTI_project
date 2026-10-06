"""
Step 3b: TxGNN (약물, 질병) 점수 계산  -  TxGNN 이 설치된 환경에서 실행

입력  : step3_ot_txgnn_link.py prepare 가 만든 폴더 (txgnn_drugs.csv, txgnn_diseases.csv)
출력  : <run-dir>/txgnn_scores.csv  (drug_id, drug_name, role, primekg_x_id, disease_name, relation, score)
        -> step3_ot_txgnn_link.py finalize --txgnn-scores 로 연결

확인된 구조 (--inspect 실행 결과, Windows CPU / TxGNN 0.0.3)
    tx_data.df       : 컬럼 [x_type, x_id, relation, y_type, y_id, x_idx, y_idx]  (약물/질병의 id -> 내부 idx)
                       숫자 id 는 4992.0 처럼 float, 묶음 질병은 '5105_5012' 처럼 문자열
    eval_disease_centric(return_raw=True) 의 반환:
        {"prediction": {질병id문자열: {DrugBank ID: 점수}},     # 질병마다 7,957개 약물 전체의 점수
         "label":      {질병id문자열: {DrugBank ID: 0/1}},       # 이미 알려진(KG 에 있는) 관계 여부
         "result":     {"ID", "Name", "Ranked List", "AUROC", ...}}
    -> 약물 index 가 필요 없고, DrugBank ID 로 바로 점수를 읽는다.
    -> 점수와 함께 "전체 약물 중 순위"와 "이미 알려진 관계인지"도 함께 저장한다.

    아직 확인하지 못한 것: 점수 값의 의미(0~1 확률로 가정, 범위를 벗어나면 경고)

환경
    README 는 python 3.8 + DGL 0.5.2 를 안내한다. 제3자 프로젝트(UsTxGNN)는 python 3.11 / torch 2.2.2 /
    dgl 1.1.3 / pip install git+https://github.com/mims-harvard/TxGNN.git 조합으로 실행한다고 밝히고 있다.
    (Windows CPU 에서 설치 가능한지는 확인하지 못함)

필요한 자료
    - TxGNN 데이터 폴더(kg.csv, node.csv, edges.csv). TxData 가 없으면 내려받는 것으로 보이며,
      직접 받으려면 Harvard Dataverse 파일 ID 7144482(node.csv) / 7144483(edges.csv) / 7144484(kg.csv)
    - 사전학습 가중치 폴더(model.pt 등): README 의 Google Drive 예제 가중치 (직접 학습은 GPU 필요)

사용 예
    python step3_txgnn_score.py --run-dir results\\step3_XXXX --data-dir D:\\txgnn_data --ckpt-dir D:\\txgnn_ckpt --inspect
    python step3_txgnn_score.py --run-dir results\\step3_XXXX --data-dir D:\\txgnn_data --ckpt-dir D:\\txgnn_ckpt
"""

import os
import re
import sys
import time
import glob
import pickle
import argparse


def _find_project_root(marker_name="DTI_project"):
    """
    이 스크립트 파일 위치에서 위로 올라가며 이름이 marker_name 인 폴더를 찾는다.
    DTI_project 폴더 전체를 다른 컴퓨터/다른 드라이브/다른 사용자 계정으로 옮겨도,
    폴더 이름만 같으면 경로를 자동으로 다시 찾는다.
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
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "results")
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "data", "txgnn_data")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_DIR, "data", "txgnn_ckpt")


def find_latest_run_dir(out_dir):
    """out_dir 안의 step3_* 폴더 중 가장 최근에 수정된 것을 찾는다 (없으면 None)."""

    found = sorted(glob.glob(os.path.join(out_dir, "step3_*")), key=os.path.getmtime)

    return found[-1] if found else None


# ============================================================
# 1. 공통 도우미 (python 3.8 호환)
# ============================================================

def describe(obj, depth=0, max_depth=3, max_items=4, out=print):
    """자료구조를 사람이 읽을 수 있게 요약 출력 (dict / list / DataFrame / ndarray / tensor)."""

    pad = "  " * depth
    name = type(obj).__name__

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        out("%s%s shape=%s dtype=%s" % (pad, name, tuple(obj.shape), obj.dtype))
    elif name == "DataFrame":
        out("%sDataFrame %s 컬럼=%s" % (pad, obj.shape, list(obj.columns)))
        out(obj.head(3).to_string())
    elif isinstance(obj, dict):
        out("%sdict (%d keys) 예시 key=%s" % (pad, len(obj), list(obj.keys())[:max_items]))
        if depth < max_depth:
            for k in list(obj.keys())[:max_items]:
                out("%s  [%r] ->" % (pad, k))
                describe(obj[k], depth + 2, max_depth, max_items, out)
    elif isinstance(obj, (list, tuple)):
        out("%s%s (len=%d)" % (pad, name, len(obj)))
        if depth < max_depth:
            for i, v in enumerate(obj[:max_items]):
                out("%s  [%d] ->" % (pad, i))
                describe(v, depth + 2, max_depth, max_items, out)
    else:
        text = repr(obj)
        out("%s%s: %s" % (pad, name, text if len(text) < 120 else text[:117] + "..."))


def digits_key(text):
    """MONDO 계열 id 를 비교 가능한 숫자 집합으로. '4992.0' 의 뒤쪽 '.0' 은 무시한다."""
    cleaned = re.sub(r"\.0+(?=$|_)", "", str(text))
    return {str(int(t)) for t in re.findall(r"\d+", cleaned)}


def norm_name(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def pick_col(columns, *keywords, **kw):
    """컬럼명에 keywords 가 모두 들어 있는 첫 컬럼 (exclude=... 로 제외어 지정)."""
    exclude = kw.get("exclude", ())
    for c in columns:
        low = str(c).lower()
        if all(k in low for k in keywords) and not any(e in low for e in exclude):
            return c
    return None


# ============================================================
# 2. TxGNN 내부 index 찾기 (추정)
# ============================================================

def _empty_lookup():
    return {"drug": {"by_id": {}, "by_name": {}, "by_digits": {}, "by_idx": {}},
            "disease": {"by_id": {}, "by_name": {}, "by_digits": {}, "by_idx": {}}}


def lookup_from_df(df):
    """
    tx_data.df (약물/질병 관계 표) 에서 약물/질병의 id -> idx (그리고 idx -> id) 사전을 만든다.
    x_* / y_* 양쪽에서 *_idx, *_id, *_type 컬럼을 찾는다. (이 표에는 이름 컬럼이 없다)
    """

    cols = list(df.columns)

    sides = []

    for side in ("x", "y"):
        idx_c = pick_col(cols, side + "_idx") or pick_col(cols, side + "_index")
        id_c = pick_col(cols, side + "_id", exclude=("idx",))
        type_c = pick_col(cols, side + "_type")
        if idx_c and id_c and type_c:
            sides.append((idx_c, id_c, type_c))

    if not sides:
        return None

    result = _empty_lookup()

    for idx_c, id_c, type_c in sides:
        for kind in ("drug", "disease"):
            sub = df.loc[df[type_c].astype(str).str.lower() == kind, [idx_c, id_c]].drop_duplicates()
            table = result[kind]
            for idx, ident in zip(sub[idx_c], sub[id_c]):
                key = str(ident)
                table["by_id"].setdefault(key, idx)
                table["by_idx"].setdefault(float(idx), key)
                for d in digits_key(ident):
                    table["by_digits"].setdefault(d, idx)

    return result


def lookup_from_node_csv(nodes):
    """
    node.csv 에서 종류별 "행 순서"를 index 로 가정해 사전을 만든다 (가정이므로 lookup_from_df 가 우선).
    컬럼 후보: node_id, node_type, node_name
    """

    id_c = pick_col(nodes.columns, "id", exclude=("idx", "index"))
    type_c = pick_col(nodes.columns, "type")
    name_c = pick_col(nodes.columns, "name")

    if not id_c or not type_c:
        return None

    result = _empty_lookup()
    counters = {"drug": 0, "disease": 0}

    for _, row in nodes.iterrows():
        kind = str(row[type_c]).lower()
        if kind not in result:
            continue
        idx = float(counters[kind])
        counters[kind] += 1
        result[kind]["by_id"][str(row[id_c])] = idx
        result[kind]["by_idx"][idx] = str(row[id_c])
        for d in digits_key(row[id_c]):
            result[kind]["by_digits"].setdefault(d, idx)
        if name_c:
            result[kind]["by_name"].setdefault(norm_name(row[name_c]), idx)

    return result


def resolve_index(lookup, kind, identifier, name=None):
    """id(정확 일치) -> 숫자 부분(질병 MONDO) -> 이름 순으로 index 를 찾는다. 반환: (index, 방법) 또는 (None, None)"""

    table = lookup[kind]

    if str(identifier) in table["by_id"]:
        return table["by_id"][str(identifier)], "id"

    if kind == "disease":
        for d in digits_key(identifier):
            if d in table["by_digits"]:
                return table["by_digits"][d], "digits"

    if name and norm_name(name) in table["by_name"]:
        return table["by_name"][norm_name(name)], "name"

    return None, None


# ============================================================
# 3. eval_disease_centric 원시 출력에서 점수 꺼내기 (추정)
# ============================================================

SCORE_WORDS = ("score", "pred", "prob", "sigmoid", "logit")


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_score(raw, disease_idx, drug_idx):
    """
    raw 안에서 (질병, 약물) 점수를 찾는다. 반환: (점수, 방법) 또는 (None, None).
    지원하는 모양:
      A) DataFrame: 약물/질병 index 컬럼 + 점수 컬럼
      B) {질병 idx: {약물 idx: 점수}}
      C) {질병 idx: DataFrame}
      D) {질병 idx: {"약물 idx 배열", "점수 배열"}}
      E) {질병 idx: 1차원 배열}  -> 배열 위치 = 약물 idx 라고 가정(positional)
    """

    def keyed(d, key):
        for k in (key, float(key), int(key), str(key), str(float(key)), str(int(key))):
            try:
                if k in d:
                    return d[k]
            except TypeError:
                continue
        return None

    def from_frame(df, need_disease):
        cols = list(df.columns)
        d_idx = (pick_col(cols, "drug", "idx") or pick_col(cols, "x_idx")
                 or pick_col(cols, "drug", exclude=("name",)))
        s_col = next((c for c in cols for w in SCORE_WORDS if w in str(c).lower()), None)
        dis_col = (pick_col(cols, "disease", "idx") or pick_col(cols, "y_idx")) if need_disease else None
        if d_idx is None or s_col is None:
            return None
        sub = df
        if dis_col is not None:
            sub = sub[sub[dis_col].astype(float) == float(disease_idx)]
        sub = sub[sub[d_idx].astype(float) == float(drug_idx)]
        if len(sub):
            return _as_float(sub.iloc[0][s_col])
        return None

    name = type(raw).__name__

    if name == "DataFrame":
        v = from_frame(raw, need_disease=True)
        return (v, "frame") if v is not None else (None, None)

    if isinstance(raw, dict):

        inner = keyed(raw, disease_idx)

        if inner is None:
            return None, None

        if type(inner).__name__ == "DataFrame":
            v = from_frame(inner, need_disease=False)
            return (v, "dict->frame") if v is not None else (None, None)

        if isinstance(inner, dict):
            direct = keyed(inner, drug_idx)
            if _as_float(direct) is not None and not isinstance(direct, (dict, list)):
                return _as_float(direct), "dict->dict"
            idx_key = next((k for k in inner if "drug" in str(k).lower() or "idx" in str(k).lower()), None)
            sc_key = next((k for k in inner if any(w in str(k).lower() for w in SCORE_WORDS)), None)
            if idx_key is not None and sc_key is not None:
                idxs = list(inner[idx_key])
                scs = list(inner[sc_key])
                if len(idxs) == len(scs):
                    for i, x in zip(idxs, scs):
                        if _as_float(i) == float(drug_idx):
                            return _as_float(x), "dict->arrays"
            return None, None

        if hasattr(inner, "shape") and len(getattr(inner, "shape", ())) == 1:
            i = int(drug_idx)
            if 0 <= i < inner.shape[0]:
                return _as_float(inner[i]), "positional(가정)"

    return None, None


def _key_variants(key):
    """질병 key 표기 차이 ('4992.0' <-> '4992')를 모두 시도."""
    key = str(key)
    variants = [key]
    if re.fullmatch(r"\d+\.0+", key):
        variants.append(key.split(".")[0])
    elif re.fullmatch(r"\d+", key):
        variants.append(key + ".0")
    return variants


def _get_key(d, key):
    if not isinstance(d, dict):
        return None
    for k in _key_variants(key):
        if k in d:
            return d[k]
    return None


def read_prediction(raw, disease_key, drug_id):
    """
    확인된 구조 {"prediction": {질병id: {DrugBank ID: 점수}}, "label": {...}} 에서 한 쌍의 점수를 읽는다.

    방향 (양성 대조군 점검으로 확인됨, 2026-09):
      TxGNN 원점수는 "낮을수록(더 음수일수록) 그 약물-질병 관계가 유력하다"는 방향이다.
      (반대로 "높을수록 유력"으로 가정했을 때는 실제 known-indication 쌍들이 하위 82% 근처에 몰렸고,
       방향을 뒤집으니 상위 18% 근처로 모였다 -> 이 방향이 맞다고 판단)

    반환: dict(score, rank, rank_desc, n_drugs, known_indication) 또는 None
      score     : 원점수(raw) 그대로. 시그모이드로 변환하지 않는다 (질병 간 비교용이 아니라 참고용).
      rank      : 그 질병에 대해 전체 약물 중 순위 (1 = 가장 유력, 즉 가장 낮은/음수인 점수) -- 질병 간 비교는 이걸로.
      rank_desc : (참고/회귀 검증용) "높을수록 유력하다"고 가정했을 때의 순위 -- 예전에 잘못 쓰던 방향
      known_ind : label 이 1 인가 (= 이미 KG 에 알려진 관계라 '새로운 예측'이 아님)
    """

    if not isinstance(raw, dict) or "prediction" not in raw:
        return None

    per_disease = _get_key(raw["prediction"], disease_key)

    if not isinstance(per_disease, dict) or drug_id not in per_disease:
        return None

    score = _as_float(per_disease[drug_id])

    if score is None:
        return None

    values = [v for v in (_as_float(x) for x in per_disease.values()) if v is not None]

    rank = 1 + sum(1 for v in values if v < score)          # 확인된 올바른 방향: 낮을수록(더 음수일수록) 유력
    rank_desc = 1 + sum(1 for v in values if v > score)     # 예전에 쓰던(잘못된) 방향 -- 참고/회귀 검증용으로만 보관

    # 시그모이드 변환은 하지 않는다: 값을 0~1로 눌러 담으면(특히 원점수가 -5 이하일 때) 서로 다른
    # 질병인데도 다 1.000 근처로 뭉개져 구분이 안 된다. score 는 항상 원점수(raw) 그대로다.
    # 질병 간 비교에는 이 원점수가 아니라 rank 를 쓸 것 (rank 는 질병마다 같은 기준: 전체 약물 수).

    labels = _get_key(raw.get("label"), disease_key)
    known = None

    if isinstance(labels, dict) and drug_id in labels:
        known = bool(_as_float(labels[drug_id]))

    return {"score": score, "rank": rank, "rank_desc": rank_desc,
            "n_drugs": len(values), "known_indication": known}


# ============================================================
# 4. 실행
# ============================================================

def parse_args(argv=None):

    p = argparse.ArgumentParser(description="TxGNN (약물, 질병) 점수 계산 / 구조 확인")

    p.add_argument("--run-dir", type=str, default=None,
                   help="step3_ot_txgnn_link.py prepare 결과 폴더 (생략하면 --out-dir 안의 가장 최근 step3_* 폴더를 자동 사용)")
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                   help="--run-dir 을 생략했을 때 가장 최근 step3_* 폴더를 찾을 위치 (기본: DTI_project/results)")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="TxGNN 데이터 폴더 (kg.csv, node.csv, edges.csv)")
    p.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR, help="사전학습 가중치 폴더")
    p.add_argument("--device", type=str, default="cpu", help="cpu 또는 cuda:0")
    p.add_argument("--n-hid", type=int, default=100, help="가중치를 만든 model_initialize 차원과 같아야 함")
    p.add_argument("--relations", nargs="+", default=["indication"],
                   help="점수를 낼 관계 (indication / contraindication / off-label use)")
    p.add_argument("--inspect", action="store_true", help="점수 계산 대신 구조만 출력하고 원시 출력을 pickle 로 저장")
    p.add_argument("--batch", type=int, default=20, help="eval_disease_centric 에 한 번에 넘길 질병 수")
    p.add_argument("--out", type=str, default=None, help="출력 CSV (기본: <run-dir>/txgnn_scores.csv, --sanity-check 는 ./txgnn_sanity_check.csv)")

    p.add_argument("--sanity-check", type=int, default=None, metavar="N",
                   help="prepare 결과 없이: tx_data.df 안의 '진짜(TxGNN 이 학습한) 약물-질병 쌍' 중 N개를 무작위로 뽑아 "
                        "순위가 보통 몇 등으로 나오는지 점검한다. 다른 약물/질병 점수 계산과 비교할 기준선을 만드는 용도.")
    p.add_argument("--sanity-relation", type=str, default="indication", help="--sanity-check 에서 표본을 뽑을 관계")
    p.add_argument("--seed", type=int, default=42, help="--sanity-check 표본 추출 시드 (재현성)")

    return p.parse_args(argv)


def load_txgnn(args, log=print):
    """README 에 문서화된 호출로 사전학습 TxGNN 을 불러온다. 반환: (tx_data, model, evaluator)."""

    try:
        from txgnn import TxData, TxGNN, TxEval
    except ImportError as e:
        raise SystemExit(
            "txgnn 패키지를 불러올 수 없습니다: %s\n"
            "  conda create -n txgnn python=3.11  (또는 README 의 3.8)\n"
            "  pip install torch  /  pip install dgl  (버전은 README 또는 UsTxGNN 안내 참고)\n"
            "  pip install git+https://github.com/mims-harvard/TxGNN.git   (또는 pip install TxGNN)" % e
        )

    tx_data = TxData(data_folder_path=args.data_dir)
    tx_data.prepare_split(split="full_graph", seed=42)

    model = TxGNN(data=tx_data, weight_bias_track=False, proj_name="TxGNN", exp_name="TxGNN", device=args.device)

    model.model_initialize(n_hid=args.n_hid, n_inp=args.n_hid, n_out=args.n_hid,
                           proto=True, proto_num=3, attention=False,
                           sim_measure="all_nodes_profile", bert_measure="disease_name",
                           agg_measure="rarity", num_walks=200, walk_mode="bit", path_length=2)

    model.load_pretrained(args.ckpt_dir)

    log("사전학습 가중치 로드 완료: %s" % args.ckpt_dir)

    return tx_data, model, TxEval(model=model)


def call_eval(evaluator, disease_idxs, relation):
    try:
        return evaluator.eval_disease_centric(disease_idxs=disease_idxs, relation=relation,
                                              save_result=False, return_raw=True)
    except TypeError:
        return evaluator.eval_disease_centric(disease_idxs=disease_idxs, relation=relation, save_result=False)


def call_eval_safe(evaluator, idxs, relation, log=print):
    """
    배치 호출이 실패하면 질병 하나씩 다시 시도한다 (알려진 약물이 없는 질병 등에서 평가 지표 계산이
    실패해도 나머지는 계속). 반환: (raw 목록, 실패한 질병 idx 목록)
    """

    try:
        return [call_eval(evaluator, idxs, relation)], []
    except Exception as e:
        if len(idxs) == 1:
            log("    [경고] 질병 idx %s 실패: %s: %s" % (idxs[0], type(e).__name__, e))
            return [], list(idxs)
        log("    [참고] 배치 호출 실패(%s: %s) -> 질병별로 다시 시도합니다." % (type(e).__name__, e))

    raws, failed = [], []

    for i in idxs:
        r, f = call_eval_safe(evaluator, [i], relation, log)
        raws.extend(r)
        failed.extend(f)

    return raws, failed


def build_lookup(tx_data, data_dir, log=print):

    import pandas as pd

    lookup, source = None, None

    df = getattr(tx_data, "df", None)

    if df is not None:
        lookup = lookup_from_df(df)
        if lookup and (lookup["drug"]["by_id"] or lookup["drug"]["by_name"]):
            source = "tx_data.df"
        else:
            lookup = None

    if lookup is None:
        node_path = os.path.join(data_dir, "node.csv")
        if os.path.exists(node_path):
            nodes = pd.read_csv(node_path, dtype=str, keep_default_na=False, sep=None, engine="python")
            lookup = lookup_from_node_csv(nodes)
            source = "node.csv 행 순서(가정)"

    return lookup, source


def inspect_mode(args, tx_data, evaluator, drugs, diseases, lookup, source, log=print):

    import pandas as pd

    log("\n" + "=" * 70)
    log("구조 확인 (--inspect)")
    log("=" * 70)

    df = getattr(tx_data, "df", None)

    if df is not None:
        log("\n[tx_data.df] %s 컬럼: %s" % (tuple(df.shape), list(df.columns)))
        log(df.head(3).to_string())
    else:
        log("\n[tx_data.df] 없음. tx_data 속성: %s" % [a for a in dir(tx_data) if not a.startswith("_")][:40])

    node_path = os.path.join(args.data_dir, "node.csv")
    if os.path.exists(node_path):
        nodes = pd.read_csv(node_path, dtype=str, keep_default_na=False, nrows=5, sep=None, engine="python")
        log("\n[node.csv] 컬럼: %s" % list(nodes.columns))
        log(nodes.head(3).to_string())

    log("\nindex 조회 방식: %s" % source)

    probe = []

    if lookup is not None:
        for _, r in diseases.head(3).iterrows():
            idx, how = resolve_index(lookup, "disease", r["primekg_x_id"], r.get("primekg_name"))
            log("  질병 %s (%s) -> idx %s (%s)" % (r["primekg_x_id"], r.get("primekg_name"), idx, how))
            if idx is not None:
                probe.append(idx)

    if not probe:
        probe = [9907.0, 12787.0]
        log("  조회에 실패해 README 예시 질병 index %s 로 호출합니다." % probe)

    raw = call_eval(evaluator, probe, args.relations[0])

    log("\n[eval_disease_centric 반환 구조]")
    describe(raw, out=log)

    if isinstance(raw, dict) and isinstance(raw.get("prediction"), dict):

        first_key = next(iter(raw["prediction"]))
        per = raw["prediction"][first_key]
        vals = sorted(((_as_float(v), k) for k, v in per.items() if _as_float(v) is not None), reverse=True)

        log("\n[점수 값 확인] 질병 key=%r, 약물 %d개" % (first_key, len(vals)))
        log("  점수 범위: %.4f ~ %.4f" % (vals[-1][0], vals[0][0]))
        log("  상위 5개 약물: " + ", ".join("%s(%.3f)" % (k, v) for v, k in vals[:5]))

        labels = _get_key(raw.get("label"), first_key)
        if isinstance(labels, dict):
            log("  이 질병의 알려진 약물(label=1) 수: %d" % sum(1 for v in labels.values() if _as_float(v)))

        for _, dr in drugs.head(2).iterrows():
            info = read_prediction(raw, first_key, dr["drug_id"])
            log("  입력 약물 %s -> %s" % (dr["drug_id"], info if info else "이 질병의 예측에 없음"))

        result = raw.get("result")
        if isinstance(result, dict):
            for k in ("AUROC", "AUPRC"):
                if k in result:
                    log("  result[%s] = %s" % (k, result[k]))

    dump = os.path.join(args.run_dir, "txgnn_raw_output.pkl")
    with open(dump, "wb") as f:
        pickle.dump(raw, f)

    log("\n원시 출력 저장: %s" % dump)
    log("위 출력(특히 tx_data.df 컬럼, node.csv 컬럼, 반환 구조)을 보내주시면 추정 부분을 정확히 맞출 수 있습니다.")


def score_mode(args, evaluator, drugs, diseases, lookup, source, log=print):

    import pandas as pd

    if lookup is None:
        raise RuntimeError("TxGNN index 를 찾지 못했습니다 (tx_data.df / node.csv 모두 실패). --inspect 로 구조를 확인하세요.")

    log("index 조회 방식: %s" % source)

    # ---- 질병 index (TxGNN 내부 idx) 와 결과 key (질병 id 문자열) ----
    dis_idx, dis_key, unresolved = {}, {}, []

    for _, r in diseases.iterrows():
        idx, how = resolve_index(lookup, "disease", r["primekg_x_id"], r.get("primekg_name"))
        if idx is None:
            unresolved.append(r["primekg_x_id"])
        else:
            dis_idx[r["primekg_x_id"]] = idx
            dis_key[r["primekg_x_id"]] = lookup["disease"]["by_idx"].get(float(idx), str(r["primekg_x_id"]))

    log("질병 %d/%d개의 TxGNN index 확인 (%d개는 TxGNN 그래프에서 못 찾음)"
        % (len(dis_idx), len(diseases), len(unresolved)))

    if not dis_idx:
        raise RuntimeError("점수를 계산할 질병이 없습니다. --inspect 로 index 조회 방식을 확인하세요.")

    ids = list(dis_idx)
    name_of = dict(zip(diseases["primekg_x_id"], diseases["primekg_name"]))

    rows, failed_ids, missing_drug, no_key = [], [], {}, 0
    last_raws = []

    for relation in args.relations:

        for start in range(0, len(ids), args.batch):

            chunk = ids[start:start + args.batch]
            idxs = [dis_idx[i] for i in chunk]

            t0 = time.time()

            raws, failed = call_eval_safe(evaluator, idxs, relation, log)

            last_raws = raws or last_raws

            failed_set = set(failed)
            failed_ids.extend([pid for pid in chunk if dis_idx[pid] in failed_set])

            for pid in chunk:

                if dis_idx[pid] in failed_set:
                    continue

                found_any = False

                for raw in raws:

                    for _, dr in drugs.iterrows():

                        info = read_prediction(raw, dis_key[pid], dr["drug_id"])

                        if info is None:
                            if _get_key(raw.get("prediction") if isinstance(raw, dict) else None, dis_key[pid]) is not None:
                                found_any = True
                                missing_drug[dr["drug_id"]] = missing_drug.get(dr["drug_id"], 0) + 1
                            continue

                        found_any = True

                        rows.append({"drug_id": dr["drug_id"], "drug_name": dr.get("drug_name"), "role": dr.get("role"),
                                     "primekg_x_id": pid, "disease_name": name_of.get(pid), "relation": relation,
                                     "score_raw": info["score"], "rank": info["rank"], "n_drugs": info["n_drugs"],
                                     "rank_percentile": info["rank"] / max(info["n_drugs"], 1),
                                     "known_indication": info["known_indication"]})

                if not found_any:
                    no_key += 1

            log("  [%s] %d/%d 질병 (%.1f초)" % (relation, min(start + args.batch, len(ids)), len(ids), time.time() - t0))

    if not rows:
        raise RuntimeError(
            "eval_disease_centric 의 반환값에서 점수를 하나도 읽지 못했습니다.\n"
            "  기대한 구조: {\"prediction\": {질병id: {DrugBank ID: 점수}}, ...}. 실제 반환 구조:\n%s\n"
            "  --inspect 출력과 함께 알려주시면 읽는 부분을 맞추겠습니다."
            % (_describe_text(last_raws[0]) if last_raws else "(반환값 없음: 모든 호출이 실패)"))

    scores = pd.DataFrame(rows)

    log("점수 %d개 저장 대상 (약물 %d개 x 질병 %d개 x 관계 %d개)"
        % (len(scores), scores["drug_id"].nunique(), scores["primekg_x_id"].nunique(), scores["relation"].nunique()))

    if failed_ids:
        log("  [경고] 계산에 실패해 건너뛴 질병 %d개: %s" % (len(failed_ids), failed_ids[:5]))
    if no_key:
        log("  [경고] 결과에서 질병 key 를 찾지 못한 질병 %d개 (key 표기 불일치 가능성)" % no_key)
    for d, c in missing_drug.items():
        log("  [경고] 약물 %s 는 TxGNN 예측 목록에 없음 (%d개 질병에서)" % (d, c))

    log("  [참고] score_raw 는 시그모이드로 변환하지 않은 원점수입니다 (범위 %.3f ~ %.3f). "
        "질병 간 비교에는 쓰지 마세요 -- 그 용도로는 rank 를 쓰세요 (아래 점검 참고)."
        % (scores["score_raw"].min(), scores["score_raw"].max()))

    # ---- 점검: 이미 알려진(label=1) 쌍이 그렇지 않은 쌍보다 순위가 앞서는가 ----
    # 방향(낮을수록/더 음수일수록 유력)은 양성 대조군 AUROC 검증(check_auroc_direction.py)으로 확인됨.
    ind = scores[scores["relation"] == "indication"]
    known = ind[ind["known_indication"] == True]
    unknown = ind[ind["known_indication"] == False]

    if len(known) and len(unknown):
        log("  [점검] 이미 알려진 적응증 %d쌍: 평균 순위 %.0f (원점수 평균 %.3f, 중앙값 %.3f) / 그 외 %d쌍: 평균 순위 %.0f (원점수 평균 %.3f, 중앙값 %.3f)"
            % (len(known), known["rank"].mean(), known["score_raw"].mean(), known["score_raw"].median(),
               len(unknown), unknown["rank"].mean(), unknown["score_raw"].mean(), unknown["score_raw"].median()))
        better = known["rank"].mean() < unknown["rank"].mean()
        log("         -> %s (순위 기준, %s 표본이라 원점수 평균은 참고만 하세요)"
            % ("알려진 쌍의 순위가 더 앞섭니다: 정상 동작으로 보입니다" if better
               else "알려진 쌍의 순위가 더 앞서지 않았습니다: 결과를 신중히 해석하세요",
               "known 쪽이 %d개로 적은" % len(known) if len(known) < 30 else "표본 수가 충분한"))

    return scores


def _describe_text(obj):
    lines = []
    describe(obj, out=lines.append)
    return "\n".join("    " + l for l in lines)


def load_node_names(data_dir):
    """node.csv 에서 id -> 표시 이름 사전을 만든다 (있으면 사람이 읽기 편하게, 없으면 조용히 {} 반환)."""

    import pandas as pd

    path = os.path.join(data_dir, "node.csv")

    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python")
    except Exception:
        return {}

    id_c = pick_col(df.columns, "id", exclude=("idx", "index"))
    name_c = pick_col(df.columns, "name")

    if not id_c or not name_c:
        return {}

    return dict(zip(df[id_c], df[name_c]))


def sample_known_pairs(df, relation, n, seed):
    """
    tx_data.df 에서 relation(기본 indication) 인 (약물, 질병) 쌍을 양방향 모두 고려해 뽑는다.
    이 쌍들은 TxGNN 이 "진짜"라고 알고 학습한 것들이라, 순위의 기준선(baseline)을 만드는 데 쓴다.
    """

    import pandas as pd

    rel = df[df["relation"] == relation]

    fwd = rel[rel["x_type"] == "drug"][["x_id", "x_idx", "y_id", "y_idx"]].rename(
        columns={"x_id": "drug_id", "x_idx": "drug_idx", "y_id": "disease_key", "y_idx": "disease_idx"})
    rev = rel[rel["y_type"] == "drug"][["y_id", "y_idx", "x_id", "x_idx"]].rename(
        columns={"y_id": "drug_id", "y_idx": "drug_idx", "x_id": "disease_key", "x_idx": "disease_idx"})

    pairs = pd.concat([fwd, rev], ignore_index=True).drop_duplicates()

    if len(pairs) == 0:
        return pairs

    return pairs.sample(n=min(n, len(pairs)), random_state=seed).reset_index(drop=True)


def sanity_check_mode(args, tx_data, evaluator, log=print):
    """
    양성 대조군 점검: TxGNN 이 이미 "진짜"라고 알고 있는 (약물, 질병) 쌍 N개를 무작위로 뽑아
    순위 분포를 본다. 다른 화합물/질병의 점수가 낮게 나왔을 때, 그게 "이 모델·이 그래프에서는
    원래 이 정도가 정상"인지 "우리 쪽 문제"인지 구분하는 기준선이 된다.
    """

    import pandas as pd

    df = tx_data.df

    pairs = sample_known_pairs(df, args.sanity_relation, args.sanity_check, args.seed)

    if pairs.empty:
        raise RuntimeError("tx_data.df 에서 relation='%s' 인 (약물, 질병) 쌍을 찾지 못했습니다." % args.sanity_relation)

    names = load_node_names(args.data_dir)

    rows = []

    for disease_idx, grp in pairs.groupby("disease_idx"):

        disease_key = grp["disease_key"].iloc[0]

        raws, failed = call_eval_safe(evaluator, [disease_idx], args.sanity_relation, log)

        if not raws:
            log("  [경고] 질병 idx %s 평가 실패, 건너뜁니다." % disease_idx)
            continue

        raw = raws[0]

        for _, r in grp.iterrows():

            info = read_prediction(raw, disease_key, r["drug_id"])

            if info is None:
                continue

            rows.append({"drug_id": r["drug_id"], "drug_name": names.get(r["drug_id"], ""),
                        "disease_key": disease_key, "disease_name": names.get(str(disease_key), ""),
                        "score_raw": info["score"], "rank": info["rank"], "rank_desc": info["rank_desc"],
                        "n_drugs": info["n_drugs"], "rank_percentile": info["rank"] / max(info["n_drugs"], 1),
                        "rank_desc_percentile": info["rank_desc"] / max(info["n_drugs"], 1)})

    if not rows:
        raise RuntimeError("표본에서 점수를 하나도 읽지 못했습니다 (모든 질병 평가가 실패한 것으로 보입니다).")

    result = pd.DataFrame(rows)

    log("\n" + "=" * 70)
    log("TxGNN 양성 대조군 점검  (relation=%s, 표본 %d쌍 중 %d쌍 평가 성공)"
        % (args.sanity_relation, len(pairs), len(result)))
    log("=" * 70)
    log("이 쌍들은 전부 TxGNN 이 '진짜'라고 학습한 약물-질병 관계입니다.")
    log("이 모델·이 그래프에서 '진짜 정답'이 보통 몇 등으로 나오는지 보여주는 기준선입니다.")
    log("")
    log("순위(1등이 가장 좋음): 중앙값 %.0f / 평균 %.0f / 최고 %d등 / 최저 %d등  (전체 약물 %d개 중)"
        % (result["rank"].median(), result["rank"].mean(), result["rank"].min(),
           result["rank"].max(), int(result["n_drugs"].iloc[0])))
    log("순위 백분위(0=1등, 1=꼴찌): 중앙값 %.3f / 평균 %.3f"
        % (result["rank_percentile"].median(), result["rank_percentile"].mean()))
    log("상위 1%% 안에 든 비율: %.0f%%  /  상위 10%% 안에 든 비율: %.0f%%"
        % (100 * (result["rank_percentile"] <= 0.01).mean(), 100 * (result["rank_percentile"] <= 0.10).mean()))

    log("\n예시 (현재 방향: 점수가 낮을수록/더 음수일수록 유력함 -- 양성 대조군 점검으로 확인된 방향):")
    for _, r in result.head(8).iterrows():
        dname = r["drug_name"] or r["drug_id"]
        disname = r["disease_name"] or r["disease_key"]
        log("  %s -> %s : %d/%d 등 (상위 %.1f%%)" % (dname, disname, r["rank"], r["n_drugs"], 100 * r["rank_percentile"]))

    # ---- 방향 재확인: 지금(낮을수록 유력, rank) 과 예전 방식(높을수록 유력, rank_desc) 을 계속 비교해서
    #      혹시 나중에 체크포인트/데이터가 바뀌어 방향이 또 달라지지는 않았는지 회귀 확인한다.
    med_now, med_old = result["rank_percentile"].median(), result["rank_desc_percentile"].median()

    log("\n[방향 점검] 지금 방향(낮을수록 유력) 중앙값 백분위 %.3f  vs  예전 방향(높을수록 유력) 중앙값 백분위 %.3f"
        % (med_now, med_old))

    if med_now < 0.4 and med_old > 0.6:
        log("  -> '진짜 정답'들이 지금 방향으로 상위권에 몰립니다. 방향은 정상입니다.")
    elif med_old < 0.4 and med_now > 0.6:
        log("  -> [경고] 예전 방향(높을수록 유력)이 오히려 더 그럴듯합니다. 방향이 다시 바뀐 것으로 보이니 코드를 재검토하세요.")
    else:
        log("  -> 어느 방향도 뚜렷하지 않습니다. 방향 문제가 아니라 다른 원인(체크포인트/그래프 불일치 등)을 의심해야 합니다.")

    return result


def main(argv=None):

    import pandas as pd

    args = parse_args(argv)

    if args.sanity_check:

        tx_data, model, evaluator = load_txgnn(args)

        result = sanity_check_mode(args, tx_data, evaluator)

        out = args.out or "txgnn_sanity_check.csv"
        result.to_csv(out, index=False, encoding="utf-8-sig")

        print("\n저장 완료: %s" % out)

        return result

    if not args.run_dir:

        args.run_dir = find_latest_run_dir(args.out_dir)

        if not args.run_dir:
            raise SystemExit(
                "--run-dir 가 필요합니다 (자동으로 찾을 step3_* 폴더도 %s 안에 없습니다).\n"
                "  점검만 하려면 --sanity-check N 을 쓰세요 (--run-dir 없이 동작)." % args.out_dir
            )

        print("[참고] --run-dir 생략 -> 가장 최근 폴더 자동 사용: %s" % args.run_dir)

    drugs_path = os.path.join(args.run_dir, "txgnn_drugs.csv")
    dis_path = os.path.join(args.run_dir, "txgnn_diseases.csv")

    for path in (drugs_path, dis_path):
        if not os.path.exists(path):
            raise FileNotFoundError("%s 가 없습니다. 먼저 step3_ot_txgnn_link.py prepare 를 실행하세요." % path)

    drugs = pd.read_csv(drugs_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    diseases = pd.read_csv(dis_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    if len(drugs) == 0:
        raise SystemExit(
            "txgnn_drugs.csv 가 비어 있습니다. 입력 약물이 PrimeKG 에 없었다는 뜻입니다.\n"
            "  --drug-id / --drug-name 을 지정하거나, prepare 에 --proxy-from-targets(또는 --proxy-if-missing)를 붙여 다시 실행하세요."
        )

    if len(diseases) == 0:
        raise SystemExit("txgnn_diseases.csv 가 비어 있습니다 (매핑된 질병이 없음).")

    print("약물 %d개 (query %d, proxy %d), 질병 %d개" % (
        len(drugs), int((drugs["role"] == "query").sum()), int((drugs["role"] == "proxy").sum()), len(diseases)))

    tx_data, model, evaluator = load_txgnn(args)

    lookup, source = build_lookup(tx_data, args.data_dir)

    if args.inspect:
        inspect_mode(args, tx_data, evaluator, drugs, diseases, lookup, source)
        return None

    scores = score_mode(args, evaluator, drugs, diseases, lookup, source)

    out = args.out or os.path.join(args.run_dir, "txgnn_scores.csv")
    scores.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n저장 완료: %s" % out)
    print("다음: python step3_ot_txgnn_link.py finalize --run-dir \"%s\" --txgnn-scores \"%s\"" % (args.run_dir, out))

    return scores


if __name__ == "__main__":
    main()
