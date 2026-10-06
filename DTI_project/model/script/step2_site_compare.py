"""
Attention Peak 구간 vs UniProt 알려진 Active / Binding site 비교 (원본 UniProt 좌표계)

왜 필요한가
    분석에 쓰는 서열은 preprocess_aa_length_to_1024.py 로 1024aa 이하로 잘린 "window" 이다.
    peak_start/peak_end 는 그 window 안의 번호(1~1024)라서, UniProt 의 site 좌표(전체 서열 기준)와
    직접 비교할 수 없다. -> window 가 원본 서열의 어디(offset)에서 시작하는지 복원해서 좌표를 옮긴다.

전처리 규칙(업로드된 crop 스크립트)과 복원 방법
    1) 길이 <= 1024                : 자르지 않은 원본 전체 서열              -> offset = 0
    2) Mutant 길이 > 1024          : 변이 위치를 중심으로 window (id 에 좌표 없음)
                                     -> canonical 서열에서 "1잔기 차이"로 위치를 찾아 복원
    3) Paired WT (변이체와 같은 구간): id = WT_<acc>_win_<start>_<end>      -> id 에서 읽음
    4) Standalone WT > 1024        : id = WT_<acc>_win_AS_<start>_<end>  (첫 Active/Binding site 중심)
                                         WT_<acc>_win_MID_<start>_<end> (서열 중앙)
    좌표 규약: [start:end) 0-based 반개구간, end-start = 1024.
              window 안 i번째 잔기(1-based) = 원본 위치 start + i.
    (같은 단백질의 서로 다른 window 37,177쌍이 겹치는 구간에서 100% 일치함을 확인한 규약)

    복원한 offset 은 UniProt canonical 서열과 대조해 검증한다. 맞지 않으면(이소폼, UniProt 버전 차이 등)
    좌표를 믿을 수 없으므로 비교하지 않고 이유를 status 에 남긴다.

비교 시 주의점 (결과 해석에 반영됨)
    - 알려진 site 가 window 밖에 있으면 "불일치"가 아니라 "판정 불가(no_site_in_window)" 이다.
    - AS window 는 첫 Active/Binding site 를 중심에 두고 만든 것이므로, 그 site 자신(anchor)은
      비교 대상에서 뺀다 (window 를 그 site 기준으로 잘랐으니 독립적인 근거가 될 수 없음).
    - 변이 중심 window 는 변이 위치가 window 중앙 근처라서, peak 가 중앙에 치우치는지도 기록한다.
    - "우연히 겹칠 확률"(chance_overlap_prob)을 함께 계산한다: 같은 폭의 창을 window 안에 무작위로
      놓았을 때 site 와 겹칠 확률. site 가 많이 주석된 단백질은 우연히도 잘 겹친다.
"""

import os
import re
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# 0. 설정
# ============================================================

WINDOW_LEN = 1024                       # 전처리 crop 길이

# 비교 대상 UniProt feature 종류 (전처리에서 AS window 를 만들 때 쓴 것과 동일)
SITE_FEATURE_TYPES = ("Active site", "Binding site")
# AS window 의 중심(anchor)을 재현할 때 쓰는 종류 (전처리 코드와 반드시 같아야 함)
ANCHOR_FEATURE_TYPES = ("Active site", "Binding site")

SITE_TOLERANCE = 5                      # peak 가 site "근처"라고 볼 residue 거리
CACHE_SCHEMA = 1


# ============================================================
# 1. id 파싱
# ============================================================

_RE_WT_WIN = re.compile(r"^WT_(?P<acc>.+?)_win_(?:(?P<tag>AS|MID)_)?(?P<a>\d+)_(?P<b>\d+)$")
_RE_WT = re.compile(r"^WT_(?P<acc>[A-Za-z0-9\-]+)$")
_RE_MUT = re.compile(r"^(?P<acc>.+)_(?P<wt>[A-Z\*])(?P<pos>\d+)(?P<mut>[A-Z\*])$")


def canonical_accession(acc):
    """이소폼 표기(-N)를 뗀 UniProt 기본 accession."""
    return str(acc).split("-")[0]


def parse_id_info(seq_id):
    """
    원본 CSV 의 id 에서 서열 종류/window 좌표/변이 정보를 읽는다.
    kind: "wt_window" | "wt" | "mutant" | "unknown"
    """

    seq_id = str(seq_id).strip()

    m = _RE_WT_WIN.match(seq_id)
    if m:
        return {
            "kind": "wt_window", "accession": m.group("acc"),
            "tag": m.group("tag"),                       # "AS" / "MID" / None
            "win_start": int(m.group("a")), "win_end": int(m.group("b")),
        }

    m = _RE_WT.match(seq_id)
    if m:
        return {"kind": "wt", "accession": m.group("acc"), "tag": None,
                "win_start": None, "win_end": None}

    m = _RE_MUT.match(seq_id)
    if m:
        return {
            "kind": "mutant", "accession": m.group("acc"), "tag": None,
            "win_start": None, "win_end": None,
            "wt_aa": m.group("wt"), "mut_pos": int(m.group("pos")), "mut_aa": m.group("mut"),
        }

    return {"kind": "unknown", "accession": seq_id, "tag": None,
            "win_start": None, "win_end": None}


def crop_start(center0, seq_len, max_len=WINDOW_LEN):
    """전처리 get_crop_bounds 와 동일한 규칙으로 window 시작(0-based)을 계산."""

    half = max_len // 2
    start = max(0, center0 - half)
    end = start + max_len

    if end > seq_len:
        start = max(0, seq_len - max_len)

    return int(start)


# ============================================================
# 2. window offset 복원
# ============================================================

def _mismatches(canonical, start, window, limit=None):
    """canonical[start:start+len(window)] 와 window 의 불일치 개수 (범위를 벗어나면 None)."""

    end = start + len(window)

    if start < 0 or end > len(canonical):
        return None

    count = 0

    for x, y in zip(canonical[start:end], window):
        if x != y:
            count += 1
            if limit is not None and count > limit:
                return count

    return count


def locate_window(canonical, window, max_mismatch=1):
    """
    canonical 안에서 window 가 놓인 위치(0-based)를 찾는다. 불일치 max_mismatch(0 또는 1)개까지 허용.
    (불일치가 1개 이하면 앞/뒤 절반 중 하나는 반드시 정확히 일치하므로 절반을 먼저 찾고 전체를 검증)
    반환: [(offset, mismatches), ...]  불일치 적은 순
    """

    if max_mismatch not in (0, 1):
        raise ValueError("max_mismatch 는 0 또는 1 만 지원합니다.")

    length = len(window)

    if length == 0 or length > len(canonical):
        return []

    half = length // 2

    pieces = [(0, window)] if max_mismatch == 0 else [(0, window[:half]), (half, window[half:])]

    found = {}

    for piece_offset, piece in pieces:

        if not piece:
            continue

        idx = canonical.find(piece)
        guard = 0

        while idx != -1 and guard < 20000:

            start = idx - piece_offset

            if 0 <= start <= len(canonical) - length and start not in found:
                mm = _mismatches(canonical, start, window, limit=max_mismatch)
                if mm is not None and mm <= max_mismatch:
                    found[start] = mm

            idx = canonical.find(piece, idx + 1)
            guard += 1

    return sorted(found.items(), key=lambda kv: (kv[1], kv[0]))


def resolve_window(info, window_seq, canonical):
    """
    window 가 원본(canonical) 서열의 어디에서 시작하는지 결정한다.
    반환: {"offset": 0-based 시작 또는 None, "status": 문자열, "mismatches": 정수 또는 None}
    """

    length = len(window_seq)
    is_mutant = info["kind"] == "mutant"
    max_mm = 1 if is_mutant else 0

    # ---- canonical 서열을 못 구한 경우: id 가 좌표를 알려주는 경우만 신뢰 ----
    if canonical is None:

        if info.get("win_start") is not None:
            return {"offset": info["win_start"], "status": "id_encoded_unverified", "mismatches": None}

        if length < WINDOW_LEN:
            return {"offset": 0, "status": "full_length_unverified", "mismatches": None}

        return {"offset": None, "status": "unresolved_no_canonical", "mismatches": None}

    # ---- id 에 좌표가 있는 경우 (WT window / AS / MID) -> 검증만 ----
    if info.get("win_start") is not None:

        mm = _mismatches(canonical, info["win_start"], window_seq, limit=max_mm)

        if mm is not None and mm <= max_mm:
            return {"offset": info["win_start"], "status": "id_encoded_verified", "mismatches": mm}

        return {"offset": None, "status": "id_encoded_MISMATCH", "mismatches": mm}

    # ---- 자르지 않은 전체 서열 (offset 0) ----
    if length <= len(canonical):
        mm0 = _mismatches(canonical, 0, window_seq, limit=max_mm) if length == len(canonical) else None

        if mm0 is not None and mm0 <= max_mm:
            return {"offset": 0, "status": "full_length_verified", "mismatches": mm0}

    # ---- 변이 중심 window: canonical 에서 위치 검색 ----
    candidates = locate_window(canonical, window_seq, max_mm)

    if is_mutant:

        pos, wt_aa, mut_aa = info["mut_pos"], info["wt_aa"], info["mut_aa"]

        candidates = [
            (a, mm) for a, mm in candidates
            if a < pos <= a + length
            and window_seq[pos - a - 1] == mut_aa
            and canonical[pos - 1] == wt_aa
        ]

    if len(candidates) == 1:
        return {"offset": candidates[0][0], "status": "located_unique", "mismatches": candidates[0][1]}

    if len(candidates) > 1:
        return {"offset": None, "status": "ambiguous_repeat", "mismatches": None}

    return {"offset": None, "status": "not_found_in_canonical", "mismatches": None}


def window_design(info, offset, canonical_len, length):
    """window 가 어떤 기준으로 만들어졌는지 (해석 시 편향 판단용)."""

    if info["kind"] == "wt_window":
        return {"AS": "site_centered(AS)", "MID": "protein_middle(MID)"}.get(
            info["tag"], "paired_WT_window(mutation_centered)")

    if canonical_len is not None and length < canonical_len:
        return "mutation_centered" if info["kind"] == "mutant" else "cropped"

    if canonical_len is None and length >= WINDOW_LEN and info["kind"] == "mutant":
        return "mutation_centered"

    return "full_length"


# ============================================================
# 3. UniProt 조회 (+ 로컬 캐시)
# ============================================================

def parse_uniprot_entry(entry_json):
    """
    UniProt REST JSON 에서 canonical 서열과 site feature 를 뽑는다.
    반환: {"sequence": str, "sites": [{type, start, end, description, order}], "anchor_order": int|None}
    anchor_order: 전처리에서 AS window 의 중심으로 쓴 feature(첫 Active/Binding site)의 sites 내 order
    """

    sequence = (entry_json.get("sequence") or {}).get("value", "")

    wanted = set(SITE_FEATURE_TYPES) | set(ANCHOR_FEATURE_TYPES)

    sites = []
    anchor_order = None

    for feature in entry_json.get("features", []):

        f_type = feature.get("type")

        if f_type not in wanted:
            continue

        location = feature.get("location") or {}
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")

        # 위치가 불명확한 feature 는 좌표 비교에 쓸 수 없음
        if not isinstance(start, int):
            continue

        if not isinstance(end, int):
            end = start

        order = len(sites)

        sites.append({
            "type": f_type,
            "start": int(start),
            "end": int(end),
            "description": feature.get("description", "") or "",
            "order": order,
        })

        # 전처리: 첫 번째 Active/Binding site 의 시작 위치를 window 중심으로 사용
        if anchor_order is None and f_type in ANCHOR_FEATURE_TYPES:
            anchor_order = order

    return {"sequence": sequence, "sites": sites, "anchor_order": anchor_order}


def fetch_uniprot_entry(accession, cache_dir=None, timeout=15, retries=2, pause=0.1):
    """
    UniProt 에서 서열 + site 정보를 가져온다. cache_dir 에 accession 별 JSON 으로 저장해 재사용한다.
    반환: (entry 또는 None, status)
    status: "cache" / "fetched" / "not_found" / "http_<코드>" / "connection_failed" / "requests_missing"
    """

    cache_path = os.path.join(cache_dir, f"{accession}.json") if cache_dir else None

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("schema") == CACHE_SCHEMA:
                return cached["entry"], "cache"
        except (OSError, ValueError, KeyError):
            pass                                            # 손상된 캐시는 무시하고 다시 조회

    try:
        import requests
    except ImportError:
        return None, "requests_missing"

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"

    last_status = "connection_failed"

    for attempt in range(retries + 1):

        try:
            response = requests.get(url, timeout=timeout)
        except Exception:
            last_status = "connection_failed"
            time.sleep(0.5 * (attempt + 1))
            continue

        if response.status_code == 200:

            entry = parse_uniprot_entry(response.json())

            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"schema": CACHE_SCHEMA, "accession": accession,
                               "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "entry": entry}, f, ensure_ascii=False)

            time.sleep(pause)                               # 서버에 부담을 주지 않도록

            return entry, "fetched"

        if response.status_code == 404:
            return None, "not_found"

        last_status = f"http_{response.status_code}"

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.0 * (attempt + 1))
            continue

        break

    return None, last_status


# ============================================================
# 4. peak 구간 vs site 비교
# ============================================================

def compare_peak_to_sites(attention, peak_start, peak_end, site_positions, tolerance=SITE_TOLERANCE):
    """
    window 좌표(1-based)에서 attention peak 구간과 site 잔기들을 비교한다.

    반환 지표
        overlap            : peak 구간 안에 site 잔기가 있는가
        near               : peak 구간에서 tolerance residue 이내에 site 잔기가 있는가
        distance           : peak 구간과 가장 가까운 site 잔기까지의 거리 (겹치면 0)
        chance_overlap     : 같은 폭의 창을 window 안에 무작위로 놓았을 때 site 와 겹칠 확률 (정확 계산)
        chance_near        : 위와 같되 tolerance 만큼 넓혀서 볼 때의 확률
        site_attention_enrichment : site 주변(±tolerance) 영역이 받은 attention 질량 / 그 영역이 차지하는 면적 비율
    """

    att = np.asarray(attention, dtype=np.float64)
    n = len(att)

    total = att.sum()
    att = att / total if total > 0 else np.full(n, 1.0 / n)

    site_positions = sorted({int(p) for p in site_positions if 1 <= int(p) <= n})

    if not site_positions:
        return None

    mask = np.zeros(n + 2, dtype=bool)                       # 1-based
    mask[site_positions] = True

    peak_start = int(peak_start)
    peak_end = int(peak_end)
    width = peak_end - peak_start + 1

    positions = np.array(site_positions)

    overlap = bool(mask[peak_start:peak_end + 1].any())

    left = positions[positions < peak_start]
    right = positions[positions > peak_end]

    distances = []
    if len(left):
        distances.append(peak_start - int(left.max()))
    if len(right):
        distances.append(int(right.min()) - peak_end)

    distance = 0 if overlap else min(distances)

    # ---- 우연히 겹칠 확률: 창 시작 위치를 균등하게 놓았을 때 ----
    csum = np.concatenate([[0], np.cumsum(mask[1:n + 1])])         # csum[k] = 1..k 중 site 개수

    def chance(expand):
        starts = np.arange(1, n - width + 2)
        lo = np.clip(starts - expand, 1, n)
        hi = np.clip(starts + width - 1 + expand, 1, n)
        return float(((csum[hi] - csum[lo - 1]) > 0).mean())

    # ---- site 주변 영역의 attention 질량 농축도 ----
    kernel = np.ones(2 * tolerance + 1)
    zone = np.convolve(mask[1:n + 1].astype(float), kernel, mode="same") > 0
    zone_fraction = float(zone.mean())
    zone_mass = float(att[zone].sum())
    enrichment = zone_mass / zone_fraction if zone_fraction > 0 else float("nan")

    return {
        "overlap": overlap,
        "near": bool(distance <= tolerance),
        "distance": int(distance),
        "chance_overlap": chance(0),
        "chance_near": chance(tolerance),
        "site_zone_attention_mass": zone_mass,
        "site_zone_fraction": zone_fraction,
        "site_attention_enrichment": float(enrichment),
    }


def _format_site(site):
    span = f"{site['start']}" if site["start"] == site["end"] else f"{site['start']}-{site['end']}"
    text = f"{site['type']}({span})"
    return f"{text}:{site['description']}" if site["description"] else text


def compare_protein(seq_id, window_seq, attention, peak_start, peak_end,
                    entry, entry_status, tolerance=SITE_TOLERANCE):
    """
    단백질 하나: window offset 복원 -> peak 를 원본 좌표로 변환 -> 알려진 site 와 비교.
    항상 하나의 dict(한 행)를 반환하며, 비교가 불가능하면 site_status 에 이유를 적는다.
    """

    info = parse_id_info(seq_id)
    length = len(window_seq)

    row = {
        "id": seq_id,
        "site_status": None,
        "window_design": None,
        "coordinate_status": None,
        "window_start_orig": None,
        "window_end_orig": None,
        "peak_orig_start": None,
        "peak_orig_end": None,
        "peak_center_from_window_center": None,
        "canonical_length": None,
        "n_sites_total": None,
        "n_sites_in_window": None,
        "site_coverage": None,
        "anchor_excluded": False,
        "anchor_consistent": None,
        "peak_overlaps_site": None,
        "peak_near_site": None,
        "peak_to_site_distance": None,
        "chance_overlap_prob": None,
        "chance_near_prob": None,
        "site_attention_enrichment": None,
        "nearest_site": None,
        "sites_in_window": None,
        "mutation_orig_pos": info.get("mut_pos"),
        "peak_to_mutation_distance": None,
    }

    canonical = entry["sequence"] if entry and entry.get("sequence") else None

    if entry is None:
        row["site_status"] = f"uniprot_unavailable({entry_status})"

    # ---- 1) window offset 복원 + 검증 ----
    resolved = resolve_window(info, window_seq, canonical)

    row["coordinate_status"] = resolved["status"]
    row["canonical_length"] = len(canonical) if canonical else None
    row["window_design"] = window_design(info, resolved["offset"], row["canonical_length"], length)

    offset = resolved["offset"]

    if offset is None:
        if row["site_status"] is None:
            row["site_status"] = "coordinate_unresolved"
        return row

    # ---- 2) peak 를 원본 좌표(1-based)로 ----
    row["window_start_orig"] = offset + 1
    row["window_end_orig"] = offset + length
    row["peak_orig_start"] = offset + int(peak_start)
    row["peak_orig_end"] = offset + int(peak_end)

    peak_center = (int(peak_start) + int(peak_end)) / 2.0
    row["peak_center_from_window_center"] = float(peak_center - (length + 1) / 2.0)

    if info["kind"] == "mutant" and info.get("mut_pos"):
        mp = info["mut_pos"] - offset                        # window 안 변이 위치
        if 1 <= mp <= length:
            row["peak_to_mutation_distance"] = int(
                0 if int(peak_start) <= mp <= int(peak_end)
                else min(abs(mp - int(peak_start)), abs(mp - int(peak_end)))
            )

    if entry is None:
        return row

    # ---- 3) site 목록 준비 (AS window 의 anchor 제외) ----
    sites = [s for s in entry["sites"] if s["type"] in SITE_FEATURE_TYPES]

    if info["tag"] == "AS" and entry.get("anchor_order") is not None:

        anchor = entry["sites"][entry["anchor_order"]]

        # 전처리 규칙으로 window 시작을 다시 계산해 anchor 재현 여부 확인
        expected = crop_start(anchor["start"] - 1, len(canonical))
        row["anchor_consistent"] = bool(expected == offset)

        sites = [s for s in sites if s["order"] != entry["anchor_order"]]
        row["anchor_excluded"] = True

    row["n_sites_total"] = len(sites)

    if not sites:
        row["site_status"] = "no_site_annotation"
        return row

    # ---- 4) window 안으로 들어오는 site 만 비교 ----
    win_lo, win_hi = offset + 1, offset + length

    in_window = [s for s in sites if s["end"] >= win_lo and s["start"] <= win_hi]

    row["n_sites_in_window"] = len(in_window)
    row["site_coverage"] = len(in_window) / len(sites)

    if not in_window:
        row["site_status"] = "no_site_in_window"
        return row

    local_positions = set()

    for s in in_window:
        lo = max(s["start"], win_lo) - offset
        hi = min(s["end"], win_hi) - offset
        local_positions.update(range(lo, hi + 1))

    row["sites_in_window"] = " | ".join(_format_site(s) for s in in_window)

    cmp = compare_peak_to_sites(attention, peak_start, peak_end, local_positions, tolerance)

    row["peak_overlaps_site"] = cmp["overlap"]
    row["peak_near_site"] = cmp["near"]
    row["peak_to_site_distance"] = cmp["distance"]
    row["chance_overlap_prob"] = cmp["chance_overlap"]
    row["chance_near_prob"] = cmp["chance_near"]
    row["site_attention_enrichment"] = cmp["site_attention_enrichment"]

    # peak 에서 가장 가까운 site (설명용)
    def gap(s):
        lo, hi = s["start"] - offset, s["end"] - offset
        ps, pe = int(peak_start), int(peak_end)
        return 0 if (lo <= pe and hi >= ps) else min(abs(lo - pe), abs(hi - ps))

    nearest = min(in_window, key=gap)
    row["nearest_site"] = _format_site(nearest) + f" @orig {nearest['start']}-{nearest['end']}"

    row["site_status"] = "ok"

    return row


def compare_many(items, cache_dir=None, tolerance=SITE_TOLERANCE, verbose=True):
    """
    items: [{"id", "window_seq", "attention", "peak_start", "peak_end"}, ...]
    한 단백질에서 오류가 나도 나머지는 계속 진행한다. 반환: DataFrame (단백질당 한 행)
    """

    rows = []
    entries = {}

    for k, item in enumerate(items, start=1):

        seq_id = item["id"]
        acc = canonical_accession(parse_id_info(seq_id)["accession"])

        try:

            if acc not in entries:
                entries[acc] = fetch_uniprot_entry(acc, cache_dir=cache_dir)

            entry, entry_status = entries[acc]

            row = compare_protein(
                seq_id, item["window_seq"], item["attention"],
                item["peak_start"], item["peak_end"], entry, entry_status, tolerance
            )

        except Exception as e:                              # 한 단백질의 오류가 전체를 멈추지 않도록
            row = {"id": seq_id, "site_status": f"error: {type(e).__name__}: {e}"}

        rows.append(row)

        if verbose:
            print(f"  site 비교 {k}/{len(items)}  {seq_id:<28} -> {row.get('site_status')}")

    return pd.DataFrame(rows)
