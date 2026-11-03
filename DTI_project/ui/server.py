"""
DTI 파이프라인 로컬 실행 서버 (torch, transformers 가 설치된 1·2단계용 conda 환경에서 실행)

    conda activate p_proj          (setup_environments.bat 기본 이름. 다른 이름으로 만들었다면 그 이름)
    python ui\\server.py            -> 브라우저에서 http://localhost:8000

분석은 서버를 켠 python 으로 실행하므로, 환경을 잘못 고르면 시작할 때 경고하고 실행 요청을 거절한다.

브라우저에서 SMILES 를 받아 아래 두 명령을 차례로 실행하고, 진행 로그와 results 폴더의 결과를 화면에 전달한다.
    1) model\\step2_Top20_regression.py --smiles ...           (Step 1 은 step2 안에서 자동 실행)
    2) model\\step3_ot_txgnn_link.py prepare --auto-chain --final-csv <1) 에서 새로 생긴 final5> ...

prepare --auto-chain 은 예전 final5_proteins.csv 가 하나라도 있으면 Step 1+2 를 건너뛰고 그 결과를 재사용하므로,
Step 2 를 항상 먼저 돌리고 새 결과를 --final-csv 로 직접 넘긴다.

표준 라이브러리만 사용하며, 127.0.0.1 에만 열려 같은 PC 의 브라우저만 접속할 수 있다.
"""

import argparse
import codecs
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

UI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UI_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

PEAK_METHODS = ("window_enrichment", "window_mass", "zscore", "entropy")
MAX_LOG_LINES = 20000

# 로그에 이 문구가 나오면 해당 단계로 넘어간 것으로 본다 (step3_ot_txgnn_link.py 의 출력 문구)
STAGE_MARKERS = (
    ("[자동 실행] TxGNN 점수 계산 (DTI_txgnn 환경) 시작", "txgnn"),
    ("이어서 finalize 자동 실행", "finalize"),
)


def missing_packages():
    """분석은 서버를 켠 python 으로 실행하므로, 그 python 에 1·2단계 필수 패키지가 있는지 확인한다."""

    import importlib.util
    return [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]


MISSING = []  # main() 에서 정한다
MISSING_MSG = ("서버를 실행한 python 에 %s 가 없어 분석을 실행할 수 없습니다. "
               "torch 와 transformers 가 설치된 conda 환경(setup_environments.bat 으로 만든 1·2단계용 환경)을 "
               "activate 한 뒤 서버를 다시 켜세요.")


def list_dirs(prefix):
    if not os.path.isdir(RESULTS_DIR):
        return set()
    return {d for d in os.listdir(RESULTS_DIR) if d.startswith(prefix) and os.path.isdir(os.path.join(RESULTS_DIR, d))}


def newest(dirs):
    return max(dirs, key=lambda d: os.path.getmtime(os.path.join(RESULTS_DIR, d))) if dirs else None


class Job:
    """한 번에 하나만 실행되는 분석 작업"""

    def __init__(self, params):
        self.params = params
        self.state = "running"          # running / done / failed / cancelled
        self.stage = "step12"           # step12 / prepare / txgnn / finalize
        self.lines = []
        self.error = None
        self.stage2_dir = None
        self.step3_dir = None
        self.started = time.time()
        self.finished = None
        self.proc = None
        self.cancel_requested = False
        self.lock = threading.Lock()

    # ---------------- 로그 ----------------
    def log(self, text, overwrite=False):
        with self.lock:
            # tqdm 진행 막대(\r)는 새 줄을 만들지 않고 마지막 줄을 덮어쓴다
            if overwrite and self.lines:
                self.lines[-1] = text
            else:
                self.lines.append(text)
                if len(self.lines) > MAX_LOG_LINES:
                    del self.lines[: len(self.lines) - MAX_LOG_LINES]
            for marker, stage in STAGE_MARKERS:
                if marker in text:
                    self.stage = stage

    def run_cmd(self, cmd):
        self.log("$ " + " ".join('"%s"' % c if " " in c else c for c in cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=MODEL_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf, progress = "", False  # progress: 마지막 줄이 \r 로 끝난 진행 막대인지
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") else self.proc.stdout.read(1)
            if not chunk:
                break
            buf += decoder.decode(chunk)
            while True:
                i = min([p for p in (buf.find("\n"), buf.find("\r")) if p >= 0], default=-1)
                if i < 0 or (buf[i] == "\r" and i == len(buf) - 1):
                    break  # 끝의 \r 은 다음 조각이 \n 인지(윈도우 줄바꿈) 보고 판단
                if buf.startswith("\r\n", i):
                    line, sep, buf = buf[:i], "\n", buf[i + 2:]
                else:
                    line, sep, buf = buf[:i], buf[i], buf[i + 1:]
                if line:
                    self.log(line, overwrite=progress)
                    progress = sep == "\r"
                elif sep == "\n":
                    if not progress:
                        self.log("")
                    progress = False
        if buf.strip("\r"):
            self.log(buf.strip("\r"), overwrite=progress)
        return self.proc.wait()

    # ---------------- 실행 ----------------
    def run(self):
        p = self.params
        try:
            before2 = list_dirs("stage2_")
            cmd2 = [sys.executable, "step2_Top20_regression.py", "--smiles", p["smiles"]]
            if p.get("top_n"):
                cmd2 += ["--top-n", str(p["top_n"])]
            if p.get("final_n"):
                cmd2 += ["--final-n", str(p["final_n"])]
            if p.get("peak_method"):
                cmd2 += ["--peak-method", p["peak_method"]]
            if self.run_cmd(cmd2) != 0:
                return self.fail("1·2단계가 실패했습니다. 로그를 확인하세요.")

            self.stage2_dir = newest(list_dirs("stage2_") - before2)
            final_csv = os.path.join(RESULTS_DIR, self.stage2_dir or "", "final5_proteins.csv")
            if not self.stage2_dir or not os.path.exists(final_csv):
                return self.fail("1·2단계는 끝났지만 새 final5_proteins.csv 를 찾지 못했습니다.")

            self.stage = "prepare"
            before3 = list_dirs("step3_")
            cmd3 = [sys.executable, "step3_ot_txgnn_link.py", "prepare", "--auto-chain",
                    "--final-csv", final_csv, "--smiles", p["smiles"]]
            if p.get("drug_id"):
                cmd3 += ["--drug-id", p["drug_id"]]
            if p.get("compound_name"):
                # 이름은 PrimeKG 약물 찾기(--drug-name)에도 쓴다. 이름이 PrimeKG 에 없으면 step3 가 SMILES 로 다시 찾는다
                cmd3 += ["--drug-name", p["compound_name"], "--compound-name", p["compound_name"]]
            if p.get("proxy"):
                cmd3.append("--proxy-if-missing")
            if p.get("write_db"):
                cmd3.append("--write-db")
            if self.run_cmd(cmd3) != 0:
                return self.fail("3단계가 실패했습니다. 로그를 확인하세요.")

            self.step3_dir = newest(list_dirs("step3_") - before3)
            # TxGNN 이 실패해도 prepare 는 정상 종료하므로 최종 표가 있는지로 판단한다
            if not self.step3_dir or not os.path.exists(os.path.join(RESULTS_DIR, self.step3_dir, "disease_candidates_final.csv")):
                if any("txgnn_drugs.csv 가 비어 있습니다" in line for line in self.lines):
                    return self.fail("입력 약물을 PrimeKG 에서 찾지 못해 TxGNN 점수를 계산하지 못했습니다. "
                                     "기존 약물이면 약물 이름(영문)이나 DrugBank ID 를 넣고, 신규 화합물이면 "
                                     "proxy-if-missing 을 체크한 뒤 다시 실행하세요.")
                return self.fail("3단계 최종 표(disease_candidates_final.csv)가 만들어지지 않았습니다. TxGNN 로그를 확인하세요.")

            self.state = "done"
        except Exception as e:
            self.fail("서버 오류: %s" % e)
        finally:
            self.finished = time.time()

    def fail(self, msg):
        if self.cancel_requested:
            self.state, self.error = "cancelled", "사용자가 중단했습니다."
        else:
            self.state, self.error = "failed", msg
        self.log("\n[실패] " + self.error)

    def cancel(self):
        self.cancel_requested = True
        if self.proc and self.proc.poll() is None:
            # step3 가 띄운 TxGNN 하위 프로세스까지 함께 종료
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.kill()

    def status(self, since):
        with self.lock:
            since = max(0, min(since, len(self.lines)))
            return {
                "state": self.state, "stage": self.stage, "error": self.error,
                "stage2_dir": self.stage2_dir, "step3_dir": self.step3_dir,
                "params": self.params, "elapsed": (self.finished or time.time()) - self.started,
                "lines": self.lines[since:], "next": len(self.lines),
            }


JOB = None
JOB_LOCK = threading.Lock()


def validate(body):
    smiles = str(body.get("smiles", "")).strip()
    if not smiles or any(c.isspace() for c in smiles):
        raise ValueError("SMILES 를 공백 없이 입력하세요.")
    params = {"smiles": smiles}
    for k in ("top_n", "final_n"):
        if body.get(k) not in (None, ""):
            v = int(body[k])
            if v < 1:
                raise ValueError("%s 는 1 이상이어야 합니다." % k)
            params[k] = v
    if body.get("peak_method"):
        if body["peak_method"] not in PEAK_METHODS:
            raise ValueError("peak_method 가 올바르지 않습니다.")
        params["peak_method"] = body["peak_method"]
    for k in ("drug_id", "compound_name"):
        v = str(body.get(k) or "").strip()
        if v:
            params[k] = v
    params["proxy"] = bool(body.get("proxy"))
    params["write_db"] = bool(body.get("write_db"))
    return params


class Handler(BaseHTTPRequestHandler):

    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # 1~2 초마다 오는 진행 조회 요청으로 터미널이 넘치지 않게 한다

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            return self.send_file(os.path.join(UI_DIR, "index.html"), "text/html; charset=utf-8")

        if url.path == "/api/list":
            # results 바로 아래 파일 + stage2_* / step3_* 폴더 안의 csv, json (캐시 폴더 제외)
            files = []
            if os.path.isdir(RESULTS_DIR):
                for name in os.listdir(RESULTS_DIR):
                    full = os.path.join(RESULTS_DIR, name)
                    if os.path.isfile(full) and name.endswith((".csv", ".json")):
                        files.append(name)
                    elif os.path.isdir(full) and name.startswith(("stage2_", "step3_")):
                        files += [name + "/" + f for f in os.listdir(full) if f.endswith((".csv", ".json"))]
            return self.send_json({"files": files})

        if url.path == "/api/file":
            rel = q.get("path", [""])[0]
            full = os.path.realpath(os.path.join(RESULTS_DIR, rel))
            if not full.startswith(os.path.realpath(RESULTS_DIR) + os.sep) or not os.path.isfile(full):
                return self.send_json({"error": "파일이 없습니다."}, 404)
            return self.send_file(full, "text/plain; charset=utf-8")

        if url.path == "/api/job":
            if JOB is None:
                return self.send_json({"state": "idle"})
            return self.send_json(JOB.status(int(q.get("since", ["0"])[0] or 0)))

        self.send_json({"error": "없는 경로입니다."}, 404)

    def do_POST(self):
        global JOB
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send_json({"error": "요청 형식이 올바르지 않습니다."}, 400)

        if url.path == "/api/run":
            try:
                params = validate(body)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            if MISSING:
                return self.send_json({"error": MISSING_MSG % ", ".join(MISSING)}, 500)
            with JOB_LOCK:
                if JOB is not None and JOB.state == "running":
                    return self.send_json({"error": "이미 실행 중인 분석이 있습니다."}, 409)
                JOB = Job(params)
                threading.Thread(target=JOB.run, daemon=True).start()
            return self.send_json({"ok": True})

        if url.path == "/api/cancel":
            if JOB is not None and JOB.state == "running":
                JOB.cancel()
            return self.send_json({"ok": True})

        self.send_json({"error": "없는 경로입니다."}, 404)


class Server(ThreadingHTTPServer):
    # 윈도우에서 SO_REUSEADDR 를 켜면 이미 쓰는 포트에 서버가 하나 더 붙어 버리므로 끈다
    allow_reuse_address = os.name != "nt"


def main():
    ap = argparse.ArgumentParser(description="DTI 파이프라인 로컬 실행 서버")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global MISSING
    MISSING = missing_packages()
    if MISSING:
        print("[경고] " + MISSING_MSG % ", ".join(MISSING) + " (지금은 결과 보기만 가능합니다)")

    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError:
        print("[실패] %d 번 포트를 이미 쓰고 있습니다. 켜 둔 서버가 있는지 확인하거나 --port 로 다른 번호를 쓰세요." % args.port)
        sys.exit(1)
    print("DTI 서버 실행 중: http://localhost:%d  (python = %s)" % (args.port, sys.executable))
    print("끄려면 Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if JOB is not None and JOB.state == "running":
            JOB.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
