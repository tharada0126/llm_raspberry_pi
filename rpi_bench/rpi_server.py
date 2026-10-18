#!/usr/bin/env python3
"""Raspberry Pi 側サーバ.

PC (pc_controller.py) からの HTTP/JSON 指示を受けて Ollama を操作する。

  GET  /health    : 生存確認
  GET  /time      : RPi の時刻 (PC との時計オフセット推定用)
  POST /prepare   : Ollama 準備 (ollama serve 起動、モデルダウンロード、モデルロード)
  POST /check     : 動作確認 (短いプロンプトで Ollama 実行)
  POST /generate  : ベンチマークのプロンプトで Ollama 実行し、結果を 1 つの JSON で返す
  POST /finish    : 終了指示 (モデルのアンロードと削除)

リクエストは 1 件ずつ逐次処理する (HTTPServer はシングルスレッド)。
認証は無いので、通常は pc_controller.py --ssh で起動し、127.0.0.1 で待ち受けて
SSH ポートフォワード経由で使う。
"""
import argparse
import json
import logging
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ollama import Client  # noqa: E402

from ollama_utils import call_ollama_generate_with_memory  # noqa: E402

# ollama_utils.call_ollama_generate は localhost:11434 固定なので合わせる
OLLAMA_URL = "http://localhost:11434"
CHECK_PROMPT = "Hello, this is a warm-up prompt to initialize the model."

log = logging.getLogger("rpi_server")


class OllamaManager:
    def __init__(self, serve_log: str):
        self.client = Client(host=OLLAMA_URL)
        self.serve_log = serve_log
        self.serve_proc = None

    def is_up(self) -> bool:
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def ensure_serve(self, timeout_sec: float = 60.0) -> None:
        if self.is_up():
            return
        log.info("Starting 'ollama serve' (log: %s)", self.serve_log)
        logf = open(self.serve_log, "ab")
        self.serve_proc = subprocess.Popen(["ollama", "serve"], stdout=logf, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.is_up():
                return
            if self.serve_proc.poll() is not None:
                raise RuntimeError(f"ollama serve exited with code {self.serve_proc.returncode}")
            time.sleep(0.5)
        raise RuntimeError("ollama serve did not become ready")

    def stop_serve(self) -> None:
        if self.serve_proc is not None and self.serve_proc.poll() is None:
            log.info("Stopping 'ollama serve'")
            self.serve_proc.terminate()
            try:
                self.serve_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.serve_proc.kill()

    def prepare(self, req: dict) -> dict:
        model = req["model"]
        self.ensure_serve()

        t0 = time.time()
        last_status = None
        for progress in self.client.pull(model, stream=True):
            status = progress.get("status")
            if status != last_status:
                log.info("pull %s: %s", model, status)
                last_status = status
        pull_sec = time.time() - t0

        # 空プロンプトの generate でモデルをメモリにロードする
        t1 = time.time()
        self.client.generate(model=model, prompt="")
        load_sec = time.time() - t1
        log.info("Prepared %s (pull %.1fs, load %.1fs)", model, pull_sec, load_sec)
        return {"model": model, "pull_sec": pull_sec, "load_sec": load_sec}

    def check(self, req: dict) -> dict:
        model = req["model"]
        completion, _, ttft = call_ollama_generate_with_memory(
            model=model, prompt=req.get("prompt") or CHECK_PROMPT, num_predict=req.get("num_predict", 32)
        )
        text = completion.choices[0].message.content or ""
        if not text.strip():
            raise RuntimeError("Empty response from model")
        return {"model": model, "completion": text, "ttft": ttft}

    def generate(self, req: dict) -> dict:
        model = req["model"]
        t_start = time.time()
        completion, rss_trace, ttft = call_ollama_generate_with_memory(
            model=model,
            prompt=req["prompt"],
            system=req.get("system"),
            num_predict=req.get("num_predict", 512),
        )
        t_end = time.time()
        usage = completion.usage
        return {
            "task_id": req.get("task_id"),
            "model": model,
            "hostname": socket.gethostname(),
            "completion": completion.choices[0].message.content or "",
            "finish_reason": completion.choices[0].finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "ttft": ttft,
            "rss_baseline_bytes": rss_trace.baseline_rss,
            "rss_peak_bytes": rss_trace.peak_rss,
            "rss_final_bytes": rss_trace.final_rss,
            "rpi_start_time": t_start,
            "rpi_end_time": t_end,
            "latency_sec": t_end - t_start,
        }

    def finish(self, req: dict) -> dict:
        model = req["model"]
        try:
            self.client.generate(model=model, prompt="", keep_alive=0)  # unload
        except Exception as e:
            log.warning("Unload of %s failed: %s", model, e)
        self.client.delete(model)
        log.info("Deleted %s", model)
        return {"model": model, "deleted": True}


def make_handler(mgr: OllamaManager):
    post_routes = {
        "/prepare": mgr.prepare,
        "/check": mgr.check,
        "/generate": mgr.generate,
        "/finish": mgr.finish,
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            now = time.time()
            if self.path == "/time":
                self._send(200, {"ok": True, "time": now})
            elif self.path == "/health":
                self._send(200, {"ok": True, "time": now, "hostname": socket.gethostname(), "ollama_up": mgr.is_up()})
            else:
                self._send(404, {"ok": False, "error": f"Unknown path: {self.path}"})

        def do_POST(self):
            route = post_routes.get(self.path)
            if route is None:
                self._send(404, {"ok": False, "error": f"Unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                result = route(req)
                self._send(200, {"ok": True, **result})
            except Exception as e:
                log.exception("%s failed", self.path)
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi side server for PC-controlled Ollama benchmarks.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to accept direct (non-SSH) connections.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--serve_log", default="/tmp/ollama_serve.log", help="Log file when this server starts ollama.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    mgr = OllamaManager(serve_log=args.serve_log)
    mgr.ensure_serve()

    # SSH 切断 (SIGHUP) や kill (SIGTERM) でも finally で ollama serve を止める
    def _exit(signum, frame):
        raise SystemExit(f"signal {signum}")

    for sig in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _exit)

    server = HTTPServer((args.host, args.port), make_handler(mgr))
    log.info("Listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit) as e:
        log.info("Shutting down (%s)", e or "KeyboardInterrupt")
    finally:
        server.server_close()
        mgr.stop_serve()


if __name__ == "__main__":
    main()
