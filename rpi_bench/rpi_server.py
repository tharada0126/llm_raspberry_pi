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

# 推論・ロード・計測は main.py (RPi 単体実行) と同じ関数を使う
from main import WARMUP_PROMPT, resolve_think  # noqa: E402
from ollama_utils import (  # noqa: E402
    chat_generate,
    describe_model,
    load_model,
    ollama_version,
    unload_all_models,
    unload_model,
)
from system_monitor import ResourceMonitor, read_device_model  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434"

log = logging.getLogger("rpi_server")


class OllamaManager:
    def __init__(self, serve_log: str, poll_sec: float):
        self.client = Client(host=OLLAMA_URL)
        self.serve_log = serve_log
        self.serve_proc = None
        self.monitor = ResourceMonitor(poll_interval_sec=poll_sec)
        # /prepare で決まる実行設定 (main.py の options / think / keep_alive と同じ)
        self.options: dict = {}
        self.think = None
        self.keep_alive = "60m"

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

        # 他のモデルが常駐していない状態から、推論と同じ options でロードする
        unloaded = unload_all_models(self.client)
        self.options = req.get("options") or {}
        self.keep_alive = req.get("keep_alive", "60m")
        self.think = resolve_think(self.client, model, req.get("think", "auto"))
        t1 = time.time()
        load_model(self.client, model, self.options, keep_alive=self.keep_alive)
        load_sec = time.time() - t1
        log.info("Prepared %s (pull %.1fs, load %.1fs, options %s, think %s)",
                 model, pull_sec, load_sec, self.options, self.think)  # fmt: skip
        try:
            model_info = describe_model(self.client, model)
        except Exception as e:
            model_info = {"error": str(e)}
        return {
            "model": model,
            "pull_sec": pull_sec,
            "load_sec": load_sec,
            "unloaded_models": unloaded,
            "options": self.options,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "model_info": model_info,
            "hostname": socket.gethostname(),
            "device_model": read_device_model(),
            "ollama_version": ollama_version(),
        }

    def check(self, req: dict) -> dict:
        model = req["model"]
        result = chat_generate(
            self.client, model, WARMUP_PROMPT, {**self.options, "num_predict": 32},
            think=self.think, keep_alive=self.keep_alive,
        )  # fmt: skip
        if not (result.content or result.thinking).strip():
            raise RuntimeError("Empty response from model")
        return {"model": model, "completion": result.content, "ttft_sec": result.ttft_sec}

    def generate(self, req: dict) -> dict:
        model = req["model"]
        self.monitor.start()
        result = chat_generate(
            self.client, model, req["prompt"], self.options, system=req.get("system"),
            think=self.think, keep_alive=self.keep_alive,
        )  # fmt: skip
        resource = self.monitor.stop()
        return {
            "task_id": req.get("task_id"),
            "model": model,
            "completion": result.content,
            "thinking": result.thinking,
            # t_start_unix / t_end_unix / latency_sec / ttft_sec / tokens / Ollama 内部時間
            **result.profile_fields(),
            # メモリ・CPU 温度・周波数・スロットリング
            **resource,
        }

    def finish(self, req: dict) -> dict:
        model = req["model"]
        try:
            unload_model(self.client, model)
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
    parser.add_argument("--poll_sec", type=float, default=0.1, help="Resource polling interval during inference.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    mgr = OllamaManager(serve_log=args.serve_log, poll_sec=args.poll_sec)
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
