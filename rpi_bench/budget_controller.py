#!/usr/bin/env python3
"""PC 側: エネルギー予算で推論を打ち切って PTME (Precision / Time / Memory / Energy) を計測する.

pc_controller.py と同じ構成 (PC + FNIRSI メータ + RPi 上の rpi_server.py) を使う。違いは推論の送り方:

  pc_controller.py   : PC -> rpi_server /generate -> Ollama (RPi 内で最後まで生成)
  budget_controller.py: PC -> (SSH ポートフォワード) -> RPi の Ollama /api/chat にストリーミングで直接送る

PC はメータのサンプルを受け取るたびに積算エネルギーを計算し、予算に達した瞬間に Ollama との
TCP 接続を切る。Ollama はクライアントの切断を検知して生成を中止する (プロンプト処理中でも
トークンを待たずに切れる)。メモリ・温度は rpi_server の /monitor_start, /monitor_stop で計測する。

for each モデル:
    model 指定 -> RPi: Ollama 準備 / 動作確認 / 時計オフセット / 待機電力 (baseline) 計測
    for each (予算, 問題)  (--order で順序を選ぶ):
        資源計測開始 (RPi)
        エネルギー計測開始 (予算を設定)
        プロンプト送信 -> Ollama からトークンを逐次受信
            予算到達 -> 接続を切って打ち切り (status = budget_exhausted)
            または生成終了 (status = completed)
        tail_sec 待って計測終了 (打ち切り後に RPi の電力が下がったことの確認用)
        資源計測終了 (RPi)
        生データ・トークン列を保存し、採点してレコードを jsonl に追記
        idling
    終了指示 -> RPi: モデル削除

予算は start (プロンプト送信) からの積算エネルギーで、プロンプト処理 (prefill) も含む。
モデルのロードは含まない (事前にロード済み)。

Python 3.9 互換.
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import json
import logging
import math
import socket
import threading
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from bench_tasks import DATASETS, Task, load_tasks, score
from pc_controller import (
    DEFAULT_MODELS,
    REPO_ROOT,
    RPiClient,
    SSHServerLauncher,
    _int_or_none,
    load_done_ids,
    safe_name,
)
from power_meter import (
    INTERVAL,
    FnirsiPowerMeter,
    NullPowerMeter,
    SimulatedPowerMeter,
    save_raw,
    summarize,
)

log = logging.getLogger("budget_controller")

DEFAULT_BUDGETS = ["100", "200", "300", "400", "500"]


# ---------------------------------------------------------------- 予算の表記


def parse_budget(v: str) -> Optional[float]:
    """'100' -> 100.0, 'inf' / 'none' -> None (無制限)."""
    if v.lower() in ("inf", "none", "unlimited"):
        return None
    b = float(v)
    if not b > 0:
        raise argparse.ArgumentTypeError(f"budget must be positive: {v}")
    return b


def budget_label(b: Optional[float]) -> str:
    if b is None:
        return "Binf"
    return f"B{int(b)}J" if float(b).is_integer() else f"B{b:g}J"


# ---------------------------------------------------------------- Ollama ストリーミング


class OllamaStream:
    """Ollama /api/chat をストリーミングで受信する 1 回分のリクエスト.

    cancel() は別スレッド (メータの受信スレッド) から呼ばれ、ソケットを shutdown して受信を即座に中断する。
    接続が切れると Ollama は生成を中止する。
    """

    def __init__(self, base_url: str, timeout_sec: float):
        u = urlparse(base_url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 11434
        self.timeout_sec = timeout_sec
        self.conn: Optional[http.client.HTTPConnection] = None
        self._cancelled = threading.Event()
        self.t_cancel: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def connect(self) -> None:
        """計測開始前に TCP 接続しておく (接続確立の時間を予算に含めないため)."""
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        self.conn.connect()
        # 小さいパケットを即送る (送信遅延を減らす)
        self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def cancel(self, _hit: Optional[dict] = None) -> None:
        with self._lock:
            if self._cancelled.is_set():
                return
            self.t_cancel = time.time()
            self._cancelled.set()
            sock = self.conn.sock if self.conn is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass

    def chat(self, payload: dict) -> dict:
        """戻り値: {"chunks": [(t_recv, content, thinking), ...], "final": dict|None, "error": str|None}."""
        chunks: List[tuple] = []
        final = None
        error = None
        try:
            if self.cancelled:
                raise ConnectionAbortedError("cancelled before sending")
            body = json.dumps(payload).encode("utf-8")
            self.conn.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
            resp = self.conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status}: {resp.read()[:500]!r}")
            while True:
                line = resp.readline()
                if not line:
                    break
                t = time.time()
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                msg = obj.get("message") or {}
                chunks.append((t, msg.get("content") or "", msg.get("thinking") or ""))
                if obj.get("done"):
                    final = obj
                    break
            if final is None and not self.cancelled:
                error = "stream ended without done"
        except Exception as e:
            if not self.cancelled:
                error = f"{type(e).__name__}: {e}"
        finally:
            self.close()
        return {"chunks": chunks, "final": final, "error": error}


# ---------------------------------------------------------------- 1 問分


def save_trace(path: Path, t_send: float, t_stop: float, chunks: List[tuple]) -> None:
    """受信したトークン列 (受信時刻は t_send からの秒数). 後から小さい予算での打ち切りを再現するのに使う."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "t_send": t_send,
        "t_stop": t_stop,
        "chunks": [[round(t - t_send, 4), c, th] for t, c, th in chunks],
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def basis_energy(win: dict, baseline_W: float) -> Optional[float]:
    """予算と同じ定義 (sum((V*I - baseline) * dt)) の区間エネルギー."""
    if win.get("energy_sum_J") is None:
        return None
    return win["energy_sum_J"] - baseline_W * win["num_samples"] * INTERVAL


def run_task(args, ctx: dict, meter, budget: Optional[float], task: Task, out_dir: Path) -> Optional[dict]:
    rpi: RPiClient = ctx["rpi"]
    model = ctx["model"]
    baseline_W = ctx["budget_baseline_W"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task.prompt}],
        "stream": True,
        "options": ctx["options"],
        "keep_alive": ctx["keep_alive"],
    }
    if ctx["think"] is not None:
        payload["think"] = ctx["think"]

    for attempt in range(1, args.max_retries + 1):
        stream = OllamaStream(ctx["ollama_url"], args.request_timeout)
        try:
            rpi.post("/monitor_start", {})
            stream.connect()
        except Exception as e:
            log.warning("[%s] attempt %d: setup failed: %s", task.task_id, attempt, e)
            stream.close()
            time.sleep(args.idle_sec)
            continue

        t_send = meter.start(budget_J=budget, baseline_W=baseline_W, on_budget=stream.cancel)
        res = stream.chat(payload)
        t_stop = stream.t_cancel if stream.cancelled else time.time()
        meter_disarm(meter)
        time.sleep(args.tail_sec)  # 打ち切り後の電力 (RPi が本当に止まったか) を記録する
        meas = meter.stop()
        try:
            resource = rpi.post("/monitor_stop", {})
        except Exception as e:
            log.warning("[%s] monitor_stop failed: %s", task.task_id, e)
            resource = {}

        if res["error"] is not None:
            log.warning("[%s] attempt %d failed: %s", task.task_id, attempt, res["error"])
            time.sleep(args.idle_sec)
            continue
        if meter.enabled and not meas.samples:
            log.warning("[%s] attempt %d: no power samples, retrying", task.task_id, attempt)
            time.sleep(args.idle_sec)
            continue
        break
    else:
        log.error("[%s] skipped after %d attempts", task.task_id, args.max_retries)
        return None

    chunks = res["chunks"]
    final = res["final"]
    cut = stream.cancelled and final is None
    content = "".join(c for _, c, _ in chunks)
    thinking = "".join(th for _, _, th in chunks)
    first = next((t for t, c, th in chunks if c or th), None)

    window = summarize(meas.samples, t_send, t_stop)
    tail = summarize(meas.samples, t_stop, meas.t_end)
    tail_last = summarize(meas.samples, max(t_stop, meas.t_end - 1.0), meas.t_end)
    win_basis = basis_energy(window, baseline_W)
    hit = meas.budget_hit
    energy = {
        "meter": None if not meter.enabled else ("simulated" if getattr(meter, "simulated", False) else "fnirsi"),
        "attempts": attempt,
        "budget_J": budget,
        "budget_basis": args.budget_basis,
        "baseline_W": ctx["baseline_W"],  # 計測した (または指定した) 待機電力
        "budget_baseline_W": baseline_W,  # 予算の積算で差し引いた電力 (gross なら 0)
        # プロンプト送信から、打ち切り (または結果受信) まで
        "window": window,
        "window_basis_J": win_basis,
        # 予算到達の検出 (メータのサンプル時刻・PC が検出した時刻・その時点の積算値)
        "budget_hit": hit,
        "cancel_delay_sec": (stream.t_cancel - hit["t_sample"]) if (hit and stream.t_cancel) else None,
        "overshoot_J": (win_basis - budget) if (cut and budget is not None and win_basis is not None) else None,
        # 打ち切り (または終了) 後の tail_sec 区間。tail_last は最後の 1 秒
        "tail": tail,
        "tail_last_mean_power_W": tail_last.get("mean_power_W"),
        "raw_file": None,
        "trace_file": None,
    }
    if meter.enabled:
        raw_path = out_dir / "energy_raw" / f"{safe_name(task.task_id)}.txt.gz"
        save_raw(raw_path, meas.samples)
        energy["raw_file"] = raw_path.relative_to(out_dir).as_posix()
    trace_path = out_dir / "trace" / f"{safe_name(task.task_id)}.json.gz"
    save_trace(trace_path, t_send, t_stop, chunks)
    energy["trace_file"] = trace_path.relative_to(out_dir).as_posix()

    if (
        ctx["baseline_W"] is not None
        and tail_last.get("mean_power_W") is not None
        and tail_last["mean_power_W"] > ctx["baseline_W"] + args.tail_warn_W
    ):
        log.warning(
            "[%s] power after stop is still %.2f W (baseline %.2f W): RPi may not have stopped generating",
            task.task_id, tail_last["mean_power_W"], ctx["baseline_W"],
        )  # fmt: skip

    f = final or {}
    record = {
        "scenario": args.scenario,
        "dataset": task.dataset,
        "task_id": task.task_id,
        "model": model,
        "budget_J": budget,
        "budget_label": budget_label(budget),
        "budget_basis": args.budget_basis,
        "status": "budget_exhausted" if cut else "completed",
        "completion": content,
        "thinking": thinking,
        "done_reason": f.get("done_reason"),
        # 打ち切った場合は受信したチャンク数 (1 チャンク ≒ 1 トークン)
        "completion_tokens": f.get("eval_count") if final else sum(1 for _, c, th in chunks if c or th),
        "prompt_tokens": f.get("prompt_eval_count"),  # 打ち切った場合は不明 (None)
        "t_start_unix": t_send,
        "t_end_unix": t_stop,
        "latency_sec": t_stop - t_send,  # PC 時刻: プロンプト送信 -> 打ち切り (または結果受信)
        "ttft_sec": (first - t_send) if first is not None else None,
        "total_duration_ns": f.get("total_duration"),
        "load_duration_ns": f.get("load_duration"),
        "prompt_eval_duration_ns": f.get("prompt_eval_duration"),
        "eval_duration_ns": f.get("eval_duration"),
        **{k: v for k, v in resource.items() if k != "t_unix"},
        **task.meta,
        **score(task, content),
        "pc_end_time": meas.t_end,
        "energy": energy,
    }
    return record


def meter_disarm(meter) -> None:
    """生成が終わった後 (tail 区間) に予算到達で on_budget が呼ばれないようにする."""
    disarm = getattr(meter, "disarm", None)
    if disarm is not None:
        disarm()


# ---------------------------------------------------------------- 集計


def write_summary(records_path: Path, summary_path: Path, model: str, dataset: str, budget: Optional[float]) -> dict:
    with records_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    summary = {
        "model_name": model,
        "dataset": dataset,
        "budget_J": budget,
        "budget_basis": records[0]["budget_basis"] if records else None,
        "num_samples": len(records),
        "frac_budget_exhausted": mean([1.0 if r["status"] == "budget_exhausted" else 0.0 for r in records]),
        "mean_energy_J": mean([r["energy"]["window"]["energy_J"] for r in records]),
        "mean_energy_basis_J": mean([r["energy"]["window_basis_J"] for r in records]),
        "mean_overshoot_J": mean([r["energy"]["overshoot_J"] for r in records]),
        "mean_latency_sec": mean([r["latency_sec"] for r in records]),
        "mean_completion_tokens": mean([r["completion_tokens"] for r in records]),
        "peak_rss_bytes": max((r.get("rss_peak_bytes") or 0 for r in records), default=None),
    }
    if records and "correct" in records[0]:
        summary["accuracy"] = mean([1.0 if r["correct"] else 0.0 for r in records])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------- 全体


def measure_baseline(meter, sec: float) -> Optional[float]:
    if sec <= 0 or not meter.enabled:
        return None
    meter.start()
    time.sleep(sec)
    meas = meter.stop()
    return summarize(meas.samples, meas.t_start, meas.t_end).get("mean_power_W")


def main():
    parser = argparse.ArgumentParser(
        description="Energy-budgeted PTME benchmark: stop Ollama on the RPi when the measured energy reaches a budget."
    )
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--ssh", help="SSH target (e.g. pi@raspberrypi.local); starts rpi_server.py over SSH.")
    conn.add_argument("--rpi_url", help="Already running rpi_server.py (e.g. http://192.168.0.10:8000).")
    parser.add_argument("--ollama_url", default=None, help="Ollama on RPi for --rpi_url mode (e.g. http://192.168.0.10:11434).")
    parser.add_argument("--ssh_remote_dir", default="~/llm_ptme", help="Repository directory on RPi.")
    parser.add_argument("--ssh_python", default="python3", help="Python on RPi (e.g. ~/venv/bin/python).")
    parser.add_argument("--ssh_sync", action="store_true", help="Copy rpi_server.py and its imports by scp.")
    parser.add_argument("--rpi_port", type=int, default=8000, help="rpi_server.py port on RPi (--ssh).")
    parser.add_argument("--local_port", type=int, default=18000, help="Local forwarded port for rpi_server (--ssh).")
    parser.add_argument("--rpi_ollama_port", type=int, default=11434, help="Ollama port on RPi (--ssh).")
    parser.add_argument("--local_ollama_port", type=int, default=21434, help="Local forwarded port for Ollama (--ssh).")
    # 予算
    parser.add_argument(
        "--budgets", nargs="+", type=parse_budget, default=[parse_budget(b) for b in DEFAULT_BUDGETS],
        help="Energy budgets [J] per problem, e.g. 100 200 300 400 500. 'inf' = no budget (reference run).",
    )  # fmt: skip
    parser.add_argument(
        "--budget_basis", choices=["gross", "net"], default="gross",
        help="gross: total RPi energy from the meter. net: minus baseline power (--idle_power_W or measured).",
    )  # fmt: skip
    parser.add_argument("--idle_power_W", type=float, default=None, help="Baseline power for --budget_basis net.")
    parser.add_argument("--baseline_sec", type=float, default=20.0, help="Measure idle power per model (model loaded).")
    parser.add_argument(
        "--order", choices=["task", "budget"], default="task",
        help="task: each problem is run at all budgets back to back (balanced if interrupted). "
        "budget: all problems at one budget, then the next budget.",
    )  # fmt: skip
    parser.add_argument("--tail_sec", type=float, default=3.0, help="Keep measuring after the stop (check RPi stopped).")
    parser.add_argument("--tail_warn_W", type=float, default=1.0, help="Warn if power after stop exceeds baseline + this.")
    # 以下は pc_controller.py と同じ
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--scenario", default="Budget", help="Label used in the output path.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results_rpi"))
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of problems per dataset.")
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--num_thread", type=_int_or_none, default=None, help="Default: Ollama's choice.")
    parser.add_argument("--num_gpu", type=_int_or_none, default=0, help="0 = CPU only, 'none' = Ollama's choice.")
    parser.add_argument("--think", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--keep_alive", default="60m")
    parser.add_argument("--idle_sec", type=float, default=5.0, help="Idle time after each problem.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=3600.0, help="Timeout [s] for RPi/Ollama requests.")
    parser.add_argument("--keep_model", action="store_true", help="Do not delete the model on RPi after a model.")
    parser.add_argument("--fnirsi_dir", default=str(REPO_ROOT.parent / "fnirsi-usb-power-data-logger"))
    parser.add_argument("--crc", action="store_true", help="Enable CRC check of power meter packets.")
    parser.add_argument("--no_meter", action="store_true", help="Run without the power meter (budgets are ignored).")
    parser.add_argument(
        "--simulate_power_W", type=float, default=None,
        help="Use a simulated constant-power meter instead of the FNIRSI meter (test the stop logic only).",
    )  # fmt: skip
    parser.add_argument("--human_eval_path", default=None, help="HumanEval.jsonl(.gz). Default: Hugging Face.")
    parser.add_argument("--gsm8k_path", default=None, help="GSM8K test.jsonl. Default: Hugging Face.")
    parser.add_argument("--gsm8k_selected_idx_file", default=str(REPO_ROOT / "scripts" / "gsm8k_selected_idx.txt"))
    parser.add_argument("--mmlu_pro_dir", default=None, help="Clone of TIGER-AI-Lab/MMLU-Pro (for initial_prompt.txt).")
    parser.add_argument(
        "--mmlu_pro_selected_ids_file", default=str(REPO_ROOT / "scripts" / "mmlu_pro_selected_ids_20.txt")
    )
    args = parser.parse_args()
    for key in ("gsm8k_selected_idx_file", "mmlu_pro_selected_ids_file"):
        if getattr(args, key).lower() in ("none", ""):
            setattr(args, key, None)
    if args.rpi_url and not args.ollama_url:
        parser.error("--rpi_url mode needs --ollama_url (Ollama on the RPi must listen on the LAN: OLLAMA_HOST=0.0.0.0)")
    if args.no_meter and args.simulate_power_W is not None:
        parser.error("--no_meter and --simulate_power_W are exclusive")
    # 重複を除き、小さい予算から (inf は最後)
    args.budgets = sorted(set(args.budgets), key=lambda b: math.inf if b is None else b)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    launcher = None
    if args.ssh:
        log_path = Path(args.output_dir) / args.scenario / "rpi_server.log"
        launcher = SSHServerLauncher(
            args.ssh, args.ssh_remote_dir, args.ssh_python, args.rpi_port, args.local_port, log_path,
            extra_forwards=[(args.local_ollama_port, args.rpi_ollama_port)],
        )  # fmt: skip
        if args.ssh_sync:
            launcher.sync()
        launcher.start()
        args.rpi_url = launcher.url
        args.ollama_url = f"http://127.0.0.1:{args.local_ollama_port}"
    try:
        run(args)
    finally:
        if launcher is not None:
            launcher.stop()


def run(args) -> None:
    options = {"temperature": 0.0, "num_predict": args.num_predict, "num_ctx": args.num_ctx}
    if args.num_gpu is not None:
        options["num_gpu"] = args.num_gpu
    if args.num_thread is not None:
        options["num_thread"] = args.num_thread
    prepare_params = {"options": options, "think": args.think, "keep_alive": args.keep_alive}

    rpi = RPiClient(args.rpi_url, args.request_timeout)
    log.info("RPi: %s", rpi.get("/health"))
    log.info("Budgets: %s (%s), order: %s", [budget_label(b) for b in args.budgets], args.budget_basis, args.order)

    tasks_by_dataset = {
        ds: load_tasks(
            ds,
            max_samples=args.max_samples,
            human_eval_path=args.human_eval_path,
            gsm8k_path=args.gsm8k_path,
            gsm8k_selected_idx_file=args.gsm8k_selected_idx_file,
            mmlu_pro_dir=args.mmlu_pro_dir,
            mmlu_pro_selected_ids_file=args.mmlu_pro_selected_ids_file,
        )
        for ds in args.datasets
    }
    for ds, tasks in tasks_by_dataset.items():
        log.info("Dataset %s: %d problems", ds, len(tasks))

    if args.no_meter:
        meter = NullPowerMeter()
        log.warning("--no_meter: budgets cannot be enforced; every problem runs to completion.")
    elif args.simulate_power_W is not None:
        meter = SimulatedPowerMeter(args.simulate_power_W)
    else:
        meter = FnirsiPowerMeter(args.fnirsi_dir, use_crc=args.crc)
    meter.open()
    scen_dir = Path(args.output_dir) / args.scenario
    scen_dir.mkdir(parents=True, exist_ok=True)
    try:
        for model in args.models:
            # 再開対応: (予算, データセット) ごとに記録済みの問題を飛ばす
            files = {}  # (budget, ds) -> (out_dir, records_path)
            done = {}
            for b in args.budgets:
                for ds in tasks_by_dataset:
                    out_dir = scen_dir / budget_label(b) / ds / safe_name(model)
                    records_path = out_dir / f"{ds}_{safe_name(model)}_records.jsonl"
                    files[(b, ds)] = (out_dir, records_path)
                    done[(b, ds)] = load_done_ids(records_path)
            jobs = []  # (budget, dataset, task)
            if args.order == "budget":
                for b in args.budgets:
                    for ds, tasks in tasks_by_dataset.items():
                        jobs += [(b, ds, t) for t in tasks if t.task_id not in done[(b, ds)]]
            else:
                for ds, tasks in tasks_by_dataset.items():
                    for t in tasks:
                        jobs += [(b, ds, t) for b in args.budgets if t.task_id not in done[(b, ds)]]
            if not jobs:
                log.info("[%s] all problems already done, skipping", model)
                continue
            n_done = sum(len(v) for v in done.values())
            log.info("========== %s: %d runs pending (%d done) ==========", model, len(jobs), n_done)

            prep = rpi.post("/prepare", {"model": model, **prepare_params})
            log.info("Prepare: %s", prep)
            log.info("Check: %s", rpi.post("/check", {"model": model}))
            clock = rpi.clock_offset()
            log.info("Clock offset (RPi - PC): %.4f s (RTT %.4f s)", *clock)

            measured_W = measure_baseline(meter, args.baseline_sec)
            if measured_W is not None:
                log.info("Idle power with %s loaded: %.3f W", model, measured_W)
            baseline_W = args.idle_power_W if args.idle_power_W is not None else measured_W
            if args.budget_basis == "net" and baseline_W is None and meter.enabled:
                raise RuntimeError("--budget_basis net needs --idle_power_W or --baseline_sec > 0")
            ctx = {
                "rpi": rpi,
                "model": model,
                "options": prep.get("options", options),
                "think": prep.get("think"),
                "keep_alive": prep.get("keep_alive", args.keep_alive),
                "ollama_url": args.ollama_url,
                "baseline_W": baseline_W,
                "budget_baseline_W": (baseline_W or 0.0) if args.budget_basis == "net" else 0.0,
            }
            with (scen_dir / "run_meta.jsonl").open("a", encoding="utf-8") as f:
                meta = {
                    "t_unix": time.time(),
                    "model": model,
                    "budgets": args.budgets,
                    "budget_basis": args.budget_basis,
                    "order": args.order,
                    "baseline_measured_W": measured_W,
                    "baseline_used_W": baseline_W,
                    "baseline_sec": args.baseline_sec,
                    "tail_sec": args.tail_sec,
                    "clock_offset_sec": clock[0],
                    "clock_rtt_sec": clock[1],
                    "meter": type(meter).__name__,
                    "prepare": prep,
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            touched = set()
            for n, (b, ds, task) in enumerate(jobs, 1):
                out_dir, records_path = files[(b, ds)]
                out_dir.mkdir(parents=True, exist_ok=True)
                record = run_task(args, ctx, meter, b, task, out_dir)
                if record is not None:
                    record["clock_offset_sec"], record["clock_rtt_sec"] = clock
                    with records_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    touched.add((b, ds))
                    e = record["energy"]
                    log.info(
                        "[%s %d/%d] %s %s %s latency=%.1fs energy=%s J tokens=%s%s",
                        model, n, len(jobs), budget_label(b), task.task_id, record["status"],
                        record["latency_sec"],
                        "-" if e["window_basis_J"] is None else f"{e['window_basis_J']:.1f}",
                        record["completion_tokens"],
                        "" if "correct" not in record else f" correct={record['correct']}",
                    )  # fmt: skip
                time.sleep(args.idle_sec)  # idling

            for b, ds in sorted(touched, key=lambda k: (math.inf if k[0] is None else k[0], k[1])):
                out_dir, records_path = files[(b, ds)]
                summary_path = out_dir / f"{ds}_{safe_name(model)}_summary.json"
                log.info("Summary: %s", write_summary(records_path, summary_path, model, ds, b))

            if args.keep_model:
                log.info("Keeping %s on RPi", model)
            else:
                log.info("Finish: %s", rpi.post("/finish", {"model": model}))
    finally:
        meter.close()


if __name__ == "__main__":
    main()
