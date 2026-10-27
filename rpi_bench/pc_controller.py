#!/usr/bin/env python3
"""PC 側コントローラ.

for each 対象モデル:
    model 指定      -> RPi: Ollama 準備 (モデルダウンロード、起動)
    動作確認        -> RPi: Ollama 実行
    for each ベンチマーク問題:
        エネルギー計測開始 (fnirsi-usb-power-data-logger)
        プロンプトを指示 -> RPi: Ollama 実行
        結果受信 (1 つの JSON)  <- RPi: 実行結果
        エネルギー計測終了
        エネルギー生データを圧縮保存
        エネルギー情報を JSON に追加
        jsonl ファイルに記録
        idling
    終了指示        -> RPi: モデル削除

--ssh を指定すると RPi 側の rpi_server.py を SSH で起動し、SSH ポートフォワード経由で通信する。

Python 3.9 互換.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from bench_tasks import DATASETS, Task, load_tasks, score
from power_meter import FnirsiPowerMeter, NullPowerMeter, save_raw, summarize

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = [
    "tinyllama:1.1b",
    "qwen2.5:1.5b",
    "phi:2.7b",
    "phi3:3.8b",
    "gemma:2b",
    "mistral:7b",
]

log = logging.getLogger("pc_controller")


def _int_or_none(v: str) -> Optional[int]:
    return None if v.lower() in ("none", "") else int(v)


def safe_name(name: str) -> str:
    # Windows のファイル名に ':' は使えない (NTFS の代替データストリーム扱いになる)
    return name.replace("/", "_").replace(":", "_")


class RPiClient:
    def __init__(self, base_url: str, timeout_sec: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def _parse(self, path: str, r: requests.Response) -> dict:
        try:
            data = r.json()
        except ValueError:
            data = {"ok": False, "error": r.text}
        if not r.ok or not data.get("ok"):
            raise RuntimeError(f"RPi {path} failed (HTTP {r.status_code}): {data.get('error')}")
        data.pop("ok", None)
        return data

    def get(self, path: str) -> dict:
        return self._parse(path, requests.get(self.base_url + path, timeout=10))

    def post(self, path: str, payload: dict) -> dict:
        return self._parse(path, requests.post(self.base_url + path, json=payload, timeout=self.timeout_sec))

    def clock_offset(self, n: int = 10) -> tuple:
        """(RPi 時刻 - PC 時刻, 往復時間) を最小 RTT のサンプルから推定する."""
        best = None
        for _ in range(n):
            t0 = time.time()
            rpi_t = self.get("/time")["time"]
            t1 = time.time()
            rtt = t1 - t0
            if best is None or rtt < best[1]:
                best = (rpi_t - (t0 + t1) / 2, rtt)
        return best


class SSHServerLauncher:
    """SSH で RPi 上の rpi_server.py を起動し、ポートフォワードで PC から使えるようにする.

    ssh は -tt で疑似端末を確保するので、ssh を終了すると RPi 側サーバも SIGHUP で終了する。
    """

    def __init__(
        self,
        target: str,
        remote_dir: str,
        python: str,
        remote_port: int,
        local_port: int,
        log_path: Path,
        extra_forwards: Optional[list] = None,
    ):
        self.target = target
        # 追加のポートフォワード [(PC 側ポート, RPi 側ポート), ...] (budget_controller.py が Ollama 用に使う)
        self.extra_forwards = extra_forwards or []
        self.remote_dir = remote_dir
        self.python = python
        self.remote_port = remote_port
        self.local_port = local_port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def sync(self) -> None:
        """RPi 側サーバが使うファイルを scp で送る."""
        subprocess.run(["ssh", self.target, f"mkdir -p {self.remote_dir}/rpi_bench"], check=True)
        names = ["main.py", "ollama_utils.py", "system_monitor.py", "rpi_bench/rpi_server.py"]
        subprocess.run(
            ["scp", "-q", *[str(REPO_ROOT / n) for n in names[:-1]], f"{self.target}:{self.remote_dir}/"], check=True
        )
        subprocess.run(["scp", "-q", str(REPO_ROOT / names[-1]), f"{self.target}:{self.remote_dir}/rpi_bench/"], check=True)
        log.info("Synced %s to %s:%s", ", ".join(names), self.target, self.remote_dir)

    def start(self, timeout_sec: float = 60.0) -> None:
        remote_cmd = (
            f"cd {self.remote_dir} && exec {self.python} rpi_bench/rpi_server.py "
            f"--host 127.0.0.1 --port {self.remote_port}"
        )
        cmd = [
            "ssh", "-tt",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L", f"{self.local_port}:127.0.0.1:{self.remote_port}",
            *[a for lp, rp in self.extra_forwards for a in ("-L", f"{lp}:127.0.0.1:{rp}")],
            self.target, remote_cmd,
        ]  # fmt: skip
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Starting RPi server via SSH (log: %s)", self.log_path)
        self._logf = self.log_path.open("ab")
        log_offset = self._logf.tell()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self._logf, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc.poll() is not None:
                with self.log_path.open("rb") as f:
                    f.seek(log_offset)
                    output = f.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ssh exited with code {self.proc.returncode}:\n{output}")
            try:
                requests.get(self.url + "/health", timeout=2)
                return
            except requests.RequestException:
                time.sleep(1)
        self.stop()
        raise RuntimeError(f"RPi server did not become ready; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            log.info("Stopping RPi server (closing SSH session)")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["task_id"] for line in f if line.strip()}


def write_summary(records_path: Path, summary_path: Path, model: str, dataset: str) -> dict:
    with records_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    energies = [r["energy"]["window"]["energy_J"] for r in records if r["energy"]["window"]["energy_J"] is not None]
    summary = {
        "model_name": model,
        "dataset": dataset,
        "num_samples": len(records),
        "total_energy_J": sum(energies) if energies else None,
        "mean_energy_J": sum(energies) / len(energies) if energies else None,
        "mean_latency_sec": sum(r["latency_sec"] for r in records) / len(records) if records else None,
    }
    if records and "correct" in records[0]:
        summary["accuracy"] = sum(1 for r in records if r["correct"]) / len(records)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def run_task(args, rpi: RPiClient, meter, model: str, task: Task, clock: tuple, out_dir: Path) -> Optional[dict]:
    """1 問分: 計測開始 -> プロンプト指示 -> 結果受信 -> 計測終了 -> 生データ保存 -> レコード作成."""
    payload = {"model": model, "prompt": task.prompt, "task_id": task.task_id}
    for attempt in range(1, args.max_retries + 1):
        # RPi 側・通信のエラーはリトライし、パワーメータのエラーは例外のまま中断する
        meter.start()
        try:
            result = rpi.post("/generate", payload)
        except Exception as e:
            meter.stop()
            log.warning("[%s] attempt %d failed: %s", task.task_id, attempt, e)
            time.sleep(args.idle_sec)
            continue
        meas = meter.stop()
        if meter.enabled and not meas.samples:
            log.warning("[%s] attempt %d: no power samples, retrying", task.task_id, attempt)
            time.sleep(args.idle_sec)
            continue
        break
    else:
        log.error("[%s] skipped after %d attempts", task.task_id, args.max_retries)
        return None

    offset, rtt = clock
    energy = {
        "meter": "fnirsi" if meter.enabled else None,
        "attempts": attempt,
        # PC がプロンプトを送ってから結果を受け取るまで (通信を含む)
        "window": summarize(meas.samples, meas.t_start, meas.t_end),
        # RPi 上の推論区間 (RPi 時刻を時計オフセットで PC 時刻に変換)
        "inference": summarize(meas.samples, result["t_start_unix"] - offset, result["t_end_unix"] - offset),
        "clock_offset_sec": offset,
        "clock_rtt_sec": rtt,
        "raw_file": None,
    }
    if meter.enabled:
        raw_path = out_dir / "energy_raw" / f"{safe_name(task.task_id)}.txt.gz"
        save_raw(raw_path, meas.samples)
        energy["raw_file"] = raw_path.relative_to(out_dir).as_posix()

    return {
        "scenario": args.scenario,
        "dataset": task.dataset,
        **result,
        "task_id": task.task_id,
        **task.meta,
        **score(task, result["completion"]),
        "pc_start_time": meas.t_start,
        "pc_end_time": meas.t_end,
        "energy": energy,
    }


def main():
    parser = argparse.ArgumentParser(description="PC side controller for Raspberry Pi Ollama energy benchmarks.")
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--ssh", help="SSH target (e.g. pi@raspberrypi.local); starts rpi_server.py over SSH.")
    conn.add_argument("--rpi_url", help="Already running rpi_server.py (e.g. http://192.168.0.10:8000).")
    parser.add_argument("--ssh_remote_dir", default="~/llm_ptme", help="Repository directory on RPi.")
    parser.add_argument("--ssh_python", default="python3", help="Python on RPi (e.g. ~/venv/bin/python).")
    parser.add_argument("--ssh_sync", action="store_true", help="Copy rpi_server.py and its imports by scp.")
    parser.add_argument("--rpi_port", type=int, default=8000, help="rpi_server.py port on RPi (--ssh).")
    parser.add_argument("--local_port", type=int, default=18000, help="Local forwarded port on PC (--ssh).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--scenario", default="RPi", help="Label used in the output path.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results_rpi"))
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of problems per dataset.")
    # 以下 5 つは main.py と同じ意味・既定値
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--num_thread", type=_int_or_none, default=None, help="Default: Ollama's choice.")
    parser.add_argument("--num_gpu", type=_int_or_none, default=0, help="0 = CPU only, 'none' = Ollama's choice.")
    parser.add_argument("--think", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--keep_alive", default="60m")
    parser.add_argument("--idle_sec", type=float, default=5.0, help="Idle time after each problem.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=float, default=3600.0, help="Timeout [s] for RPi requests.")
    parser.add_argument("--keep_model", action="store_true", help="Do not delete the model on RPi after a model.")
    parser.add_argument("--fnirsi_dir", default=str(REPO_ROOT.parent / "fnirsi-usb-power-data-logger"))
    parser.add_argument("--crc", action="store_true", help="Enable CRC check of power meter packets.")
    parser.add_argument("--no_meter", action="store_true", help="Run without the power meter (protocol test).")
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    launcher = None
    if args.ssh:
        log_path = Path(args.output_dir) / args.scenario / "rpi_server.log"
        launcher = SSHServerLauncher(
            args.ssh, args.ssh_remote_dir, args.ssh_python, args.rpi_port, args.local_port, log_path
        )
        if args.ssh_sync:
            launcher.sync()
        launcher.start()
        args.rpi_url = launcher.url
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

    meter = NullPowerMeter() if args.no_meter else FnirsiPowerMeter(args.fnirsi_dir, use_crc=args.crc)
    meter.open()
    try:
        for model in args.models:
            # 再開対応: 記録済みの問題は飛ばす
            plan = []
            for ds, tasks in tasks_by_dataset.items():
                out_dir = Path(args.output_dir) / args.scenario / ds / safe_name(model)
                records_path = out_dir / f"{ds}_{safe_name(model)}_records.jsonl"
                done = load_done_ids(records_path)
                pending = [t for t in tasks if t.task_id not in done]
                if done:
                    log.info("[%s/%s] resuming: %d done, %d pending", model, ds, len(done), len(pending))
                plan.append((ds, out_dir, records_path, pending))
            if not any(pending for _, _, _, pending in plan):
                log.info("[%s] all problems already done, skipping", model)
                continue

            log.info("========== %s ==========", model)
            log.info("Prepare: %s", rpi.post("/prepare", {"model": model, **prepare_params}))
            log.info("Check: %s", rpi.post("/check", {"model": model}))
            clock = rpi.clock_offset()
            log.info("Clock offset (RPi - PC): %.4f s (RTT %.4f s)", *clock)

            for ds, out_dir, records_path, pending in plan:
                out_dir.mkdir(parents=True, exist_ok=True)
                for n, task in enumerate(pending, 1):
                    record = run_task(args, rpi, meter, model, task, clock, out_dir)
                    if record is not None:
                        with records_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        e = record["energy"]["window"]
                        log.info(
                            "[%s %d/%d] %s latency=%.1fs energy=%s J",
                            model, n, len(pending), task.task_id, record["latency_sec"],
                            "-" if e["energy_J"] is None else f"{e['energy_J']:.2f}",
                        )
                    time.sleep(args.idle_sec)  # idling
                if records_path.exists():
                    summary_path = out_dir / f"{ds}_{safe_name(model)}_summary.json"
                    log.info("Summary: %s", write_summary(records_path, summary_path, model, ds))

            if args.keep_model:
                log.info("Keeping %s on RPi", model)
            else:
                log.info("Finish: %s", rpi.post("/finish", {"model": model}))
    finally:
        meter.close()


if __name__ == "__main__":
    main()
