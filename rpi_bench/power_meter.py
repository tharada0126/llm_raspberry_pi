"""PC 側: FNIRSI USB パワーメータによるエネルギー計測.

../fnirsi-usb-power-data-logger/fnirsi_logger.py のデバイス検出・通信処理を再利用する。
問題ごとにデバイスを開き直すとリセット待ちやフリーズ (同 README の TODO 参照) が起きやすいため、
USB 接続は実験全体で開いたままにし、受信スレッドが常時データを読み続ける。
start() から stop() までに受信したサンプルを 1 回の計測として返す。

エネルギー予算 (budget_controller.py): start(budget_J=...) を指定すると、受信スレッドが
サンプルを受け取るたびにエネルギーを積算し、予算に達した時点で on_budget を呼ぶ
(推論の打ち切りを、次のトークンを待たずに即座に行うため)。

Python 3.9 互換.
"""
from __future__ import annotations

import atexit
import gzip
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional

log = logging.getLogger("power_meter")

SPS = 100  # fnirsi_logger と同じく 100 samples/s 固定
INTERVAL = 1.0 / SPS
KEEPALIVE = b"\xaa\x83" + b"\x00" * 61 + b"\x9e"
RAW_HEADER = "timestamp sample_in_packet voltage_V current_A dp_V dn_V temp_C_ema energy_Ws capacity_As"


class Sample(NamedTuple):
    t: float  # PC の UNIX 時刻 [s]
    idx: int  # パケット内のサンプル番号 (0-3)
    voltage: float
    current: float
    dp: float
    dn: float
    temp: float


@dataclass
class Measurement:
    t_start: float
    t_end: float
    samples: List[Sample] = field(default_factory=list)
    # start(budget_J=...) で予算に達した場合: {"t_sample", "t_fired", "energy_J"} (それ以外は None)
    budget_hit: Optional[dict] = None


class _BudgetAccumulator:
    """start() から stop() までのエネルギーを受信のたびに積算し、予算到達で on_budget を呼ぶ.

    サブクラスは __init__ で _init_accumulator() を呼び、サンプル受信時に _feed() を呼ぶ。
    積算は fnirsi_logger の energy_Ws と同じく sum((V*I - baseline_W) * dt)。
    """

    def _init_accumulator(self) -> None:
        self._lock = threading.Lock()
        self._recording = False
        self._buf: List[Sample] = []
        self._t_start = 0.0
        self._last_t: Optional[float] = None
        self._budget_J: Optional[float] = None
        self._baseline_W = 0.0
        self._on_budget = None
        self._acc_J = 0.0
        self._budget_hit: Optional[dict] = None

    @property
    def energy_J(self) -> float:
        """start() からの積算エネルギー [J] (baseline_W を差し引いた値)."""
        with self._lock:
            return self._acc_J

    def _begin(self, budget_J: Optional[float], baseline_W: float, on_budget) -> float:
        with self._lock:
            self._buf = []
            self._t_start = time.time()
            self._recording = True
            self._budget_J = budget_J
            self._baseline_W = baseline_W
            self._on_budget = on_budget
            self._acc_J = 0.0
            self._budget_hit = None
        return self._t_start

    def disarm(self) -> None:
        """予算の監視だけを止める (計測は stop() まで続く). 生成終了後の tail 区間で on_budget を呼ばないため."""
        with self._lock:
            self._budget_J = None
            self._on_budget = None

    def _feed(self, samples: List[Sample]) -> None:
        fire = None
        with self._lock:
            if self._recording:
                self._buf.extend(samples)
                for s in samples:
                    if s.t < self._t_start:  # start() 直後に届いたパケットの、start() より前のサンプル
                        continue
                    self._acc_J += (s.voltage * s.current - self._baseline_W) * INTERVAL
                    if self._budget_J is not None and self._budget_hit is None and self._acc_J >= self._budget_J:
                        self._budget_hit = {"t_sample": s.t, "t_fired": time.time(), "energy_J": self._acc_J}
                        fire = (self._on_budget, dict(self._budget_hit))
            self._last_t = samples[-1].t
        if fire is not None and fire[0] is not None:
            try:
                fire[0](fire[1])  # ロックの外で呼ぶ (打ち切り処理は短時間で終わること)
            except Exception as e:
                log.warning("on_budget callback failed: %s", e)

    def _end(self, t_end: float) -> Measurement:
        with self._lock:
            self._recording = False
            samples, self._buf = self._buf, []
            t_start = self._t_start
            hit, self._budget_hit = self._budget_hit, None
            self._on_budget = None
            self._budget_J = None
        return Measurement(t_start, t_end, [s for s in samples if t_start <= s.t <= t_end], hit)


def _load_libusb() -> None:
    """libusb-package が入っていれば、同梱の libusb-1.0 (dll/dylib/so) を pyusb に読み込ませる.

    pyusb は一度読み込んだ libusb を使い回すので、以降の usb.core.find() (fnirsi_logger 内) もこれを使う。
    Windows / macOS では libusb を別途インストールしなくて済む。
    """
    try:
        import libusb_package
    except ImportError:
        return
    libusb_package.get_libusb1_backend()


class _HidEndpoint:
    """hidapi のデバイスを pyusb のエンドポイントと同じ write() / read() で使えるようにする."""

    def __init__(self, dev):
        self._dev = dev

    def write(self, data: bytes) -> int:
        # 先頭 1 バイトはレポート ID (このデバイスはレポート ID なしなので 0)
        return self._dev.write(b"\x00" + bytes(data))

    def read(self, size_or_buffer: int, timeout: int):
        import usb.core

        data = self._dev.read(size_or_buffer, timeout)
        if not data:
            raise usb.core.USBTimeoutError("Operation timed out")
        return data


class FnirsiPowerMeter(_BudgetAccumulator):
    enabled = True

    def __init__(self, logger_dir: str, use_crc: bool = False):
        self.logger_dir = Path(logger_dir).resolve()
        self.use_crc = use_crc
        self._init_accumulator()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._crc = None

    def open(self) -> None:
        sys.path.insert(0, str(self.logger_dir))
        import fnirsi_logger as fl

        if self.use_crc:
            try:
                self._crc = fl.setup_crc()
            except Exception as e:
                log.warning("CRC check disabled: %s", e)

        if sys.platform == "darwin":
            self._ep_out = self._ep_in = self._open_hid(fl)
        else:
            self._ep_out, self._ep_in = self._open_usb(fl)

        fl.request_data(self._is_fnb58_or_fnb48s, self._ep_out)
        time.sleep(0.1)

        self._thread = threading.Thread(target=self._run, name="fnirsi-reader", daemon=True)
        self._thread.start()
        # 読み込み中にプロセスが終了するとメータが固まるので、必ず close() (FIFO の読み切り) を通す
        atexit.register(self.close)

        deadline = time.time() + 5.0
        while self._last_t is None:
            self._check_error()
            if time.time() > deadline:
                raise RuntimeError("No data received from FNIRSI power meter.")
            time.sleep(0.05)
        log.info("FNIRSI power meter ready (%s)", "FNB58/FNB48S" if self._is_fnb58_or_fnb48s else "FNB48/C1")

    def _open_usb(self, fl):
        """Windows / Linux: pyusb (libusb) で HID インターフェースのエンドポイントを直接使う."""
        _load_libusb()
        import usb.util

        dev, self._is_fnb58_or_fnb48s = fl.find_device()
        if dev is None:
            raise RuntimeError("FNIRSI power meter not found (check USB connection / WinUSB driver).")
        dev.reset()
        interface_hid_num = fl.find_hid_interface_num(dev)
        dev.set_configuration()
        intf = dev.get_active_configuration()[(interface_hid_num, 0)]
        ep_out = usb.util.find_descriptor(
            intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        )
        ep_in = usb.util.find_descriptor(
            intf, custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )
        assert ep_in and ep_out
        return ep_out, ep_in

    def _open_hid(self, fl) -> "_HidEndpoint":
        """macOS: カーネルの HID ドライバがインターフェースを掴んでいて libusb から claim できないため hidapi を使う."""
        import hid

        candidates = [
            (fl.VID, fl.PID_FNB48, False),
            (fl.VID, fl.PID_C1, False),
            (fl.VID_FNB58, fl.PID_FNB58, True),
            (fl.VID_FNB48S, fl.PID_FNB48S, True),
        ]
        for vid, pid, is_fnb58_or_fnb48s in candidates:
            if hid.enumerate(vid, pid):
                dev = hid.device()
                try:
                    dev.open(vid, pid)
                except OSError as e:
                    # macOS の hidapi は排他オープンなので、他プロセスが開いていると失敗する
                    raise RuntimeError(
                        "Could not open FNIRSI power meter. Another process may be using it "
                        "(previous run still alive?); otherwise unplug and replug the meter."
                    ) from e
                self._is_fnb58_or_fnb48s = is_fnb58_or_fnb48s
                return _HidEndpoint(dev)
        raise RuntimeError("FNIRSI power meter not found (check USB connection).")

    def close(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=15)
            self._thread = None

    def start(self, budget_J: Optional[float] = None, baseline_W: float = 0.0, on_budget=None) -> float:
        """計測開始. budget_J を指定すると、積算エネルギーが達した時点で on_budget(hit) を受信スレッドから呼ぶ."""
        self._check_error()
        return self._begin(budget_J, baseline_W, on_budget)

    def stop(self) -> Measurement:
        t_end = time.time()
        # USB の受信遅延分、t_end 以降のサンプルが届くまで少し待つ
        deadline = t_end + 1.0
        while time.time() < deadline:
            with self._lock:
                if self._last_t is not None and self._last_t >= t_end:
                    break
            time.sleep(0.01)
        meas = self._end(t_end)
        self._check_error()
        return meas

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Power meter reader failed: {self._error!r}")

    def _parse(self, data) -> List[Sample]:
        # パケット形式は fnirsi_logger.decode() と同じ
        if data[1] != 0x04:
            return []
        if self._crc is not None and data[-1] != self._crc(bytearray(data[1:-1])):
            log.warning("Ignoring packet with bad checksum")
            return []
        t0 = time.time() - 4 * INTERVAL
        out = []
        for i in range(4):
            o = 2 + 15 * i
            out.append(
                Sample(
                    t=t0 + i * INTERVAL,
                    idx=i,
                    voltage=int.from_bytes(bytes(data[o : o + 4]), "little") / 100000,
                    current=int.from_bytes(bytes(data[o + 4 : o + 8]), "little") / 100000,
                    dp=(data[o + 8] + data[o + 9] * 256) / 1000,
                    dn=(data[o + 10] + data[o + 11] * 256) / 1000,
                    temp=(data[o + 13] + data[o + 14] * 256) / 10.0,
                )
            )
        return out

    def _run(self) -> None:
        import usb.core

        refresh = 1.0 if self._is_fnb58_or_fnb48s else 0.003
        next_keepalive = time.time() + refresh
        try:
            while not self._stop_evt.is_set():
                samples = self._parse(self._ep_in.read(size_or_buffer=64, timeout=5000))
                if samples:
                    self._feed(samples)
                if time.time() >= next_keepalive:
                    next_keepalive = time.time() + refresh
                    self._ep_out.write(KEEPALIVE)
        except BaseException as e:
            self._error = e
        finally:
            # 終了時はメータ内部 FIFO を読み切っておく (fnirsi_logger と同様)
            try:
                while self._ep_in.read(size_or_buffer=64, timeout=1000):
                    pass
            except usb.core.USBTimeoutError:
                pass
            except Exception as e:
                log.warning("Draining failed: %s", e)


class NullPowerMeter:
    """--no_meter 用。計測は行わず、通信フローだけを確認する."""

    enabled = False

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def start(self, budget_J: Optional[float] = None, baseline_W: float = 0.0, on_budget=None) -> float:
        self._t_start = time.time()
        return self._t_start

    def stop(self) -> Measurement:
        return Measurement(self._t_start, time.time())


class SimulatedPowerMeter(_BudgetAccumulator):
    """--simulate_power_W 用。一定電力のサンプルを 100 sps で生成する (メータなしで打ち切り処理を確認する).

    記録されるエネルギーは実測ではないので、本番の計測には使わない。
    """

    enabled = True
    simulated = True

    def __init__(self, power_W: float, voltage_V: float = 5.0):
        self.power_W = power_W
        self.voltage_V = voltage_V
        self._init_accumulator()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self._thread = threading.Thread(target=self._run, name="simulated-meter", daemon=True)
        self._thread.start()
        log.warning("Using SIMULATED power meter (%.2f W constant); energies are not measured.", self.power_W)

    def close(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=5)
            self._thread = None

    def start(self, budget_J: Optional[float] = None, baseline_W: float = 0.0, on_budget=None) -> float:
        return self._begin(budget_J, baseline_W, on_budget)

    def stop(self) -> Measurement:
        t_end = time.time()
        time.sleep(0.05)  # 実機と同様に t_end 以降のパケットを待つ
        return self._end(t_end)

    def _run(self) -> None:
        current = self.power_W / self.voltage_V
        while not self._stop_evt.wait(4 * INTERVAL):  # 実機と同じく 4 サンプル / パケット
            t0 = time.time() - 4 * INTERVAL
            self._feed([Sample(t0 + i * INTERVAL, i, self.voltage_V, current, 0.0, 0.0, 25.0) for i in range(4)])


def summarize(samples: List[Sample], t_start: float, t_end: float) -> dict:
    """[t_start, t_end] 区間のエネルギー情報."""
    sel = [s for s in samples if t_start <= s.t <= t_end]
    duration = t_end - t_start
    info = {
        "t_start": t_start,
        "t_end": t_end,
        "duration_sec": duration,
        "num_samples": len(sel),
        "expected_samples": duration * SPS,
    }
    if not sel:
        info.update(energy_J=None, energy_sum_J=None, mean_power_W=None, max_power_W=None)
        return info
    powers = [s.voltage * s.current for s in sel]
    mean_power = sum(powers) / len(powers)
    info.update(
        energy_J=mean_power * duration,  # 平均電力 x 区間長 (サンプル欠落に強い)
        energy_sum_J=sum(powers) * INTERVAL,  # sum(V*I) * dt (fnirsi_logger の energy_Ws と同じ積算)
        mean_power_W=mean_power,
        max_power_W=max(powers),
        min_power_W=min(powers),
        mean_voltage_V=sum(s.voltage for s in sel) / len(sel),
        mean_current_A=sum(s.current for s in sel) / len(sel),
    )
    return info


def save_raw(path: Path, samples: List[Sample], alpha: float = 0.9) -> None:
    """fnirsi_logger の出力と同じ書式で gzip 圧縮保存する (energy/capacity は区間開始からの積算).

    `./plot.gnuplot "<zcat file.txt.gz"` でそのままプロットできる。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    energy = capacity = 0.0
    temp_ema = None
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n" + RAW_HEADER + "\n")
        for s in samples:
            energy += s.voltage * s.current * INTERVAL
            capacity += s.current * INTERVAL
            temp_ema = s.temp if temp_ema is None else s.temp * (1.0 - alpha) + temp_ema * alpha
            f.write(
                f"{s.t:.3f} {s.idx} {s.voltage:7.5f} {s.current:7.5f} {s.dp:5.3f} {s.dn:5.3f} "
                f"{temp_ema:6.3f} {energy:.6f} {capacity:.6f}\n"
            )
