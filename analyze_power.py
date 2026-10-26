"""Explain the RPi vs envelope energy gap by the power draw of the two machines.

Energy per token = mean power x time per token, so for each (model, dataset)

    E_rpi / E_env = (P_rpi / P_env) x (t_rpi / t_env)

The envelope power (RAPL package) is fitted against the number of active P-cores,
P_env = a + b * cores, and the break-even baseline B (the part of P_env that would have to be
discounted for the RPi to lose its energy advantage) is reported per envelope.

    power_breakdown.csv, figures/fig_power_breakdown.{pdf,png}

Usage:
    python compare_rpi_envelopes.py
    python analyze_power.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_rpi_envelopes as prp

# Envelope settings (Harada et al., arXiv:2607.20806, Table II): active P-cores and CPU frequency cap [GHz]
ENVELOPE_SPECS = {"E1": (1, 1.5), "E2": (2, 1.8), "E3": (4, 2.0), "E4": (6, 2.2), "E5": (8, 2.5)}
RPI = "#eb6834"  # categorical slot 2, distinct from the envelope blue ramp


def per_unit(tasks: pd.DataFrame) -> pd.DataFrame:
    t = tasks[tasks["tokens"].notna()]
    g = (
        t.groupby(["env", "model", "dataset"])
        .agg(energy=("energy_J", "sum"), energy_total=("energy_total_J", "sum"), time=("time_s", "sum"),
             tokens=("tokens", "sum"))
        .reset_index()
    )  # fmt: skip
    g["power_W"] = g["energy"] / g["time"]
    g["power_total_W"] = g["energy_total"] / g["time"]
    g["time_per_token"] = g["time"] / g["tokens"]
    g["energy_per_token"] = g["energy"] / g["tokens"]
    return g


def breakdown(g: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    keys = ["model", "dataset"]
    rpi = g[g["env"] == "RPi"].set_index(keys)
    rows = []
    for env, (cores, freq) in ENVELOPE_SPECS.items():
        e = g[g["env"] == env].set_index(keys)
        power_ratio = (rpi["power_W"] / e["power_W"]).median()
        time_ratio = (rpi["time_per_token"] / e["time_per_token"]).median()
        rows.append(
            {
                "envelope": env, "p_cores": cores, "freq_GHz": freq,
                "env_power_W": e["power_W"].median(), "env_power_min_W": e["power_W"].min(),
                "env_power_max_W": e["power_W"].max(),
                "rpi_power_W": rpi["power_W"].median(), "rpi_power_total_W": rpi["power_total_W"].median(),
                "power_ratio": power_ratio, "time_ratio": time_ratio,
                "energy_ratio": (rpi["energy_per_token"] / e["energy_per_token"]).median(),
                "env_tokens_per_s": 1 / e["time_per_token"].median(),
                "break_even_baseline_W": e["power_W"].median() - rpi["power_W"].median() * time_ratio,
            }
        )  # fmt: skip
    df = pd.DataFrame(rows)
    fit = np.polyfit(df["p_cores"], df["env_power_W"], 1)  # [slope, intercept]
    df["env_power_fit_W"] = np.polyval(fit, df["p_cores"])
    # 当てはめの切片 (コア数に依らない分) を envelope 側から除いた場合のエネルギー比
    df["energy_ratio_wo_intercept"] = (
        df["rpi_power_W"] / (df["env_power_W"] - fit[1]) * df["time_ratio"]
    )
    df["rpi_tokens_per_s"] = 1 / g[g["env"] == "RPi"]["time_per_token"].median()
    return df, fit


def make_figure(df: pd.DataFrame, fit: np.ndarray, stem: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) power vs active P-cores
    xs = np.array([0, 8.5])
    ax1.plot(xs, np.polyval(fit, xs), color=prp.FIT, lw=1.2, ls=(0, (4, 2)), zorder=2)
    for _, r in df.iterrows():
        env = r["envelope"]
        ax1.errorbar(
            r["p_cores"], r["env_power_W"],
            yerr=[[r["env_power_W"] - r["env_power_min_W"]], [r["env_power_max_W"] - r["env_power_W"]]],
            fmt=prp.ENV_MARKERS[env], ms=6, color=prp.ENV_COLORS[env], mec="white", mew=0.6,
            elinewidth=0.8, capsize=2, zorder=3,
        )  # fmt: skip
        ax1.annotate(env, (r["p_cores"], r["env_power_W"]), xytext=(6, -3), textcoords="offset points",
                     fontsize=6, color=prp.INK_SECONDARY)  # fmt: skip
    rpi_total, rpi_dyn = df["rpi_power_total_W"].iloc[0], df["rpi_power_W"].iloc[0]
    ax1.axhline(rpi_total, color=RPI, lw=1.2, zorder=2)
    ax1.axhline(rpi_dyn, color=RPI, lw=1.2, ls=(0, (1, 1.5)), zorder=2)
    ax1.text(8.4, rpi_total + 0.8, f"RPi total {rpi_total:.1f} W", ha="right", va="bottom", fontsize=6,
             color=prp.INK)  # fmt: skip
    ax1.text(8.4, rpi_dyn - 0.8, f"RPi minus idle {rpi_dyn:.1f} W", ha="right", va="top", fontsize=6,
             color=prp.INK)  # fmt: skip
    # 式は直線の右下 (点やエラーバーのない場所) に置く
    ax1.text(4.6, np.polyval(fit, 4.6) - 9, f"fit: {fit[1]:.1f} W + {fit[0]:.2f} W × cores", fontsize=6,
             color=prp.INK)  # fmt: skip
    ax1.set_xlim(0, 8.5)
    ax1.set_ylim(0, df["env_power_max_W"].max() * 1.1)
    ax1.set_xlabel("Active P-cores (envelope)")
    ax1.set_ylabel("Mean power during inference [W]")
    ax1.set_title("(a) Power draw", loc="left", color=prp.INK)
    prp.style_axes(ax1)

    # (b) energy ratio = power ratio x time ratio
    x = np.arange(len(df))
    w = 0.26
    bars = [
        ("power_ratio", "Power ratio", "#86b6ef"),
        ("time_ratio", "Time ratio", "#2a78d6"),
        ("energy_ratio", "Energy ratio", "#104281"),
    ]
    for k, (col, label, color) in enumerate(bars):
        ax2.bar(x + (k - 1) * w, df[col], width=w - 0.03, color=color, label=label, zorder=2)
        for xi, v in zip(x + (k - 1) * w, df[col]):
            ax2.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=5, color=prp.INK)
    ax2.axhline(1.0, color=prp.IDENTITY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax2.set_xticks(x, df["envelope"])
    ax2.set_ylim(0, df["time_ratio"].max() * 1.15)
    ax2.set_ylabel("RPi / envelope (median)")
    ax2.set_title("(b) Energy ratio = power ratio × time ratio", loc="left", color=prp.INK)
    ax2.legend(frameon=False, fontsize=6, loc="upper left")
    prp.style_axes(ax2)

    handles = [
        Line2D([], [], color=prp.FIT, lw=1.2, ls=(0, (4, 2)), label="Linear fit (envelopes)"),
        Line2D([], [], color=RPI, lw=1.2, label="RPi"),
    ]
    ax1.legend(handles=handles, frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    prp.save(fig, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in_dir", type=Path, default=Path("results_analysis/rpi_vs_envelopes"))
    args = parser.parse_args()
    tasks = pd.read_csv(args.in_dir / "task_metrics.csv", dtype={"task": str})
    df, fit = breakdown(per_unit(tasks))
    df.to_csv(args.in_dir / "power_breakdown.csv", index=False)
    make_figure(df, fit, args.in_dir / "figures" / "fig_power_breakdown")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(f"\nenvelope power fit: {fit[1]:.2f} W + {fit[0]:.3f} W x P-cores")


if __name__ == "__main__":
    main()
