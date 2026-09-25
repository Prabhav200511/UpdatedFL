"""Static result figures (matplotlib PNG).

Conventions: categorical colours are assigned in a fixed slot order (never
cycled), each panel has exactly one y-axis (different measures get their own
small-multiple panel), 2 px lines, >= 8 px markers, a recessive grid, and a
legend whenever a panel shows two or more series.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel("Round", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _line(ax, x, y, slot: int, label: str) -> None:
    ax.plot(x, y, color=SLOTS[slot], linewidth=2, marker="o", markersize=4, label=label)


def _stack(ax, frame: pd.DataFrame, x) -> None:
    bottom = None
    for i, column in enumerate(frame.columns):
        values = frame[column].to_numpy()
        ax.bar(x, values, bottom=bottom, color=SLOTS[i % len(SLOTS)], label=column,
               edgecolor=SURFACE, linewidth=1.5, width=0.7)
        bottom = values if bottom is None else bottom + values


def _expand(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([json.loads(v) if isinstance(v, str) else {} for v in series]).fillna(0)


def _legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        # Below the plot area so it never covers data.
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=min(len(handles), 3))


def plot_run(rounds_csv: Path, rsu_csv: Path, output_dir: Path) -> List[Path]:
    rounds = pd.read_csv(rounds_csv)
    rsu = pd.read_csv(rsu_csv)
    x = rounds["round"]
    out: List[Path] = []

    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), facecolor=SURFACE)
    ax = axes[0, 0]
    _line(ax, x, rounds["test_accuracy"], 0, "Accuracy")
    _line(ax, x, rounds["test_macro_f1"], 1, "Macro-F1")
    ax.set_ylim(0, 1)
    _style(ax, "Global model M on held-out test set", "Score")
    _legend(ax)

    ax = axes[0, 1]
    _line(ax, x, rounds["selected"], 0, "Selected")
    _line(ax, x, rounds["accepted"], 1, "Aggregated")
    _line(ax, x, rounds["dropouts"], 2, "Dropped out")
    _style(ax, "Participation per round", "Vehicles")
    _legend(ax)

    ax = axes[0, 2]
    if len(rsu):
        causes = _expand(rsu["dropouts"])
        causes["round"] = rsu["round"].to_numpy()
        drop_by_round = causes.groupby("round").sum()
    else:
        drop_by_round = pd.DataFrame()
    if drop_by_round.empty or drop_by_round.values.sum() == 0:
        ax.text(0.5, 0.5, "No dropouts", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    else:
        _stack(ax, drop_by_round, drop_by_round.index)
    _style(ax, "Dropout causes (selected vehicles)", "Vehicles")
    _legend(ax)

    ax = axes[1, 0]
    rej = _expand(rounds["auth_rejections"])
    if rej.empty or rej.values.sum() == 0:
        ax.text(0.5, 0.5, "No rejections", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    else:
        _stack(ax, rej, x)
    _style(ax, "Rejected authentication attempts", "Beacons")
    _legend(ax)

    ax = axes[1, 1]
    _line(ax, x, rounds["upload_bytes"] / 1024, 0, "Upload")
    _style(ax, "Model-update upload volume", "KiB")

    ax = axes[1, 2]
    _line(ax, x, rounds["availability_brier"], 0, "Brier score")
    _style(ax, "Availability calibration, Eq. (8) (lower is better)", "Brier score")

    ax = axes[2, 0]
    _line(ax, x, rounds["mean_reward"], 0, "Reward")
    _style(ax, "Mean RSU reward, Eq. (14)", "Reward")

    ax = axes[2, 1]
    actions = pd.DataFrame([json.loads(a) for a in rsu["action"]]) if len(rsu) else pd.DataFrame()
    if not actions.empty:
        actions["round"] = rsu["round"].to_numpy()
        comp = actions.groupby(["round", "compression"]).size().unstack(fill_value=0)
        _stack(ax, comp, comp.index)
    _style(ax, "Compression Omega chosen by RSUs", "RSUs")
    _legend(ax)

    ax = axes[2, 2]
    xi = pd.DataFrame([json.loads(v) for v in rsu["xi"] if isinstance(v, str) and v != "[]"])
    if not xi.empty:
        xi["round"] = rsu.loc[rsu["xi"] != "[]", "round"].to_numpy()
        mean_xi = xi.groupby("round").mean()
        names = ["xi1 (availability)", "xi2 (trust)", "xi3 (utility)", "xi4 (uncertainty)"]
        for i, col in enumerate(mean_xi.columns[:4]):
            _line(ax, mean_xi.index, mean_xi[col], i, names[i])
    _style(ax, "Learned score exponents xi, Eq. (10)", "Mean over RSUs")
    _legend(ax)

    fig.tight_layout()
    path = output_dir / "bcpafl_dashboard.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    out.append(path)
    return out


def plot_comparison(rounds_csvs: Dict[str, Path], output_dir: Path) -> Path:
    frames = {name: pd.read_csv(path) for name, path in rounds_csvs.items()}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    labels = {"pomdp": "BC-PAFL (POMDP)", "random": "Random", "all": "All in range",
              "greedy": "Greedy trust"}
    for i, (name, frame) in enumerate(frames.items()):
        _line(axes[0], frame["round"], frame["test_macro_f1"], i, labels.get(name, name))
        _line(axes[1], frame["round"], frame["dropout_rate"], i, labels.get(name, name))
        _line(axes[2], frame["round"], frame["upload_bytes"].cumsum() / 1024, i,
              labels.get(name, name))
    _style(axes[0], "Global macro-F1", "Macro-F1")
    _style(axes[1], "Dropout rate of selected vehicles", "Share of selected")
    _style(axes[2], "Cumulative upload volume", "KiB")
    for ax in axes:
        _legend(ax)
    fig.tight_layout()
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path
