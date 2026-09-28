"""VANET plot suite for BC-PAFL (ProxyFL v1 style).

Reads ``vanet_metrics.csv`` + ``vanet_routing_rounds.csv`` (+ routing
metadata) and writes the ``vanet_*.png`` figures plus
``vanet_plot_explanations.md`` with measured min/max ranges, mirroring
ProxyFL v1's ``plot_metrics.py`` / ``routing_plots.py``: same figure sizes,
titles, legend-inside-axes-headroom layout and round ticks.

Differences from v1 (documented in the outputs, not hidden):
* accuracy/loss curves come from the metrics CSV, not a training-logs file;
* accuracy/loss cover every vehicle every round (private models always
  train); the per-vehicle cost figures average only the vehicles that trained
  M_i for upload that round (``fl_participant``);
* communication latency is modeled wireless airtime (v2 delivers in-process,
  there is no host TCP thread to time);
* AODV is a modeled overlay on the frozen round-start topology (see
  :mod:`bcpafl.vanet_aodv`), not the live transport.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AODV_REPORT_MARKER = "<!-- AODV_PLOT_EXPLANATIONS -->"


def _legend_above() -> None:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(loc="upper center", bbox_to_anchor=(0.01, 0.98, 0.98, 0.0),
              ncol=min(4, len(handles)), mode="expand" if len(handles) > 1 else None,
              frameon=True, borderaxespad=0.0, columnspacing=1.0, handletextpad=0.5)


def _reserve_legend_headroom(axes, renderer, padding_px=6.0) -> None:
    legend = axes.get_legend()
    if legend is None or axes.get_yscale() != "linear":
        return
    axes_box = axes.get_window_extent(renderer)
    legend_box = legend.get_window_extent(renderer)
    if axes_box.height <= 0:
        return
    clear_fraction = (legend_box.y0 - padding_px - axes_box.y0) / axes_box.height
    if clear_fraction <= 0:
        return
    data_top = axes.dataLim.ymax
    lower, upper = axes.get_ylim()
    if not np.isfinite(data_top) or upper <= lower or data_top <= lower:
        return
    data_fraction = (data_top - lower) / (upper - lower)
    if data_fraction <= clear_fraction:
        return
    axes.set_ylim(lower, lower + (data_top - lower) / clear_fraction)


def _apply_plot_layout() -> None:
    figure = plt.gcf()
    figure.tight_layout()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axes in figure.axes:
        _reserve_legend_headroom(axes, renderer)
    figure.canvas.draw()


def _set_round_ticks(rounds, max_ticks=12) -> None:
    values = sorted({int(r) for r in rounds})
    if not values:
        return
    if len(values) <= max_ticks:
        ticks = values
    else:
        pos = np.linspace(0, len(values) - 1, num=max_ticks, dtype=int)
        ticks = [values[p] for p in pos]
    plt.xticks(ticks)


def _vehicle_frame(df: pd.DataFrame) -> pd.DataFrame:
    vehicle_df = df[df["node"].str.contains(r"^C\d+_[VD]\d+$", regex=True, na=False)]
    if vehicle_df.empty:
        vehicle_df = df[~df["node"].isin(["Server"])
                        & ~df["node"].str.startswith(("Cluster", "RSU_"), na=False)]
    return vehicle_df


def _participant_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vehicle rows of the round's FL exchange, without private-only training rows.

    Every vehicle trains its private model every round, but only vehicles the
    POMDP had train M_i for upload spend the round's FL security and
    communication time; mixing private-only rows into per-vehicle cost means
    would dilute them.
    """
    vehicle_df = _vehicle_frame(df)
    if "fl_participant" not in vehicle_df.columns:
        return vehicle_df
    return vehicle_df[vehicle_df["fl_participant"] != 0.0]


def _series(group: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    if col not in group.columns:
        return pd.Series(fill, index=group.index, dtype=float)
    return group[col].fillna(fill)


def _extrema_summary(series: pd.Series, unit: str = "") -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "No values were recorded for this run."
    suffix = f" {unit}" if unit else ""
    return (f"Minimum: {numeric.loc[numeric.idxmin()]:.3f}{suffix} in round "
            f"{numeric.idxmin()}; maximum: {numeric.loc[numeric.idxmax()]:.3f}{suffix} "
            f"in round {numeric.idxmax()}.")


def _wireless_goodput_by_round(df: pd.DataFrame) -> pd.Series:
    required = {"round", "vanet_wireless_bits", "vanet_airtime_s"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float)
    wireless = df.groupby("round", as_index=True).agg(
        bits=("vanet_wireless_bits", "sum"), airtime=("vanet_airtime_s", "sum"))
    wireless = wireless[wireless.index > 0]
    return (wireless["bits"] / wireless["airtime"].replace(0.0, np.nan)).fillna(0.0)


# ----------------------------------------------------------------------
# Accuracy / loss from the metrics CSV
# ----------------------------------------------------------------------
def _plot_accuracy_loss(df: pd.DataFrame, out_dir: Path, saved: List[Path]) -> None:
    vehicle_df = _vehicle_frame(df)
    markers = ['o', 's', '^', 'v', 'd', 'x', '*', '+']
    # Population view (v1 semantics).  Every vehicle trains its private model
    # every round, so gaps are rare; where one occurs the private weights were
    # unchanged, and carrying the last accuracy forward equals re-evaluating.
    priv = vehicle_df.pivot_table(index="round", columns="node",
                                  values="private_test_accuracy_pct", aggfunc="max")
    priv = priv[priv.index > 0].sort_index().ffill()
    server = df[df["node"] == "Server"].set_index("round").sort_index()
    if not priv.empty:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(sorted(priv.columns)):
            vals = priv[col].dropna()
            plt.plot(vals.index, vals.values, label=f"Private: {col}", linestyle='--',
                     alpha=0.5, marker=markers[i % len(markers)])
        mean_priv = priv.mean(axis=1, skipna=True).dropna()
        if not mean_priv.empty:
            plt.plot(mean_priv.index, mean_priv.values, label="Mean Private Model Accuracy",
                     color="darkgreen", linewidth=3.0, marker="D")
        if "global_proxy_accuracy_pct" in server and server[
                "global_proxy_accuracy_pct"].notna().any():
            g = server["global_proxy_accuracy_pct"].dropna()
            g = g[g.index > 0]
            if not g.empty:
                plt.plot(g.index, g.values, label="Global Proxy Model Accuracy",
                         color="royalblue", linewidth=3.0, marker="o")
        plt.title("Accuracy vs. Communication Rounds (VANET)", fontsize=14, fontweight='bold')
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        _legend_above()
        _set_round_ticks(priv.index)
        _apply_plot_layout()
        path = out_dir / "vanet_accuracy_vs_rounds.png"
        plt.savefig(path, dpi=130)
        plt.close()
        saved.append(path)

    # A training loss is only measured while training, so it is never carried
    # forward (that drew flat "staircase" segments for idle rounds).
    loss = vehicle_df.pivot_table(index="round", columns="node", values="train_loss",
                                  aggfunc="max")
    loss = loss[loss.index > 0].sort_index()
    if not loss.empty:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(sorted(loss.columns)):
            vals = loss[col].dropna()
            plt.plot(vals.index, vals.values, label=f"{col} Training Loss",
                     linestyle='--', alpha=0.5, marker=markers[i % len(markers)])
        mean_loss = loss.mean(axis=1, skipna=True).dropna()
        if not mean_loss.empty:
            plt.plot(mean_loss.index, mean_loss.values, label="Mean Training Loss",
                     color="crimson", linewidth=3.0, marker="s")
        plt.title("Training Loss vs. Communication Rounds (VANET)", fontsize=14,
                  fontweight='bold')
        plt.xlabel("Communication Round", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        _legend_above()
        _set_round_ticks(loss.index)
        _apply_plot_layout()
        path = out_dir / "vanet_loss_vs_rounds.png"
        plt.savefig(path, dpi=130)
        plt.close()
        saved.append(path)


# ----------------------------------------------------------------------
# Component plots from per-round vehicle means
# ----------------------------------------------------------------------
def _plot_from_csv(df: pd.DataFrame, out_dir: Path, saved: List[Path]) -> None:
    vehicle_df = _participant_frame(df)
    if vehicle_df.empty:
        return
    r_group = vehicle_df.groupby("round", as_index=True).mean(numeric_only=True)
    r_group = r_group[r_group.index > 0]
    if r_group.empty:
        return
    r_indices = np.array(r_group.index)
    title = " (VANET)"

    def _fig(name: str) -> Path:
        _legend_above()
        _set_round_ticks(r_indices)
        plt.grid(True, linestyle="--", alpha=0.5)
        _apply_plot_layout()
        path = out_dir / name
        plt.savefig(path, dpi=130)
        plt.close()
        saved.append(path)
        return path

    plt.figure(figsize=(10, 6))
    plt.plot(r_indices, _series(r_group, "energy_training_j").values, marker='d',
             linewidth=2.5, label="E_training (DML+DP Compute)", color="#FF6F91")
    plt.title(f"Training Energy vs. Communication Rounds{title}", fontsize=13,
              fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _fig("vanet_energy_training_vs_rounds.png")

    sec_energy = _series(r_group, "energy_security_j").values
    comm_energy = _series(r_group, "energy_communication_j").values
    tot_energy = _series(r_group, "energy_total_j").values
    train_energy = _series(r_group, "energy_training_j").values
    plt.figure(figsize=(10, 6))
    plt.plot(r_indices, sec_energy, marker='o', linewidth=2.5,
             label="E_security (Keygen + Sig + Verify + Encrypt)", color="#845EC2")
    plt.plot(r_indices, comm_energy, marker='s', linewidth=2.5,
             label="E_communication (TX + RX)", color="#00C9A7")
    plt.plot(r_indices, tot_energy, marker='^', linewidth=3.0,
             label="E_total = E_security + E_comm", color="#D65DB1")
    plt.title(f"Non-Training Energy vs. Communication Rounds{title}", fontsize=13,
              fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _fig("vanet_energy_other_vs_rounds.png")

    plt.figure(figsize=(10, 6))
    plt.plot(r_indices, sec_energy, marker='o', linewidth=2.5, label="E_security",
             color="#845EC2")
    plt.plot(r_indices, comm_energy, marker='s', linewidth=2.5, label="E_communication",
             color="#00C9A7")
    plt.plot(r_indices, tot_energy, marker='^', linewidth=3.0, label="E_total",
             color="#D65DB1")
    plt.plot(r_indices, train_energy, marker='d', linewidth=2.0, linestyle='--',
             label="E_training", color="#FF6F91", alpha=0.7)
    plt.title(f"Per-Vehicle Energy Consumption vs. Communication Rounds{title}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Energy (Joules)", fontsize=11)
    _fig("vanet_energy_breakdown.png")

    sec_e2e = _series(r_group, "security_latency_ms").values
    comm_e2e = _series(r_group, "communication_latency_ms").values
    train_e2e = _series(r_group, "training_ms").values
    idle_e2e = _series(r_group, "idle_latency_ms").values
    total_e2e = (_series(r_group, "end_to_end_time_ms").values
                 if "end_to_end_time_ms" in r_group.columns
                 else sec_e2e + comm_e2e + train_e2e + idle_e2e)
    for name in ("vanet_end_to_end_time_vs_rounds.png", "vanet_latency_breakdown.png"):
        plt.figure(figsize=(11, 6))
        plt.plot(r_indices, total_e2e, marker='o', linewidth=3.0,
                 label="End-to-End Time (total)", color="#2C73D2")
        plt.plot(r_indices, train_e2e, marker='d', linewidth=2.0, label="Training",
                 color="#FF9671")
        plt.plot(r_indices, idle_e2e, marker='v', linewidth=2.0, linestyle='--',
                 label="Idle / Sync Wait", color="#C4A484")
        plt.plot(r_indices, sec_e2e, marker='s', linewidth=2.0, label="Security (Crypto)",
                 color="#845EC2")
        plt.plot(r_indices, comm_e2e, marker='^', linewidth=2.0,
                 label="Communication (TX/RX)", color="#00C9A7")
        plt.title(f"End-to-End Time vs. Communication Rounds{title}", fontsize=13,
                  fontweight='bold')
        plt.xlabel("Communication Round", fontsize=11)
        plt.ylabel("End-to-End Time (ms)", fontsize=11)
        _fig(name)

    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, sec_e2e, marker='s', linewidth=2.5, label="Security (Crypto)",
             color="#845EC2")
    plt.plot(r_indices, comm_e2e, marker='^', linewidth=2.5, label="Communication (TX/RX)",
             color="#00C9A7")
    plt.title(f"End-to-End Time — Security & Communication Detail{title}", fontsize=13,
              fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("End-to-End Time (ms)", fontsize=11)
    _fig("vanet_end_to_end_security_comm_detail.png")

    if "action_to_response_ms" in vehicle_df.columns and vehicle_df[
            "action_to_response_ms"].notna().any():
        a2r = _series(r_group, "action_to_response_ms").values
        a2r_label = "Action-to-Response Latency"
    else:
        a2r = np.maximum(_series(r_group, "device_round_execution_ms").values - train_e2e,
                         0.0)
        a2r_label = "Action-to-Response Latency (approx: round − training)"
    plt.figure(figsize=(10, 5))
    plt.plot(r_indices, a2r, marker='o', linewidth=2.5, color="#B83227", label=a2r_label)
    plt.title(f"Latency (Action-to-Response) vs. Communication Rounds{title}", fontsize=13,
              fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Latency (ms)", fontsize=11)
    _fig("vanet_action_to_response_latency.png")

    sig_gen = _series(r_group, "signature_generation_ms").values
    batch_ver = _series(r_group, "batch_verification_ms").values
    verification_total = sig_gen * 0 + _series(
        r_group, "signature_verification_ms").values + batch_ver
    enc_ms = _series(r_group, "encryption_ms").values
    for name in ("vanet_cryptographic_operations.png", "vanet_security_overhead.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(r_indices, sig_gen, marker='o', label="Signature Generation",
                 color="#4D8076", linewidth=2)
        plt.plot(r_indices, batch_ver, marker='s', label="Batch Verification",
                 color="#845EC2", linewidth=2)
        plt.plot(r_indices, verification_total, marker='^',
                 label="Signature Verification (total)", color="#C34A36", linestyle=":",
                 linewidth=1.5)
        plt.plot(r_indices, enc_ms, marker='x', label="AES-GCM Encryption",
                 color="#0081CF", linewidth=1.5)
        plt.title(f"Cryptographic Operations{title}", fontsize=13, fontweight='bold')
        plt.xlabel("Communication Round", fontsize=11)
        plt.ylabel("Time (ms)", fontsize=11)
        _fig(name)

    plt.figure(figsize=(10, 5))
    goodput = _wireless_goodput_by_round(df)
    if not goodput.empty:
        rounds = np.array(goodput.index)
        plt.plot(rounds, goodput.values / 1_000_000.0, marker="o", color="#2C73D2",
                 linewidth=2.5, label="Modeled VANET Goodput")
        plt.ylabel("Modeled VANET Goodput (Mbps)", fontsize=11)
        _set_round_ticks(rounds)
    else:
        plt.plot(r_indices, np.zeros_like(r_indices), marker="o", color="#2C73D2",
                 linewidth=2.5, label="Modeled VANET Goodput")
        plt.ylabel("Modeled VANET Goodput (Mbps)", fontsize=11)
    plt.title(f"VANET Link Goodput vs. Rounds{title}", fontsize=13, fontweight='bold')
    plt.xlabel("Communication Round", fontsize=11)
    _fig("vanet_throughput_vs_rounds.png")


def _plot_coverage(df: pd.DataFrame, out_dir: Path, saved: List[Path]) -> Optional[pd.Series]:
    coverage_total = None
    coverage_column = ("vehicles_served" if "vehicles_served" in df
                       and df["vehicles_served"].notna().any() else "vehicles_in_range")
    coverage_rows = df[df["node"].str.startswith("RSU_", na=False)
                       & df.get(coverage_column,
                                pd.Series(index=df.index, dtype=float)).notna()]
    if coverage_rows.empty:
        return None
    coverage = coverage_rows.pivot_table(index="round", columns="node",
                                         values=coverage_column, aggfunc="max").sort_index()
    plt.figure(figsize=(11, 6))
    for rsu_name in coverage.columns:
        plt.plot(coverage.index, coverage[rsu_name], marker="o", linewidth=1.8,
                 label=str(rsu_name).replace("_", " "))
    server_total_column = ("vehicles_served_total" if "vehicles_served_total" in df
                           and df["vehicles_served_total"].notna().any()
                           else "vehicles_in_range_total")
    server_coverage = df[(df["node"] == "Server") & df.get(
        server_total_column, pd.Series(index=df.index, dtype=float)).notna()]
    if not server_coverage.empty:
        coverage_total = server_coverage.set_index("round")[server_total_column].sort_index()
    else:
        coverage_total = coverage.sum(axis=1)
    plt.plot(coverage_total.index, coverage_total.values, marker="D", linewidth=3.2,
             color="black", label="Total served")
    plt.title("Vehicles Served by Closest In-Range RSU vs. Rounds (VANET)", fontsize=13,
              fontweight="bold")
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Number of served vehicles", fontsize=11)
    _set_round_ticks(coverage.index)
    max_count = int(max(coverage_total.max(), coverage.max().max()))
    plt.yticks(range(0, max_count + 2))
    _legend_above()
    plt.grid(True, linestyle="--", alpha=0.5)
    _apply_plot_layout()
    path = out_dir / "vanet_vehicles_in_range_vs_rounds.png"
    plt.savefig(path, dpi=130)
    plt.close()
    saved.append(path)
    return coverage_total


# ----------------------------------------------------------------------
# Routing plots
# ----------------------------------------------------------------------
def _plot_routing(routing_csv: Path, metadata_path: Path, out_dir: Path,
                  saved: List[Path]) -> bool:
    import json as _json
    if not routing_csv.exists() or not metadata_path.exists() or routing_csv.stat(
            ).st_size == 0:
        return False
    metadata = _json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("routing_mode") != "aodv":
        return False
    frame = pd.read_csv(routing_csv).sort_values("round")
    rounds = frame["round"]
    title_suffix = " (VANET)"

    def _figure(ylabel: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.sca(ax)
        _set_round_ticks(rounds)
        margin = max(0.1, (rounds.max() - rounds.min()) * 0.03)
        ax.set_xlim(rounds.min() - margin, rounds.max() + margin)
        return fig, ax

    def _save(fig, name: str, title: str):
        plt.figure(fig.number)
        plt.sca(fig.axes[0])
        fig.axes[0].set_title(title + "\n" + title_suffix.strip(), fontsize=12)
        _legend_above()
        _apply_plot_layout()
        path = out_dir / name
        plt.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)

    fig, ax = _figure("Transmitted volume (kibibytes, KiB)")
    plt.sca(ax)
    total = frame["rreq_bytes_tx"] + frame["rrep_bytes_tx"] + frame["rerr_bytes_tx"]
    for kind, full in (("rreq", "Route Request (RREQ)"), ("rrep", "Route Reply (RREP)"),
                       ("rerr", "Route Error (RERR)")):
        ax.plot(rounds, frame[kind + "_bytes_tx"] / 1024, marker="o",
                label=full + "\n+ headers")
    ax.plot(rounds, total / 1024, marker="s", linewidth=2, label="Total routing")
    _save(fig, "vanet_aodv_routing_overhead_vs_rounds.png",
          "Ad hoc On-Demand Distance Vector (AODV)\n"
          "Routing transmissions, including Internet Protocol / User Datagram Protocol headers")

    fig, ax = _figure("Transmitted volume (kibibytes, KiB)")
    plt.sca(ax)
    for column, label in (("fl_application_bytes_tx", "Federated Learning (FL) /\napplication"),
                          ("security_bytes_tx", "Security increment"),
                          ("routing_control_bytes_tx", "Routing bodies"),
                          ("ip_udp_header_bytes_tx", "Internet Protocol (IP) /\nUser Datagram "
                           "Protocol (UDP) headers")):
        ax.plot(rounds, frame[column] / 1024, marker="o", label=label)
    ax.plot(rounds, frame["total_wireless_bytes_tx"] / 1024, color="black",
            linestyle="--", label="Total")
    _save(fig, "vanet_communication_volume_vs_rounds.png",
          "Wireless communication volume (Internet Protocol boundary)")

    fig, ax = _figure("Control packet transmissions /\nfinal data packet arrivals")
    plt.sca(ax)
    ax.plot(rounds, frame["normalized_routing_load"], marker="o",
            label="Normalized Routing Load (NRL)")
    for round_num in frame.loc[frame["normalized_routing_load"].isna(), "round"]:
        ax.annotate("Undefined\n(no data arrivals)", (round_num, 0.03),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=8)
    ax.set_ylim(bottom=0)
    _save(fig, "vanet_normalized_routing_load_vs_rounds.png",
          "Normalized Routing Load (NRL)")

    fig, ax = _figure("Mean simulated latency (seconds)")
    plt.sca(ax)
    ax.plot(rounds, frame["successful_network_latency_mean_s"], marker="o",
            label="Successful envelopes")
    ax.plot(rounds, frame["network_latency_mean_s"], marker="s", linestyle="--",
            label="All attempts, including failed route discovery")
    _save(fig, "vanet_aodv_network_latency_vs_rounds.png",
          "Modeled network latency\n(not Federated Learning wall-clock time)")
    return True


# ----------------------------------------------------------------------
# Explanations markdown
# ----------------------------------------------------------------------
def _write_explanations(df: pd.DataFrame, rsu_range_m: float, out_dir: Path,
                        routing_csv: Optional[Path]) -> Path:
    vehicle_group = _participant_frame(df).groupby("round", as_index=True).mean(
        numeric_only=True)
    vehicle_group = vehicle_group[vehicle_group.index > 0]

    def _v(column: str) -> pd.Series:
        return _series(vehicle_group, column) if column in vehicle_group else pd.Series(
            dtype=float)

    # Population (carried-forward) means for accuracy/loss, matching the plots.
    _priv_pivot = _vehicle_frame(df).pivot_table(
        index="round", columns="node", values="private_test_accuracy_pct", aggfunc="max")
    _priv_pivot = _priv_pivot[_priv_pivot.index > 0].sort_index().ffill()
    _pop_priv = _priv_pivot.mean(axis=1, skipna=True).dropna()
    _loss_pivot = _vehicle_frame(df).pivot_table(
        index="round", columns="node", values="train_loss", aggfunc="max")
    _loss_pivot = _loss_pivot[_loss_pivot.index > 0].sort_index()
    _pop_loss = _loss_pivot.mean(axis=1, skipna=True).dropna()

    security, communication = _v("security_latency_ms"), _v("communication_latency_ms")
    end_to_end, action_response = _v("end_to_end_time_ms"), _v("action_to_response_ms")
    sig_gen = _v("signature_generation_ms")
    sig_ver = _v("signature_verification_ms").add(_v("batch_verification_ms"),
                                                  fill_value=0.0)
    encryption = _v("encryption_ms")
    goodput_mbps = _wireless_goodput_by_round(df) / 1_000_000.0
    server_total = df[(df["node"] == "Server")]
    coverage_total = (server_total.set_index("round")["vehicles_in_range_total"]
                      .sort_index() if "vehicles_in_range_total" in server_total else
                      pd.Series(dtype=float))
    coverage_total = pd.to_numeric(coverage_total, errors="coerce").dropna()
    if coverage_total.empty:
        coverage_reason = "No closest in-range RSU service values were recorded."
    elif coverage_total.min() == coverage_total.max():
        coverage_reason = (f"Coverage remained constant at {coverage_total.iloc[0]:.0f} "
                           "vehicles across all rounds.")
    else:
        coverage_reason = (
            f"Observed closest in-range RSU service varied from {coverage_total.min():.0f} "
            f"to {coverage_total.max():.0f} vehicles, recording entries or exits relative "
            f"to the {rsu_range_m:.0f} m radius. The graph alone does not establish the "
            "cause of each change.")

    entries = [
        ("vanet_accuracy_vs_rounds.png", _extrema_summary(_pop_priv, "%"),
         "Private accuracy is each vehicle's private model on its local held-out "
         "split; every vehicle trains it every round (DML against the latest global "
         "model), as in ProxyFL v1. The global proxy line is the aggregated model on "
         "the attack benchmark (attack1-5_test.csv), which is what v1's server "
         "reports. Accuracy can dip when non-IID data (--alpha) is used, the "
         "participating set changes with mobility, or DP-SGD adds noise to proxy "
         "updates. The learning rate decays per round (lr_decay), so the curves "
         "settle as training progresses."),
        ("vanet_loss_vs_rounds.png", _extrema_summary(_pop_loss),
         "The per-vehicle loss is the private model's DML training loss, "
         "(1-alpha) CE + alpha KL, averaged over the round's batches (v1 semantics). "
         "It falls while the private models fit their local samples under a "
         "decaying learning rate. Short rises are expected when DML transfers "
         "changing predictions between private and proxy models."),
        ("vanet_energy_training_vs_rounds.png", _extrema_summary(_v("energy_training_j"),
                                                                 "J"),
         "Training energy is computed from measured training time, so its rises and "
         "falls follow per-round CPU scheduling, batch execution time, and the "
         "number of active local samples rather than model accuracy alone."),
        ("vanet_energy_other_vs_rounds.png", _extrema_summary(_v("energy_total_j"), "J"),
         "Non-training energy follows cryptographic and communication durations. It "
         "increases when more peers are in range, more messages are verified, or a "
         "round spends longer transmitting updates."),
        ("vanet_energy_breakdown.png", _extrema_summary(_v("energy_total_j"), "J"),
         "The component lines move differently because training, security, and "
         "communication work are timed separately. The total follows whichever "
         "measured component dominates that round."),
        ("vanet_end_to_end_time_vs_rounds.png", _extrema_summary(end_to_end, "ms"),
         "End-to-end time includes training, cryptography, communication, and idle "
         "synchronization. Peaks usually indicate stragglers, RSU/server collection "
         "waits, or operating-system scheduling contention."),
        ("vanet_latency_breakdown.png", _extrema_summary(end_to_end, "ms"),
         "This is the legacy alias of the end-to-end component graph. Its total rises "
         "when training or synchronization wait rises and falls when all participants "
         "complete the hierarchy promptly."),
        ("vanet_end_to_end_security_comm_detail.png",
         f"Security: {_extrema_summary(security, 'ms')} Communication: "
         f"{_extrema_summary(communication, 'ms')}",
         "Security varies with batch size and verification work; communication is "
         "modeled wireless airtime and varies with message sizes and link distances."),
        ("vanet_action_to_response_latency.png", _extrema_summary(action_response, "ms"),
         "This timer starts when a vehicle sends LOCAL_UPDATE and stops when its "
         "upload airtime completes. It rises when that vehicle trains longer or waits "
         "for a later simulated-time slot and falls when uploads finish quickly."),
        ("vanet_cryptographic_operations.png",
         f"Generation: {_extrema_summary(sig_gen, 'ms')} Verification: "
         f"{_extrema_summary(sig_ver, 'ms')} Encryption: "
         f"{_extrema_summary(encryption, 'ms')}",
         "The curves change with the number of signed messages, the number of items "
         "in each batch, individual fallback verification, and runtime contention. "
         "They are operation-time measurements, not convergence indicators."),
        ("vanet_security_overhead.png", _extrema_summary(sig_ver, "ms"),
         "This legacy alias contains the same cryptographic series. Verification "
         "increases with more received signatures or fallback work and decreases "
         "when batches contain fewer items or execute faster."),
        ("vanet_throughput_vs_rounds.png", _extrema_summary(goodput_mbps, "Mbps"),
         "Modeled VANET goodput is the sum of successfully delivered wireless "
         "bits divided by their modeled PHY airtime. It reflects V2V, V2RSU, "
         "and RSU-to-vehicle link capacities, excluding wired backhaul timing."),
        ("vanet_vehicles_in_range_vs_rounds.png", _extrema_summary(coverage_total,
                                                                   "vehicles"),
         coverage_reason),
    ]
    lines = ["# VANET plot explanations", "",
             "These notes use the measured CSV values from this run. The listed causes "
             "explain the implementation mechanisms that can produce the observed changes; "
             "a graph alone does not prove which mechanism caused a particular point.",
             ""]
    for filename, observation, reason in entries:
        lines.extend([f"## `{filename}`", "", f"![{filename}]({filename})", "",
                      f"Observed range: {observation}", "",
                      f"Why the line rises and falls: {reason}", ""])
    if routing_csv is not None and routing_csv.exists() and routing_csv.stat().st_size:
        frame = pd.read_csv(routing_csv).sort_values("round").reset_index(drop=True)
        routing_total = (frame["rreq_bytes_tx"] + frame["rrep_bytes_tx"]
                         + frame["rerr_bytes_tx"]) / 1024.0
        wireless_total = frame["total_wireless_bytes_tx"] / 1024.0

        def _r(values, unit=""):
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            if numeric.empty:
                return "No values were recorded for this run."
            lo, hi = numeric.idxmin(), numeric.idxmax()
            suffix = f" {unit}" if unit else ""
            return (f"Minimum: {numeric.loc[lo]:.3f}{suffix} in round "
                    f"{int(frame.loc[lo, 'round'])}; maximum: {numeric.loc[hi]:.3f}{suffix} "
                    f"in round {int(frame.loc[hi, 'round'])}.")

        ok = frame["successful_network_latency_mean_s"].dropna()
        allm = frame["network_latency_mean_s"].dropna()
        aodv_entries = [
            ("vanet_aodv_routing_overhead_vs_rounds.png", _r(routing_total, "KiB"),
             "AODV routing overhead rises when Route Request floods, Route Reply "
             "returns, retries, or Route Error notifications require more hop-by-hop "
             "transmissions. It falls when active routes are reused and remain valid."),
            ("vanet_communication_volume_vs_rounds.png", _r(wireless_total, "KiB"),
             "Total wireless volume combines Federated Learning payloads, security "
             "bytes, AODV control bodies, and Internet Protocol/User Datagram Protocol "
             "headers. It changes with delivered updates, path length, and route "
             "discovery or repair traffic."),
            ("vanet_normalized_routing_load_vs_rounds.png",
             _r(frame["normalized_routing_load"]),
             "Normalized Routing Load increases when more AODV control-packet "
             "transmissions are needed per delivered data packet. It is undefined "
             "when no data packet reaches its destination and remains a separate "
             "metric rather than part of byte overhead."),
            ("vanet_aodv_network_latency_vs_rounds.png",
             "Successful deliveries: " + _r(
                 frame["successful_network_latency_mean_s"], "s") + " All attempts: "
             + _r(frame["network_latency_mean_s"], "s"),
             "Modeled AODV latency rises with longer multi-hop paths, route discovery, "
             "retries, and broken-link recovery. It falls when a short active route is "
             "available and packets can be forwarded immediately."),
        ]
        lines.extend(["", AODV_REPORT_MARKER, ""])
        for filename, observation, reason in aodv_entries:
            lines.extend([f"## `{filename}`", "", f"![{filename}]({filename})", "",
                          f"Observed range: {observation}", "",
                          f"Why the line rises and falls: {reason}", ""])
    path = out_dir / "vanet_plot_explanations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def plot_vanet_suite(metrics_csv: Path, routing_csv: Path, metadata_path: Path,
                     output_dir: Path, rsu_range_m: float) -> Dict[str, List[str]]:
    """Generate the full v1-style VANET figure suite. Returns PNG + doc paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    df = pd.read_csv(metrics_csv)
    _plot_accuracy_loss(df, output_dir, saved)
    _plot_from_csv(df, output_dir, saved)
    _plot_coverage(df, output_dir, saved)
    routing_ok = _plot_routing(routing_csv, metadata_path, output_dir, saved)
    doc = _write_explanations(df, rsu_range_m, output_dir,
                              routing_csv if routing_ok else None)
    return {"pngs": [str(p) for p in saved], "explanations": str(doc)}
