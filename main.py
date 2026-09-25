"""Command-line entry point for BC-PAFL (ProxyFL Version 2).

Examples::

    python main.py                                  # BC-PAFL, default settings
    python main.py --rounds 20 --vehicles 40
    python main.py --selection random               # conventional baseline
    python main.py --compare --rounds 15            # BC-PAFL vs. all baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from bcpafl.config import PROJECT_ROOT, SELECTION_STRATEGIES, SimulationConfig


def build_config(args: argparse.Namespace) -> SimulationConfig:
    cfg = SimulationConfig(
        rounds=args.rounds, seed=args.seed, num_vehicles=args.vehicles,
        selection=args.selection, malicious_fraction=args.malicious,
        dirichlet_alpha=None if args.iid else args.alpha,
        use_private_models=not args.no_private, dp_noise_multiplier=args.dp_noise,
        output_dir=args.output, make_plots=not args.no_plots, verbose=not args.quiet)
    return cfg.validate()


def run_one(cfg: SimulationConfig, output_dir: Path) -> dict:
    from bcpafl.plotting import plot_run
    from bcpafl.simulation import Simulation
    from bcpafl.vanet_plots import plot_vanet_suite

    started = time.perf_counter()
    sim = Simulation(cfg)
    summary = sim.run()
    summary["wall_clock_s"] = time.perf_counter() - started
    paths = sim.save(output_dir)
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str))
    if cfg.make_plots:
        plot_run(paths["rounds"], paths["rsu_rounds"], output_dir)
        vanet = plot_vanet_suite(paths["vanet_metrics"], paths["vanet_routing_rounds"],
                                 paths["vanet_routing_metadata"], output_dir,
                                 cfg.rsu_range_m)
        summary["vanet_figures"] = vanet
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--vehicles", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", choices=SELECTION_STRATEGIES, default="pomdp")
    parser.add_argument("--malicious", type=float, default=0.1,
                        help="fraction of vehicles that poison their updates")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for non-IID label skew")
    parser.add_argument("--iid", action="store_true", help="IID partitions instead")
    parser.add_argument("--no-private", action="store_true",
                        help="disable ProxyFL private models (train M_i on Eq. 1 only)")
    parser.add_argument("--dp-noise", type=float, default=0.0,
                        help="DP-SGD noise multiplier on the shared model (0 = off)")
    parser.add_argument("--compare", action="store_true",
                        help="run BC-PAFL and every baseline selection strategy")
    parser.add_argument("--output", default="results")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = build_config(args)
    root = Path(args.output)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not args.compare:
        summary = run_one(cfg, root / cfg.selection)
        print(json.dumps({k: summary[k] for k in (
            "final_test_accuracy", "final_test_macro_f1", "overall_dropout_rate",
            "total_upload_bytes", "total_anomalies_rejected", "malicious_accepted",
            "auth_rejected", "revocations", "chain_height", "chain_valid",
            "replicas_consistent", "wall_clock_s")}, indent=2, default=str))
        return 0

    from bcpafl.plotting import plot_comparison
    results = {}
    for strategy in SELECTION_STRATEGIES:
        print(f"\n===== selection = {strategy} =====", flush=True)
        results[strategy] = run_one(replace(cfg, selection=strategy), root / strategy)
    comparison = {s: {k: r[k] for k in ("final_test_accuracy", "final_test_macro_f1",
                                         "best_test_accuracy", "overall_dropout_rate",
                                         "total_upload_bytes", "malicious_accepted",
                                         "total_dropouts", "total_selected")}
                  for s, r in results.items()}
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2))
    if cfg.make_plots:
        plot_comparison({s: root / s / "rounds.csv" for s in results}, root)
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
