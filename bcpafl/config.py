"""Configuration for the BC-PAFL (ProxyFL Version 2) simulation.

Every tunable lives in :class:`SimulationConfig`.  The defaults reproduce the
reference setup described in README.md; the CLI in ``main.py`` overrides a
subset of them.  Paper equation numbers refer to the BC-PAFL architecture
(POMDP-driven adaptive FL with blockchain-managed pseudonyms).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

# Keep BLAS/OpenMP single-threaded: every vehicle trains in the same process
# and oversubscription only slows the simulation down.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Fixed RSU layout (name, x, y).  Five RSUs on a cross, 1800 m apart, as in
# ProxyFL v1, so coverage has real gaps that vehicles drive through.
DEFAULT_RSU_LAYOUT: Tuple[Tuple[str, float, float], ...] = (
    ("RSU_0", 0.0, 0.0),
    ("RSU_1", 0.0, 1800.0),
    ("RSU_2", 1800.0, 0.0),
    ("RSU_3", 0.0, -1800.0),
    ("RSU_4", -1800.0, 0.0),
)

COMPRESSION_MODES: Tuple[str, ...] = ("none", "q8", "q4", "topk")
AGGREGATION_STRATEGIES: Tuple[str, ...] = ("score", "data", "uniform")
SELECTION_STRATEGIES: Tuple[str, ...] = ("pomdp", "random", "all", "greedy")


@dataclass
class SimulationConfig:
    # ------------------------------------------------------------------
    # Federated learning
    # ------------------------------------------------------------------
    rounds: int = 10
    seed: int = 42
    num_vehicles: int = 30
    batch_size: int = 64
    learning_rate: float = 0.0015
    # Per-round exponential learning-rate decay, lr_t = lr * decay^(t-1), for
    # both the shared and the private model (v1's ExponentialLR, gamma=0.95).
    # 1.0 keeps the learning rate constant.
    lr_decay: float = 0.95
    max_samples_per_vehicle: int = 3000
    # Dirichlet concentration for label-skewed non-IID partitions; None = IID
    # (v1's setting).  With alpha=0.5 a few vehicles hold <50% benign rows and
    # their local accuracy sits far below the rest of the fleet.
    dirichlet_alpha: float | None = None
    # Rows held out from the training pool as each RSU's validation set, used
    # to "evaluate model performance and compute R" (Algorithm 1, line 17).
    rsu_validation_rows: int = 1500

    # ProxyFL heritage: the shared FL model M_i is the proxy model; each
    # vehicle also keeps a heterogeneous private model trained by Deep Mutual
    # Learning.  Disable to train M_i on Eq. (1) cross-entropy alone.  The
    # private model never leaves the vehicle, so -- as in v1 -- every vehicle
    # refines it every round, selected by the POMDP or not.
    use_private_models: bool = True
    private_local_epochs: int = 2
    dml_alpha: float = 0.5
    dml_beta: float = 0.5
    dml_temperature: float = 3.0
    # Optional DP-SGD on the shared model (0 disables it).
    dp_noise_multiplier: float = 0.0
    dp_clip_norm: float = 1.0
    dp_delta: float = 1e-5

    # ------------------------------------------------------------------
    # Selection strategy: "pomdp" is BC-PAFL; the others are baselines.
    # ------------------------------------------------------------------
    selection: str = "pomdp"
    random_selection_fraction: float = 0.5
    # Fixed configuration used by the non-POMDP baselines.
    baseline_compression: str = "none"
    baseline_aggregation: str = "data"
    baseline_local_epochs: int = 2

    # ------------------------------------------------------------------
    # POMDP controller (Eqs. 3-18)
    # ------------------------------------------------------------------
    belief_levels: int = 3                       # levels for trust / utility / uncertainty
    # Observation noise std; "rho" is in the log-seconds domain.
    observation_noise: Dict[str, float] = field(default_factory=lambda: {
        "trust": 0.10, "utility": 0.12, "uncertainty": 0.10, "rho": 0.5,
    })
    # Remaining-connection-time level centres in seconds (log-spaced).
    rho_level_centres: Tuple[float, ...] = (10.0, 30.0, 90.0, 270.0, 810.0)
    rho_log_spread: float = 0.55                  # std of log(Gamma_conn) around a centre
    # Vehicles perturb self-reported utility / uncertainty statistics (privacy).
    report_noise_std: float = 0.05
    transition_prior_strength: float = 8.0       # Dirichlet pseudo-counts on T
    xi_init: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    xi_learning_rate: float = 0.05
    xi_max: float = 4.0
    # Candidate thresholds Psi_Th the policy chooses from (Eq. 11).
    psi_thresholds: Tuple[float, ...] = (0.02, 0.08, 0.2)
    local_epoch_options: Tuple[int, ...] = (1, 2)   # update frequency f (Eq. 9)
    compression_options: Tuple[str, ...] = COMPRESSION_MODES
    aggregation_options: Tuple[str, ...] = AGGREGATION_STRATEGIES
    topk_fraction: float = 0.1
    min_selected_per_rsu: int = 1
    # Reward weights varsigma_1..5 (Eq. 14).
    reward_weights: Tuple[float, float, float, float, float] = (20.0, 0.3, 0.3, 0.3, 1.0)
    discount: float = 0.9                        # gamma in Eq. 16
    q_learning_rate: float = 0.05
    exploration_start: float = 0.5
    exploration_decay: float = 0.85
    exploration_min: float = 0.05
    # Budgets (Eqs. 17-18): per RSU per round.
    comm_budget_bytes: float = 60_000.0
    comp_budget_seconds: float = 400.0

    # ------------------------------------------------------------------
    # Trust (beta reputation) and anomaly screening
    # ------------------------------------------------------------------
    # RSU validation guard (extension): apply Eq. (12) with the largest step
    # eta in {1, 1/2, 1/4, 0} whose class-weighted RSU-validation loss does not
    # rise by more than this relative tolerance.  Disable for Eq. (12) verbatim.
    rsu_validation_guard: bool = True
    rsu_guard_tolerance: float = 0.02
    trust_forgetting: float = 0.9
    trust_median_multiplier: float = 3.0
    malicious_fraction: float = 0.1              # share of poisoning vehicles
    malicious_scale: float = 8.0                 # magnitude of a poisoned delta

    # ------------------------------------------------------------------
    # Mobility and wireless (simulated time)
    # ------------------------------------------------------------------
    rsu_layout: Tuple[Tuple[str, float, float], ...] = DEFAULT_RSU_LAYOUT
    rsu_range_m: float = 1000.0
    area_half_width_m: float = 2400.0
    speed_range_mps: Tuple[float, float] = (10.0, 30.0)   # 36-108 km/h
    heading_jitter_rad: float = 0.25
    gps_noise_m: float = 25.0
    round_duration_s: float = 90.0               # simulated seconds per FL round
    compute_samples_per_s: Tuple[float, float] = (25.0, 150.0)
    per_message_overhead_s: float = 0.05
    link_bandwidth_hz: float = 10_000_000.0
    link_tx_power_dbm: float = 23.0
    link_antenna_gain_db: float = 10.0
    link_path_loss_1m_db: float = 46.4
    link_path_loss_exponent: float = 2.7
    link_noise_figure_db: float = 9.0
    link_max_rate_bps: float = 6_000_000.0
    # Frame error model: per-frame success = 1 - exp(-snr / snr_ref), with ARQ.
    link_snr_reference: float = 1.0
    link_max_retries: int = 3
    # Temporary loss of connectivity independent of range (per round).
    random_disconnect_prob: Tuple[float, float] = (0.0, 0.15)

    # ------------------------------------------------------------------
    # Blockchain-managed pseudonyms
    # ------------------------------------------------------------------
    pseudonym_lifetime_rounds: int = 3
    pseudonym_pool_size: int = 3                 # fresh pseudonyms held per vehicle
    revoke_malicious_after: int = 3              # anomalous updates before TA revocation
    forged_auth_attempts_per_round: int = 1      # adversarial Sybil attempts to reject

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir: str = "results"
    make_plots: bool = True
    verbose: bool = True

    def validate(self) -> "SimulationConfig":
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.num_vehicles < 1:
            raise ValueError("num_vehicles must be >= 1")
        if self.selection not in SELECTION_STRATEGIES:
            raise ValueError(f"selection must be one of {SELECTION_STRATEGIES}")
        for mode in self.compression_options:
            if mode not in COMPRESSION_MODES:
                raise ValueError(f"unknown compression mode {mode!r}")
        for mode in self.aggregation_options:
            if mode not in AGGREGATION_STRATEGIES:
                raise ValueError(f"unknown aggregation strategy {mode!r}")
        if self.belief_levels < 2 or len(self.rho_level_centres) < 2:
            raise ValueError("every belief component needs at least two levels")
        if list(self.rho_level_centres) != sorted(self.rho_level_centres) or \
                min(self.rho_level_centres) <= 0:
            raise ValueError("rho_level_centres must be positive and increasing")
        if self.baseline_compression not in COMPRESSION_MODES:
            raise ValueError("unknown baseline compression")
        if self.baseline_aggregation not in AGGREGATION_STRATEGIES:
            raise ValueError("unknown baseline aggregation")
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        if not 0.0 <= self.malicious_fraction < 1.0:
            raise ValueError("malicious_fraction must be in [0, 1)")
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must be in (0, 1]")
        if self.private_local_epochs < 1:
            raise ValueError("private_local_epochs must be >= 1")
        if any(x < 0 for x in self.xi_init):
            raise ValueError("xi must be non-negative (Eq. 10)")
        if self.pseudonym_lifetime_rounds < 1 or self.pseudonym_pool_size < 1:
            raise ValueError("pseudonym lifetime and pool size must be >= 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    def link_params(self):
        from .channel import LinkParams
        return LinkParams(
            bandwidth_hz=self.link_bandwidth_hz, tx_power_dbm=self.link_tx_power_dbm,
            antenna_gain_db=self.link_antenna_gain_db,
            path_loss_1m_db=self.link_path_loss_1m_db,
            path_loss_exponent=self.link_path_loss_exponent,
            noise_figure_db=self.link_noise_figure_db, max_rate_bps=self.link_max_rate_bps,
            snr_reference=self.link_snr_reference, max_retries=self.link_max_retries)
