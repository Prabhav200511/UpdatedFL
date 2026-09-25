"""Trust scores tau_i and model-update anomaly screening.

tau_i is a beta-reputation estimate (Josang & Ismail) from historic
participation, "periodically updated" through on-chain RSU feedback:

    tau = (alpha + 1) / (alpha + beta + 2)

alpha counts accepted contributions; beta counts failures (dropouts) and --
with a heavier weight -- updates rejected as anomalous.  The TA carries each
identity's (alpha, beta) into its fresh pseudonyms, so reputation survives
pseudonym changes without RSUs being able to link them.

Anomaly screening generalises ProxyFL v1's median-times-k L2 filter to model
*deltas*: an update is rejected if its norm exceeds k times the robust norm
reference, or if it points away from the coordinate-wise median update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

POSITIVE_ACCEPTED = 1.0
NEGATIVE_DROPOUT = 0.5
NEGATIVE_ANOMALY = 3.0


def beta_trust(alpha: float, beta: float) -> float:
    return float((alpha + 1.0) / (alpha + beta + 2.0))


def ledger_trust(record: Dict) -> float:
    """tau~ for a pseudonym from its on-chain seed and feedback."""
    alpha = float(record.get("trust_alpha", 0.0)) + float(record.get("positive", 0.0))
    beta = float(record.get("trust_beta", 0.0)) + float(record.get("negative", 0.0))
    return beta_trust(alpha, beta)


@dataclass
class ScreeningResult:
    accepted: List[str]
    rejected: List[str]
    norms: Dict[str, float]
    cosines: Dict[str, float]


class AnomalyScreen:
    def __init__(self, median_multiplier: float, min_cosine: float = -0.1) -> None:
        self.k = float(median_multiplier)
        self.min_cosine = float(min_cosine)
        self._norm_history: Optional[float] = None

    def screen(self, deltas: Dict[str, np.ndarray],
               validation_loss: Optional[Callable[[np.ndarray], float]] = None,
               base_loss: Optional[float] = None,
               loss_inflation: float = 1.25) -> ScreeningResult:
        """``validation_loss(delta)`` (optional) scores a single update on the
        RSU validation set; it is used only in the cold start (fewer than three
        updates and no norm history), where robust statistics are unavailable."""
        ids = list(deltas)
        if not ids:
            return ScreeningResult([], [], {}, {})
        norms = {i: float(np.linalg.norm(deltas[i])) for i in ids}
        cosines: Dict[str, float] = {}
        if len(ids) < 3 and self._norm_history is None and validation_loss is not None \
                and base_loss is not None:
            accepted, rejected = [], []
            for i in ids:
                ok = np.isfinite(norms[i]) and validation_loss(deltas[i]) <= base_loss * loss_inflation
                (accepted if ok else rejected).append(i)
            if accepted:
                self._norm_history = float(np.median([norms[i] for i in accepted]))
            return ScreeningResult(accepted, rejected, norms, cosines)
        if len(ids) >= 3:
            reference_norm = float(np.median(list(norms.values())))
            median_delta = np.median(np.stack([deltas[i] for i in ids]), axis=0)
            m_norm = np.linalg.norm(median_delta)
            for i in ids:
                denom = norms[i] * m_norm
                cosines[i] = float(deltas[i] @ median_delta / denom) if denom > 0 else 1.0
        else:
            reference_norm = self._norm_history if self._norm_history is not None \
                else float(np.median(list(norms.values())))
        accepted, rejected = [], []
        for i in ids:
            too_large = reference_norm > 0 and norms[i] > self.k * reference_norm
            opposed = cosines.get(i, 1.0) < self.min_cosine
            (rejected if (too_large or opposed or not np.isfinite(norms[i])) else accepted).append(i)
        if accepted:
            accepted_median = float(np.median([norms[i] for i in accepted]))
            self._norm_history = accepted_median if self._norm_history is None \
                else 0.7 * self._norm_history + 0.3 * accepted_median
        return ScreeningResult(accepted, rejected, norms, cosines)
