"""Model architectures, parameter-vector utilities and evaluation.

The shared FL model M (global), M_k (RSU) and M_i (vehicle) all use
:class:`SharedModel`.  In ProxyFL terms it is the *proxy*: the only model that
is ever transmitted.  Each vehicle can additionally keep a heterogeneous
*private* model (reused from ProxyFL v1) that never leaves the vehicle and
learns from the shared model through Deep Mutual Learning.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from .data import NUM_CLASSES, NUM_FEATURES


class SharedModel(nn.Module):
    """Common-architecture FL model: 4 -> 64 -> 64 -> 6 (4,870 parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrivateSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


class PrivateMedium(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


class PrivateLarge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


PRIVATE_ARCHITECTURES = (PrivateSmall, PrivateMedium, PrivateLarge)


def dml_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor,
           temperature: float) -> torch.Tensor:
    """KL(teacher || student) at temperature T, scaled by T^2 (Deep Mutual Learning)."""
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_p, teacher_probs, reduction="batchmean") * temperature ** 2


# ----------------------------------------------------------------------
# Parameter vectors
# ----------------------------------------------------------------------
def model_spec(model: nn.Module) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    return tuple((name, tuple(t.shape)) for name, t in model.state_dict().items())


def state_to_vector(state: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate([t.detach().cpu().numpy().astype(np.float32).ravel()
                           for t in state.values()])


def vector_to_state(vector: np.ndarray,
                    spec: Iterable[Tuple[str, Tuple[int, ...]]]) -> "OrderedDict[str, torch.Tensor]":
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for name, shape in spec:
        size = int(np.prod(shape)) if shape else 1
        chunk = vector[offset:offset + size]
        if chunk.size != size:
            raise ValueError("parameter vector is shorter than the model spec")
        state[name] = torch.from_numpy(np.array(chunk, dtype=np.float32).reshape(shape))
        offset += size
    if offset != vector.size:
        raise ValueError("parameter vector is longer than the model spec")
    return state


def get_vector(model: nn.Module) -> np.ndarray:
    return state_to_vector(model.state_dict())


def set_vector(model: nn.Module, vector: np.ndarray) -> None:
    model.load_state_dict(vector_to_state(vector, model_spec(model)))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset,
             class_weights: torch.Tensor | None = None) -> Dict[str, float]:
    """Accuracy, macro-F1 and (weighted) cross-entropy on a dataset."""
    model.eval()
    x, y = dataset.tensors
    logits = model(x)
    loss = F.cross_entropy(logits, y, weight=class_weights).item()
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    f1 = f1_score(y.numpy(), pred.numpy(), average="macro", zero_division=0)
    rec = recall_score(y.numpy(), pred.numpy(), average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": float(f1), "macro_recall": float(rec), "loss": loss}


@torch.no_grad()
def local_statistics(model: nn.Module, dataset: TensorDataset,
                     class_weights: torch.Tensor | None = None,
                     max_rows: int = 1024) -> Dict[str, float]:
    """Per-sample loss moments and predictive entropy of ``model`` on local data.

    Used for the vehicle's observation report: statistical utility (Oort-style
    |D| * sqrt(mean loss^2)) and normalised predictive entropy (uncertainty).
    """
    model.eval()
    x, y = dataset.tensors
    x, y = x[:max_rows], y[:max_rows]
    logits = model(x)
    losses = F.cross_entropy(logits, y, weight=class_weights, reduction="none")
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1) / np.log(NUM_CLASSES)
    return {
        "loss_rms": float(torch.sqrt((losses ** 2).mean()).item()),
        "loss_mean": float(losses.mean().item()),
        "entropy": float(entropy.mean().item()),
    }


def make_loader(dataset: TensorDataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
