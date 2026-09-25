"""VeReMi-style VANET misbehaviour data: loading, preprocessing, partitioning.

Each vehicle V_i owns a private data model D_i (Eq. 1).  The training pool is
split into (a) a held-out global test set evaluated at the base station, (b)
one validation set per RSU used to compute the reward (Algorithm 1, line 17),
and (c) per-vehicle partitions, optionally label-skewed with a Dirichlet
prior so that vehicles differ in data utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

FEATURE_COLS = ["velocity_x", "velocity_y", "constant_offset_check", "total_displacement"]
TARGET_COL = "attacktype"
NUM_CLASSES = 6
NUM_FEATURES = len(FEATURE_COLS)


class FeatureTransform:
    """Signed log1p followed by standardisation.

    The raw features have heavy tails (|x| up to 1.6e4 with std ~1e2), which
    makes a plain StandardScaler squash almost every row into a tiny range.
    The transform is fit on the training pool only.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @staticmethod
    def _signed_log(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.log1p(np.abs(values))

    def fit(self, frame: pd.DataFrame) -> "FeatureTransform":
        logged = self._signed_log(frame[FEATURE_COLS].to_numpy(dtype=np.float64))
        self.mean = logged.mean(axis=0)
        self.std = logged.std(axis=0) + 1e-8
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureTransform must be fit before transform")
        logged = self._signed_log(frame[FEATURE_COLS].to_numpy(dtype=np.float64))
        return ((logged - self.mean) / self.std).astype(np.float32)


def to_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.asarray(features, dtype=np.float32)),
                         torch.from_numpy(np.asarray(labels, dtype=np.int64)))


@dataclass
class FederatedData:
    vehicle_datasets: List[TensorDataset]
    vehicle_validation: List[TensorDataset]
    rsu_validation: Dict[str, TensorDataset]
    global_test: TensorDataset
    attack_benchmark: TensorDataset | None
    class_weights: torch.Tensor
    transform: FeatureTransform

    def label_histogram(self, index: int) -> np.ndarray:
        labels = self.vehicle_datasets[index].tensors[1].numpy()
        return np.bincount(labels, minlength=NUM_CLASSES)


def _dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                         rng: np.random.Generator, max_per_client: int) -> List[np.ndarray]:
    """Label-skewed partition.

    Benign traffic (class 0) is observed by every vehicle, so it is split
    evenly; each attack class is split across vehicles by Dir(alpha), so
    vehicles differ in which misbehaviour types they have witnessed.
    """
    indices_by_class = [rng.permutation(np.flatnonzero(labels == c)) for c in range(NUM_CLASSES)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c, class_indices in enumerate(indices_by_class):
        if len(class_indices) == 0:
            continue
        proportions = (np.full(num_clients, 1.0 / num_clients) if c == 0
                       else rng.dirichlet(np.full(num_clients, alpha)))
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(class_indices, cuts)):
            client_indices[client].extend(chunk.tolist())
    result = []
    for client in range(num_clients):
        idx = np.asarray(client_indices[client], dtype=np.int64)
        if len(idx) > max_per_client:
            # Subsample without destroying the vehicle's label mix.
            idx = rng.choice(idx, size=max_per_client, replace=False)
        result.append(rng.permutation(idx))
    return result


def load_federated_data(num_vehicles: int, rsu_names: Sequence[str], *, seed: int,
                        dirichlet_alpha: float | None, max_samples_per_vehicle: int,
                        rsu_validation_rows: int, data_dir: Path,
                        global_test_fraction: float = 0.1,
                        min_samples_per_vehicle: int = 60) -> FederatedData:
    rng = np.random.default_rng(seed)
    frame = pd.read_csv(data_dir / "Main_data_shuffled.csv")
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)

    n_test = int(len(frame) * global_test_fraction)
    test_frame = frame.iloc[:n_test]
    rest = frame.iloc[n_test:]
    n_val_total = rsu_validation_rows * len(rsu_names)
    val_frame = rest.iloc[:n_val_total]
    pool = rest.iloc[n_val_total:].reset_index(drop=True)

    transform = FeatureTransform().fit(pool)
    pool_x = transform.transform(pool)
    pool_y = pool[TARGET_COL].to_numpy(dtype=np.int64)

    if dirichlet_alpha is None:
        order = rng.permutation(len(pool))
        per_client = min(max_samples_per_vehicle, len(pool) // num_vehicles)
        partitions = [order[i * per_client:(i + 1) * per_client] for i in range(num_vehicles)]
    else:
        partitions = _dirichlet_partition(pool_y, num_vehicles, dirichlet_alpha, rng,
                                          max_samples_per_vehicle)
    # Guarantee every vehicle has enough rows to train and validate locally.
    spare = rng.permutation(len(pool))
    cursor = 0
    fixed = []
    for part in partitions:
        if len(part) < min_samples_per_vehicle:
            need = min_samples_per_vehicle - len(part)
            part = np.concatenate([part, spare[cursor:cursor + need]])
            cursor += need
        fixed.append(part)

    vehicle_train, vehicle_val = [], []
    for part in fixed:
        n_local_val = max(10, int(0.15 * len(part)))
        val_idx, train_idx = part[:n_local_val], part[n_local_val:]
        vehicle_train.append(to_dataset(pool_x[train_idx], pool_y[train_idx]))
        vehicle_val.append(to_dataset(pool_x[val_idx], pool_y[val_idx]))

    rsu_validation = {}
    for k, name in enumerate(rsu_names):
        chunk = val_frame.iloc[k * rsu_validation_rows:(k + 1) * rsu_validation_rows]
        rsu_validation[name] = to_dataset(transform.transform(chunk),
                                          chunk[TARGET_COL].to_numpy(dtype=np.int64))

    global_test = to_dataset(transform.transform(test_frame),
                             test_frame[TARGET_COL].to_numpy(dtype=np.int64))

    attack_frames = [pd.read_csv(p) for p in sorted(Path(data_dir).glob("attack*_test.csv"))]
    attack_benchmark = None
    if attack_frames:
        attacks = pd.concat(attack_frames, ignore_index=True)
        attack_benchmark = to_dataset(transform.transform(attacks),
                                      attacks[TARGET_COL].to_numpy(dtype=np.int64))

    counts = np.bincount(pool_y, minlength=NUM_CLASSES).astype(np.float64)
    weights = np.sqrt(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0)))
    weights = weights / weights.mean()
    return FederatedData(vehicle_train, vehicle_val, rsu_validation, global_test,
                         attack_benchmark, torch.tensor(weights, dtype=torch.float32), transform)
