"""Deterministic batch ordering for ECG training."""
import math
import torch
import numpy as np


def build_seeded_batches(Y, batch_size, n_epochs, seed=42):
    """Deterministic seeded shuffle batches.

    Excludes samples with no labels (all-zero rows).
    Each epoch gets a fresh permutation from seed + epoch.
    """
    has_label = Y.sum(dim=1) > 0 if isinstance(Y, torch.Tensor) else torch.tensor(Y.sum(axis=1) > 0)
    valid_indices = torch.where(has_label)[0]
    N_valid = len(valid_indices)

    batches = []
    for ep in range(n_epochs):
        g = torch.Generator()
        g.manual_seed(seed + ep)
        perm = torch.randperm(N_valid, generator=g)
        epoch_indices = valid_indices[perm]
        for i in range(0, N_valid, batch_size):
            batches.append(epoch_indices[i:i + batch_size])
    return batches


def _signal_hash(X, valid_indices):
    """Deterministic hash of each sample's signal content. Returns float in [0,1)."""
    with torch.no_grad():
        vals = X[valid_indices].abs().sum(dim=(1, 2)).cpu().numpy()
    return np.mod(vals * 0.618033988749895, 1.0)


def build_golden_ratio_batches(X, Y, batch_size, epoch):
    """Golden ratio quasi-random permutation. Fully seed-free.

    key[i] = (signal_hash(X[i]) + epoch * phi) mod 1
    Sort by key. Different permutation each epoch, data-dependent.
    """
    has_label = Y.sum(dim=1) > 0
    valid_indices = torch.where(has_label)[0]
    N_valid = len(valid_indices)

    PHI = (math.sqrt(5) - 1) / 2
    hashes = _signal_hash(X, valid_indices)
    keys = np.mod(hashes + epoch * PHI, 1.0)
    order = np.argsort(keys)
    ordered_indices = valid_indices[order]

    batches = []
    for i in range(0, N_valid, batch_size):
        batches.append(ordered_indices[i:i + batch_size])
    return batches
