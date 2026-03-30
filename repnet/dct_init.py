"""Deterministic weight initialization using structured orthogonal bases.

This module provides seed-free, fully deterministic initialization for neural
networks using the Discrete Cosine Transform (DCT) and related orthogonal
transforms. Weights are constructed analytically from orthonormal basis
matrices, eliminating dependence on random seeds.

Supports both 1D convolutions (time series, audio, ECG) and 2D convolutions
(images). All functions produce identical weights regardless of the PyTorch
random seed or call order.

Usage:
    from repnet.dct_init import dct_init_1d, dct_init_2d, mixed_basis_init

    # 1D models (ECG, audio, time series)
    model = MyModel1D()
    dct_init_1d(model, fixup_scale=0.0)

    # 2D models (images)
    model = MyModel2D()
    dct_init_2d(model, fixup_scale=0.0)

    # Advanced: mixed bases + Halton perturbation
    mixed_basis_init(model, fixup_scale=0.01, halton_noise=0.1)

References:
    - DCT-II orthonormal basis: Ahmed, Natarajan, Rao (1974)
    - Fixup initialization: Zhang et al. (2019)
    - ETF classifier: Yang et al., Neural Collapse (2022)
    - ZerO init (mixed bases): Zhao et al. (arXiv 2110.12661)
    - Low-discrepancy sequences: Halton (1960)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basis matrix constructors (cached, deterministic)
# ---------------------------------------------------------------------------

_cache = {}


def _dct_matrix(n):
    """Construct an n x n DCT-II matrix (orthonormal).

    The DCT-II basis provides maximally smooth, frequency-ordered filters.
    Row 0 is the DC (constant) component; subsequent rows increase in
    frequency. The matrix is cached after first construction.

    Args:
        n: Matrix dimension.

    Returns:
        Tensor of shape (n, n), orthonormal DCT-II matrix.
    """
    key = ("dct", n)
    if key in _cache:
        return _cache[key]
    j = torch.arange(n, dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    D = torch.cos(math.pi * i.unsqueeze(1) * (2 * j.unsqueeze(0) + 1) / (2 * n))
    D[0] *= 1.0 / math.sqrt(n)
    D[1:] *= math.sqrt(2.0 / n)
    D = D.float()
    _cache[key] = D
    return D


def _hadamard_matrix(n):
    """Construct an n x n Hadamard-like matrix (Walsh-Hadamard, truncated).

    Uses the Sylvester construction for power-of-2 sizes, then truncates
    to n x n. Provides square-wave basis functions -- complementary to the
    smooth DCT basis when used in alternating stages.

    Args:
        n: Matrix dimension.

    Returns:
        Tensor of shape (n, n), normalized Hadamard matrix.
    """
    key = ("hadamard", n)
    if key in _cache:
        return _cache[key]
    size = 1
    H = torch.ones(1, 1, dtype=torch.float32)
    while size < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
        size *= 2
    H = H[:n, :n] / math.sqrt(n)
    _cache[key] = H
    return H


def _hartley_matrix(n):
    """Construct an n x n Discrete Hartley Transform matrix.

    The Hartley basis combines cosine and sine components into a single
    real-valued transform. Provides frequency-ordered filters like DCT
    but with different phase characteristics.

    Args:
        n: Matrix dimension.

    Returns:
        Tensor of shape (n, n), normalized Hartley matrix.
    """
    key = ("hartley", n)
    if key in _cache:
        return _cache[key]
    j = torch.arange(n, dtype=torch.float32)
    i = torch.arange(n, dtype=torch.float32)
    angle = 2 * math.pi * i.unsqueeze(1) * j.unsqueeze(0) / n
    H = (torch.cos(angle) + torch.sin(angle)) / math.sqrt(n)
    _cache[key] = H
    return H


def _halton_sequence(n, dim):
    """Generate n points of a Halton quasi-random sequence in dim dimensions.

    Low-discrepancy sequence that fills the space more uniformly than
    pseudorandom samples. Used as deterministic perturbation to break
    filter correlation while preserving reproducibility.

    Args:
        n: Number of points.
        dim: Number of dimensions.

    Returns:
        Tensor of shape (n, dim), values in [0, 1).
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    result = torch.zeros(n, dim)
    for d in range(dim):
        b = primes[d % len(primes)]
        for i in range(n):
            f = 1.0 / b
            idx = i + 1
            val = 0.0
            while idx > 0:
                val += f * (idx % b)
                idx //= b
                f /= b
            result[i, d] = val
    return result


# Basis function registry for mixed-basis init (stage index -> constructor)
_BASIS_REGISTRY = {
    0: _dct_matrix,
    1: _hadamard_matrix,
    2: _hartley_matrix,
    3: _dct_matrix,
}


# ---------------------------------------------------------------------------
# ETF (Equiangular Tight Frame) construction
# ---------------------------------------------------------------------------

def compute_etf(k, d):
    """Compute an Equiangular Tight Frame (ETF) simplex for k classes in d dimensions.

    ETF vertices are maximally separated directions in d-dimensional space,
    forming the vertices of a regular simplex. Used to initialize classifier
    head weights so that class directions start maximally apart.

    Returns k+1 vertices (the extra vertex can serve as an OOD direction).

    Args:
        k: Number of classes.
        d: Embedding dimension.

    Returns:
        Tensor of shape (k+1, d), ETF simplex vertices.
    """
    k1 = k + 1
    M = torch.eye(k1) - torch.ones(k1, k1) / k1
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    etf = U[:, : min(k1 - 1, d)] * math.sqrt(k1 / (k1 - 1))
    if d > k1 - 1:
        etf = torch.cat([etf, torch.zeros(k1, d - (k1 - 1))], 1)
    return etf[:, :d]


# ---------------------------------------------------------------------------
# 1D initialization (Conv1d + Linear)
# ---------------------------------------------------------------------------

def dct_init_1d(model, fixup_scale=0.0):
    """Initialize all Conv1d and Linear layers with DCT basis vectors.

    Each filter is assigned a unique row from the DCT-II matrix computed
    over the full fan-in dimension (in_channels * kernel_size). Weights are
    then rescaled to match Kaiming-uniform target standard deviation.

    Fixup scaling: conv layers whose name contains 'conv3' (the last
    convolution in a residual branch) are scaled by fixup_scale, so each
    residual block starts near identity.

    Classifier heads (modules with 'heads' in name) are skipped -- use
    compute_etf() for those.

    This function is fully deterministic and seed-independent.

    Args:
        model: A torch.nn.Module with Conv1d and/or Linear layers.
        fixup_scale: Scale factor for residual-branch output convolutions.
            0.0 = zero init (original Fixup, block starts as identity).
            0.01 = small residual contribution (gentle start).
            1.0 = no fixup (full DCT for all convolutions).
    """
    for name, m in model.named_modules():
        if "heads" in name:
            continue

        if isinstance(m, nn.Conv1d):
            w = m.weight.data
            out_ch, in_ch, ks = w.shape
            fan_in = in_ch * ks
            target_std = 1.0 / math.sqrt(3.0 * fan_in)

            if "conv3" in name and fixup_scale == 0.0:
                nn.init.zeros_(w)
            elif "conv3" in name and fixup_scale < 1.0:
                N = max(out_ch, fan_in)
                D = _dct_matrix(N)
                for i in range(out_ch):
                    w[i] = D[i % N, :fan_in].reshape(in_ch, ks)
                w.data -= w.data.mean()
                w.data *= target_std / (w.data.std() + 1e-8)
                w.data *= fixup_scale
            else:
                N = max(out_ch, fan_in)
                D = _dct_matrix(N)
                for i in range(out_ch):
                    w[i] = D[i % N, :fan_in].reshape(in_ch, ks)
                w.data -= w.data.mean()
                w.data *= target_std / (w.data.std() + 1e-8)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            target_std = 1.0 / math.sqrt(3.0 * fan_in)
            N = max(m.weight.shape[0], m.weight.shape[1])
            D = _dct_matrix(N)
            w_new = D[: m.weight.shape[0], : m.weight.shape[1]].clone()
            w_new = w_new - w_new.mean()
            w_new = w_new * (target_std / (w_new.std() + 1e-8))
            m.weight.data.copy_(w_new)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 2D initialization (Conv2d + Linear)
# ---------------------------------------------------------------------------

def dct_init_2d(model, fixup_scale=0.0):
    """Initialize all Conv2d and Linear layers with 2D DCT basis vectors.

    For Conv2d layers, the DCT matrix is constructed over the flattened
    fan-in dimension (in_channels * kH * kW). Each filter receives a
    unique DCT row, reshaped to the kernel shape.

    Follows the same Fixup conventions as dct_init_1d: layers with 'conv3'
    in their name are scaled by fixup_scale.

    Args:
        model: A torch.nn.Module with Conv2d and/or Linear layers.
        fixup_scale: Scale factor for residual-branch output convolutions.
            Same semantics as dct_init_1d.
    """
    for name, m in model.named_modules():
        if "heads" in name or "head" in name:
            continue

        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            out_ch, in_ch, kH, kW = w.shape
            fan_in = in_ch * kH * kW
            target_std = 1.0 / math.sqrt(3.0 * fan_in)

            if "conv3" in name and fixup_scale == 0.0:
                nn.init.zeros_(w)
            elif "conv3" in name and fixup_scale < 1.0:
                N = max(out_ch, fan_in)
                D = _dct_matrix(N)
                for i in range(out_ch):
                    w[i] = D[i % N, :fan_in].reshape(in_ch, kH, kW)
                w.data -= w.data.mean()
                w.data *= target_std / (w.data.std() + 1e-8)
                w.data *= fixup_scale
            else:
                N = max(out_ch, fan_in)
                D = _dct_matrix(N)
                for i in range(out_ch):
                    w[i] = D[i % N, :fan_in].reshape(in_ch, kH, kW)
                w.data -= w.data.mean()
                w.data *= target_std / (w.data.std() + 1e-8)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            target_std = 1.0 / math.sqrt(3.0 * fan_in)
            N = max(m.weight.shape[0], m.weight.shape[1])
            D = _dct_matrix(N)
            w_new = D[: m.weight.shape[0], : m.weight.shape[1]].clone()
            w_new = w_new - w_new.mean()
            w_new = w_new * (target_std / (w_new.std() + 1e-8))
            m.weight.data.copy_(w_new)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Mixed-basis initialization (advanced)
# ---------------------------------------------------------------------------

def mixed_basis_init(model, fixup_scale=0.0, halton_noise=0.0,
                     depth_scaled_fixup=False):
    """Initialize with different orthogonal bases per network stage.

    Uses DCT for stages 0 and 3, Hadamard for stage 1, and Hartley for
    stage 2. This decorrelates inter-layer gradients by ensuring adjacent
    stages use structurally different weight matrices.

    Optionally adds deterministic quasi-random Halton perturbation to break
    exact filter correlation while maintaining full reproducibility.

    Supports Conv1d, Conv2d, and Linear layers. Stage assignment is inferred
    from module names containing 'stage0', 'stage1', 'layer0', etc.

    Args:
        model: A torch.nn.Module.
        fixup_scale: Scale factor for residual-branch output convolutions.
        halton_noise: Strength of Halton quasi-random perturbation.
            0.0 = off. Recommended range: 0.1-0.2 (10-20% of target std).
        depth_scaled_fixup: If True, scale fixup_scale by layer depth so
            deeper blocks start with more capacity (derived from Fixup,
            Zhang et al. 2019).
    """
    n_blocks = sum(1 for name, _ in model.named_modules() if "conv3" in name)
    block_idx = 0

    for name, m in model.named_modules():
        if "heads" in name:
            continue

        # Determine stage for basis selection
        stage = 0
        for s in [3, 2, 1, 0]:
            if f"stage{s}" in name or f"layer{s}" in name:
                stage = s
                break

        basis_fn = _BASIS_REGISTRY.get(stage, _dct_matrix)

        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            w = m.weight.data
            if isinstance(m, nn.Conv1d):
                out_ch, in_ch, ks = w.shape
                fan_in = in_ch * ks
                reshape_dims = (in_ch, ks)
            else:
                out_ch, in_ch, kH, kW = w.shape
                fan_in = in_ch * kH * kW
                reshape_dims = (in_ch, kH, kW)

            target_std = 1.0 / math.sqrt(3.0 * fan_in)

            # Handle fixup for conv3 layers
            if "conv3" in name:
                block_idx += 1
                if fixup_scale == 0.0:
                    nn.init.zeros_(w)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    continue

                if depth_scaled_fixup and n_blocks > 0:
                    layer_scale = fixup_scale * (0.5 + 0.5 * block_idx / n_blocks)
                else:
                    layer_scale = fixup_scale
            else:
                layer_scale = 1.0

            # Fill from structured basis
            N = max(out_ch, fan_in)
            D = basis_fn(N)
            for i in range(out_ch):
                w[i] = D[i % N, :fan_in].reshape(*reshape_dims)

            # Add deterministic Halton perturbation
            if halton_noise > 0 and layer_scale > 0:
                noise = _halton_sequence(out_ch, fan_in) * 2 - 1
                noise = noise[:, :fan_in].reshape(out_ch, *reshape_dims).to(w.device)
                w.data += halton_noise * target_std * noise

            # Rescale to target std
            w.data -= w.data.mean()
            w.data *= target_std / (w.data.std() + 1e-8)

            # Apply fixup scaling for conv3
            if "conv3" in name:
                w.data *= layer_scale

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            target_std = 1.0 / math.sqrt(3.0 * fan_in)
            N = max(m.weight.shape[0], m.weight.shape[1])
            D = basis_fn(N)
            w_new = D[: m.weight.shape[0], : m.weight.shape[1]].clone()

            if halton_noise > 0:
                noise = _halton_sequence(m.weight.shape[0], m.weight.shape[1]) * 2 - 1
                w_new += halton_noise * target_std * noise

            w_new = w_new - w_new.mean()
            w_new = w_new * (target_std / (w_new.std() + 1e-8))
            m.weight.data.copy_(w_new)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
