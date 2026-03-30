"""Deterministic weight initialization: DCT, Hadamard, Hartley, ETF.

All functions produce identical weights regardless of random seed.
No random number generators are used.
"""
import math
import numpy as np
import torch
import torch.nn as nn

# ── Basis matrices (cached) ──────────────────────────────────────────────────

_dct_cache = {}


def _dct_matrix(n):
    """Construct n x n DCT-II matrix (orthonormal). Cached."""
    if n in _dct_cache:
        return _dct_cache[n]
    j = torch.arange(n, dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    D = torch.cos(math.pi * i.unsqueeze(1) * (2 * j.unsqueeze(0) + 1) / (2 * n))
    D[0] *= 1.0 / math.sqrt(n)
    D[1:] *= math.sqrt(2.0 / n)
    D = D.float()
    _dct_cache[n] = D
    return D


def _hadamard_matrix(n):
    """Construct n x n Hadamard-like matrix (Walsh-Hadamard, padded to power of 2)."""
    size = 1
    H = torch.ones(1, 1, dtype=torch.float64)
    while size < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
        size *= 2
    return (H[:n, :n] / math.sqrt(n)).float()


def _hartley_matrix(n):
    """Construct n x n discrete Hartley transform matrix."""
    j = torch.arange(n, dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    angle = 2 * math.pi * i.unsqueeze(1) * j.unsqueeze(0) / n
    return ((torch.cos(angle) + torch.sin(angle)) / math.sqrt(n)).float()


def _sinusoidal_matrix(n):
    """Discrete Sine Transform Type II (orthonormal)."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    i = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    D = torch.sin(math.pi * (i + 1) * (k + 0.5) / n)
    D *= math.sqrt(2.0 / n)
    return D.float()


# Map stage index to orthogonal basis (for mixed-basis init)
_BASIS_FUNCTIONS = {
    0: _dct_matrix,
    1: _hadamard_matrix,
    2: _hartley_matrix,
    3: _dct_matrix,
}

# Named basis lookup for --basis argument
_NAMED_BASES = {
    'dct': _dct_matrix,
    'hadamard': _hadamard_matrix,
    'hartley': _hartley_matrix,
    'sinusoidal': _sinusoidal_matrix,
}


# ── ETF construction ─────────────────────────────────────────────────────────

def compute_etf(k, d):
    """Compute ETF simplex for k classes in d dimensions.

    Returns (k+1, d) matrix — row k is the OOD vertex.
    """
    k1 = k + 1
    M = torch.eye(k1) - torch.ones(k1, k1) / k1
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    etf = U[:, :min(k1 - 1, d)] * math.sqrt(k1 / (k1 - 1))
    if d > k1 - 1:
        etf = torch.cat([etf, torch.zeros(k1, d - (k1 - 1))], 1)
    return etf[:, :d]


# ── 1D DCT initialization ────────────────────────────────────────────────────

def deterministic_init(model, fixup_scale=0.01, mixed_bases=False, basis='dct'):
    """Deterministic initialization for 1D convolutional models.

    Each filter is a unique structured basis row, giving maximum diversity.
    conv3 in each residual branch is scaled by fixup_scale so each block
    starts near identity.

    Args:
        model: nn.Module to initialize
        fixup_scale: scaling for last conv in residual branches (0.01 optimal)
        mixed_bases: use DCT/Hadamard/Hartley per stage for gradient decorrelation
        basis: which basis to use when mixed_bases=False
               ('dct', 'hadamard', 'hartley', 'sinusoidal')
    """
    single_basis_fn = _NAMED_BASES.get(basis, _dct_matrix)
    for name, m in model.named_modules():
        if 'heads' in name:
            continue

        stage = 0
        for s in [3, 2, 1, 0]:
            if f'stage{s}' in name or f'layer{s}' in name:
                stage = s
                break

        if isinstance(m, nn.Conv1d):
            w = m.weight.data
            out_ch, in_ch, ks = w.shape
            fan_in = in_ch * ks
            target_std = 1.0 / math.sqrt(3.0 * fan_in)

            basis_fn = _BASIS_FUNCTIONS.get(stage, _dct_matrix) if mixed_bases else single_basis_fn

            if 'conv3' in name and fixup_scale == 0.0:
                nn.init.zeros_(w)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                continue

            N = max(out_ch, fan_in)
            D = basis_fn(N)
            for i in range(out_ch):
                w[i] = D[i % N, :fan_in].reshape(in_ch, ks)

            w.data -= w.data.mean()
            w.data *= target_std / (w.data.std() + 1e-8)

            if 'conv3' in name:
                w.data *= fixup_scale

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            target_std = 1.0 / math.sqrt(3.0 * fan_in)
            basis_fn = _BASIS_FUNCTIONS.get(stage, _dct_matrix) if mixed_bases else single_basis_fn
            N = max(m.weight.shape[0], m.weight.shape[1])
            D = basis_fn(N)
            w_new = D[:m.weight.shape[0], :m.weight.shape[1]].clone()
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


# ── 2D structured initialization (CIFAR) ─────────────────────────────────────

def _basis_2d(kh, kw, basis_fn):
    """Compute separable 2D basis from any 1D basis function for (kh, kw) kernel."""
    basis_h = basis_fn(kh).numpy().astype(np.float64)  # numpy imported at module level
    basis_w = basis_fn(kw).numpy().astype(np.float64)
    basis_2d = []
    for i in range(kh):
        for j in range(kw):
            basis_2d.append(np.outer(basis_h[i], basis_w[j]))
    return np.array(basis_2d)


def dct_init_2d(model, fixup_scale=0.01, basis='dct'):
    """Deterministic 2D structured initialization for Conv2d + Linear layers.

    For CIFAR ResNets. Separable 2D basis for spatial kernels,
    1D basis for channel mixing and Linear layers.

    Args:
        model: nn.Module to initialize
        fixup_scale: scaling for conv2 in residual branches
        basis: which orthogonal basis to use
               ('dct', 'hadamard', 'hartley', 'sinusoidal')
    """
    basis_fn = _NAMED_BASES.get(basis, _dct_matrix)

    conv_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for i, (name, conv) in enumerate(conv_layers):
        w = conv.weight
        c_out, c_in, kh, kw = w.shape
        fan_in = c_in * kh * kw
        spatial_basis = _basis_2d(kh, kw, basis_fn)
        n_spatial = len(spatial_basis)

        n_ch_basis = min(c_in, c_out)
        ch_basis = basis_fn(c_in).numpy().astype(np.float64)[:n_ch_basis, :c_in]

        filters = []
        for ci in range(n_ch_basis):
            for si in range(n_spatial):
                f = ch_basis[ci, :, None, None] * spatial_basis[si][None, :, :]
                filters.append(f)
                if len(filters) >= c_out:
                    break
            if len(filters) >= c_out:
                break
        n_generated = len(filters)
        while len(filters) < c_out:
            filters.append(filters[len(filters) % n_generated])
        filters = np.array(filters[:c_out])

        sigma_target = 1.0 / np.sqrt(3.0 * fan_in)
        for fi in range(c_out):
            filters[fi] -= filters[fi].mean()
            row_std = filters[fi].std()
            if row_std > 1e-10:
                filters[fi] *= sigma_target / row_std

        is_conv2 = 'conv2' in name
        scale = fixup_scale if is_conv2 else 1.0
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(filters, dtype=torch.float32) * scale)
            if conv.bias is not None:
                conv.bias.zero_()

    # Structured init for Linear layers (fully deterministic)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n_out, n_in = m.weight.shape
            B = basis_fn(max(n_out, n_in)).numpy().astype(np.float64)
            lin_w = B[:n_out, :n_in].copy()
            lin_w -= lin_w.mean(axis=1, keepdims=True)
            sigma = 1.0 / np.sqrt(3.0 * n_in)
            for i in range(n_out):
                s = lin_w[i].std()
                if s > 1e-10:
                    lin_w[i] *= sigma / s
            with torch.no_grad():
                m.weight.copy_(torch.tensor(lin_w, dtype=torch.float32))
            if m.bias is not None:
                m.bias.data.zero_()
