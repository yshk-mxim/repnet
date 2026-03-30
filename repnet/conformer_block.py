"""Lightweight Conformer block for 1D ECG signals.

Conformer = Conv + Self-Attention (Gulati et al. 2020).
Uses multi-head self-attention (window_size=0 = global).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LocalMultiHeadAttention1d(nn.Module):
    """Multi-head attention with local windowing for 1D sequences."""
    def __init__(self, channels, n_heads=4, window_size=0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.window_size = window_size  # 0 = full attention

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        """x: (B, C, L)"""
        B, C, L = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, H, D, L).transpose(2, 3)
        k = k.view(B, H, D, L).transpose(2, 3)
        v = v.view(B, H, D, L).transpose(2, 3)

        scale = math.sqrt(D)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.window_size > 0 and L > self.window_size:
            mask = torch.ones(L, L, device=x.device, dtype=torch.bool)
            for i in range(L):
                start = max(0, i - self.window_size // 2)
                end = min(L, i + self.window_size // 2 + 1)
                mask[i, start:end] = False
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).reshape(B, C, L)
        return self.out_proj(out)


class ConformerBlock1d(nn.Module):
    """Conformer block: FFN -> Attention -> Conv -> FFN, with residuals.

    Half-step FFN before and after, sandwiching attention and conv.
    """
    def __init__(self, channels, n_heads=4, conv_kernel=5, window_size=0,
                 ff_expansion=2):
        super().__init__()
        ff_dim = channels * ff_expansion

        self.ff1_norm = nn.LayerNorm(channels)
        self.ff1 = nn.Sequential(
            nn.Linear(channels, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, channels),
        )

        self.attn_norm = nn.LayerNorm(channels)
        self.attn = LocalMultiHeadAttention1d(channels, n_heads, window_size)

        self.conv_norm = nn.LayerNorm(channels)
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, conv_kernel, padding=conv_kernel // 2,
                      groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

        self.ff2_norm = nn.LayerNorm(channels)
        self.ff2 = nn.Sequential(
            nn.Linear(channels, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, channels),
        )

        self.final_norm = nn.LayerNorm(channels)

    def forward(self, x):
        """x: (B, C, L)"""
        x_t = x.transpose(1, 2)
        x_t = x_t + 0.5 * self.ff1(self.ff1_norm(x_t))

        x_c = x_t.transpose(1, 2)
        x_c = x_c + self.attn(self.attn_norm(x_c.transpose(1, 2)).transpose(1, 2))

        x_c = x_c + self.conv(self.conv_norm(x_c.transpose(1, 2)).transpose(1, 2))

        x_t = x_c.transpose(1, 2)
        x_t = x_t + 0.5 * self.ff2(self.ff2_norm(x_t))

        return self.final_norm(x_t).transpose(1, 2)
