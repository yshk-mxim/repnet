"""ECG model definitions: RepNet V9M, Conformer, BaselineCNN.

Three architectures from the paper, plus supporting blocks.
No training, no data loading, no device management.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from repnet.init import compute_etf

# ── Constants ─────────────────────────────────────────────
K = 12                   # number of rhythm classes
N_LEADS = 12             # ECG leads
BN_DIM = K + 2           # 14-dim bottleneck
BN_REG_WEIGHT = 0.01     # bottleneck regularization weight

RHYTHM_CLASSES = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE',
                  'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']


# ── Deterministic pooling ─────────────────────────────────

class _DetAdaptiveAvgPool1d(torch.autograd.Function):
    """Deterministic adaptive_avg_pool1d with explicit backward."""
    @staticmethod
    def forward(ctx, x, output_size):
        ctx.save_for_backward(x)
        ctx.output_size = output_size
        return F.adaptive_avg_pool1d(x, output_size)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        L = x.shape[-1]
        out_size = ctx.output_size
        grad_input = torch.zeros_like(x)
        for i in range(out_size):
            s = (i * L) // out_size
            e = ((i + 1) * L + out_size - 1) // out_size
            grad_input[..., s:e] += grad_output[..., i:i+1] / (e - s)
        return grad_input, None


def _deterministic_adaptive_avg_pool1d(x, output_size):
    if x.shape[-1] == output_size:
        return x
    return _DetAdaptiveAvgPool1d.apply(x, output_size)


def _pad_even_avg_pool(x, target_len):
    """Deterministic skip connection pool: pad to even length, then avg_pool1d."""
    L = x.shape[-1]
    if L == target_len:
        return x
    if L % 2 == 1:
        x = F.pad(x, (0, 1), mode='replicate')
    return F.avg_pool1d(x, kernel_size=2, stride=2)


# ── LayerNorm for 1D conv tensors ────────────────────────

class LayerNorm1d(nn.Module):
    """LayerNorm for (B, C, L) tensors — normalizes over C."""
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


# ── Bottleneck blocks ────────────────────────────────────

class SwiGLU1d(nn.Module):
    """SwiGLU activation for 1D conv."""
    def forward(self, x):
        gate, value = x.chunk(2, dim=1)
        return F.silu(gate) * value


class BottleneckBlockLN(nn.Module):
    """xresnet-style bottleneck + LayerNorm."""
    def __init__(self, ch, ks=5, expansion=4, stride=1, act_type='gelu',
                 pre_norm=False, pad_skip=False):
        super().__init__()
        mid = ch
        wide = ch * expansion
        self.pre_norm = pre_norm
        self.pad_skip = pad_skip
        if act_type == 'swiglu':
            self.conv1 = nn.Conv1d(wide, mid * 2, 1)
            self.act1 = SwiGLU1d()
            self.conv2 = nn.Conv1d(mid, mid * 2, ks, stride=stride, padding=ks // 2)
            self.act2 = SwiGLU1d()
            self.conv3 = nn.Conv1d(mid, wide, 1)
            self.act3 = nn.GELU()
        else:
            self.conv1 = nn.Conv1d(wide, mid, 1)
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv1d(mid, mid, ks, stride=stride, padding=ks // 2)
            self.act2 = nn.GELU()
            self.conv3 = nn.Conv1d(mid, wide, 1)
            self.act3 = nn.GELU()
        self.needs_pool = stride > 1
        self.ln = LayerNorm1d(wide)

    def forward(self, x):
        if self.pre_norm:
            normed = self.ln(x)
            out = self.act1(self.conv1(normed))
            out = self.act2(self.conv2(out))
            out = self.act3(self.conv3(out))
            if self.needs_pool:
                target_len = out.shape[2]
                skip = (_pad_even_avg_pool(x, target_len) if self.pad_skip
                        else _deterministic_adaptive_avg_pool1d(x, target_len))
            else:
                skip = x
            return out + skip
        else:
            out = self.act1(self.conv1(x))
            out = self.act2(self.conv2(out))
            out = self.conv3(out)
            if self.needs_pool:
                target_len = out.shape[2]
                skip = (_pad_even_avg_pool(x, target_len) if self.pad_skip
                        else _deterministic_adaptive_avg_pool1d(x, target_len))
            else:
                skip = x
            return self.ln(self.act3(out + skip))


class IntermediateBottleneck(nn.Module):
    """1x1 conv bottleneck: channels -> bn_dim -> channels + residual."""
    def __init__(self, channels, bn_dim=BN_DIM):
        super().__init__()
        self.compress = nn.Conv1d(channels, bn_dim, 1)
        self.act = nn.GELU()
        self.expand = nn.Conv1d(bn_dim, channels, 1)

    def forward(self, x):
        z = self.act(self.compress(x))
        return x + self.expand(z)


class ResBlock1dBN(nn.Module):
    """Simple residual block with BatchNorm."""
    def __init__(self, ch, ks=5):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, ks, padding=ks // 2)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        return x + F.relu(self.bn(self.conv(x)))


# ── RepNet V9H (base for V9M) ────────────────────────────

class ECGRepNetV9H(nn.Module):
    """V9 with LayerNorm after each bottleneck block."""
    def __init__(self, base_ch=64, ks=5, blocks=None, expansion=4,
                 etf_init=False, act_type='gelu', bn_dim=None, pre_norm=False,
                 pad_skip=False):
        super().__init__()
        if blocks is None:
            blocks = [3, 4, 23, 3]
        self.bn_dim = bn_dim or BN_DIM
        wide = base_ch * expansion

        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, base_ch, ks, stride=2, padding=ks // 2), nn.GELU(),
            nn.Conv1d(base_ch, base_ch, ks, padding=ks // 2), nn.GELU(),
            nn.Conv1d(base_ch, wide, ks, padding=ks // 2), nn.GELU(),
        )
        self.stem_pool = nn.AvgPool1d(3, 2, 1)

        self.stages = nn.ModuleList()
        for si, nb in enumerate(blocks):
            stage = nn.ModuleList()
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                stage.append(BottleneckBlockLN(base_ch, ks, expansion, stride,
                                               act_type, pre_norm=pre_norm,
                                               pad_skip=pad_skip))
            self.stages.append(stage)

        self.bn_linear = nn.Linear(wide * 2, self.bn_dim)
        self.bn_act = nn.GELU()
        self.heads = nn.ModuleList([nn.Linear(self.bn_dim, 1) for _ in range(K)])
        if etf_init:
            self._init_etf_heads()

    def _init_etf_heads(self):
        etf = compute_etf(K, self.bn_dim)
        for k in range(K):
            self.heads[k].weight.data = etf[k:k+1]
            self.heads[k].bias.data.zero_()

    def features(self, x):
        h = self.stem_pool(self.stem(x))
        for stage in self.stages:
            for blk in stage:
                h = blk(h)
        avg = h.mean(dim=-1)
        mx = h.max(dim=-1).values
        return torch.cat([avg, mx], 1)

    def penultimate(self, x):
        return self.bn_act(self.bn_linear(self.features(x)))

    def forward(self, x):
        h = self.penultimate(x)
        return torch.cat([hd(h) for hd in self.heads], 1), h


# ── RepNet V9M (multi-bottleneck, paper's primary model) ─

class ECGRepNetV9M(ECGRepNetV9H):
    """V9H + IntermediateBottleneck at each stage (1.93M params)."""
    def __init__(self, base_ch=64, ks=5, blocks=None, expansion=4,
                 etf_init=False, act_type='gelu', bn_dim=None, pre_norm=False,
                 pad_skip=False):
        if blocks is None:
            blocks = [3, 4, 23, 3]
        super().__init__(base_ch, ks, blocks, expansion, etf_init=etf_init,
                         act_type=act_type, bn_dim=bn_dim, pre_norm=pre_norm,
                         pad_skip=pad_skip)
        wide = base_ch * expansion
        self.ibns = nn.ModuleList([IntermediateBottleneck(wide) for _ in range(len(blocks))])

    def features(self, x):
        h = self.stem_pool(self.stem(x))
        for i, stage in enumerate(self.stages):
            for blk in stage:
                h = blk(h)
            h = self.ibns[i](h)
        avg = h.mean(dim=-1)
        mx = h.max(dim=-1).values
        return torch.cat([avg, mx], 1)



# ── Conformer variant ─────────────────────────────────────

class ECGRepNetConformer(nn.Module):
    """Hybrid Conv-Conformer: conv stages 0-2 + Conformer stage 3 (1.83M params)."""
    def __init__(self, base_ch=64, ks=5, blocks=None, expansion=4,
                 etf_init=False, act_type='gelu', bn_dim=None, n_heads=4,
                 pre_norm=False):
        super().__init__()
        if blocks is None:
            blocks = [3, 4, 23, 3]
        from repnet.conformer_block import ConformerBlock1d
        self.bn_dim = bn_dim or BN_DIM
        wide = base_ch * expansion

        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, base_ch, ks, stride=2, padding=ks // 2), nn.GELU(),
            nn.Conv1d(base_ch, base_ch, ks, padding=ks // 2), nn.GELU(),
            nn.Conv1d(base_ch, wide, ks, padding=ks // 2), nn.GELU(),
        )
        self.stem_pool = nn.AvgPool1d(3, 2, 1)

        self.stages = nn.ModuleList()
        for si in range(min(3, len(blocks))):
            stage = nn.ModuleList()
            for bi in range(blocks[si]):
                stride = 2 if (bi == 0 and si > 0) else 1
                stage.append(BottleneckBlockLN(base_ch, ks, expansion, stride,
                                               act_type, pre_norm=pre_norm))
            self.stages.append(stage)

        if len(blocks) > 3:
            self.conformer_blocks = nn.ModuleList([
                ConformerBlock1d(wide, n_heads=n_heads, conv_kernel=ks, window_size=0)
                for _ in range(blocks[3])
            ])
            self.conformer_downsample = nn.Sequential(
                nn.Conv1d(wide, wide, ks, stride=2, padding=ks // 2), nn.GELU(),
            )
        else:
            self.conformer_blocks = nn.ModuleList()
            self.conformer_downsample = None

        self.bn_linear = nn.Linear(wide * 2, self.bn_dim)
        self.bn_act = nn.GELU()
        self.heads = nn.ModuleList([nn.Linear(self.bn_dim, 1) for _ in range(K)])
        if etf_init:
            etf = compute_etf(K, self.bn_dim)
            for k in range(K):
                self.heads[k].weight.data = etf[k:k+1]
                self.heads[k].bias.data.zero_()

    def features(self, x):
        h = self.stem_pool(self.stem(x))
        for stage in self.stages:
            for blk in stage:
                h = blk(h)
        if self.conformer_downsample is not None:
            h = self.conformer_downsample(h)
            for blk in self.conformer_blocks:
                h = blk(h)
        avg = h.mean(dim=-1)
        mx = h.max(dim=-1).values
        return torch.cat([avg, mx], 1)

    def penultimate(self, x):
        return self.bn_act(self.bn_linear(self.features(x)))

    def forward(self, x):
        h = self.penultimate(x)
        return torch.cat([hd(h) for hd in self.heads], 1), h


# ── Baseline CNN ──────────────────────────────────────────

class BaselineCNN(nn.Module):
    """BN + ReLU + MaxPool + Dropout baseline (1.65M params)."""
    def __init__(self, ch=64, ks=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, ch, ks, stride=2, padding=ks // 2),
            nn.BatchNorm1d(ch), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        self.stage1 = nn.Sequential(*[ResBlock1dBN(ch, ks) for _ in range(3)])
        self.down1 = nn.Sequential(
            nn.Conv1d(ch, ch * 2, ks, stride=2, padding=ks // 2),
            nn.BatchNorm1d(ch * 2), nn.ReLU())
        self.stage2 = nn.Sequential(*[ResBlock1dBN(ch * 2, ks) for _ in range(4)])
        self.down2 = nn.Sequential(
            nn.Conv1d(ch * 2, ch * 4, ks, stride=2, padding=ks // 2),
            nn.BatchNorm1d(ch * 4), nn.ReLU())
        self.stage3 = nn.Sequential(*[ResBlock1dBN(ch * 4, ks) for _ in range(3)])
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(ch * 4 * 2, 128),
            nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, K))

    def features(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.down1(h)
        h = self.stage2(h)
        h = self.down2(h)
        h = self.stage3(h)
        avg = h.mean(dim=-1)
        mx = h.max(dim=-1).values
        return torch.cat([avg, mx], 1)

    def penultimate(self, x):
        f = self.features(x)
        h = self.head[0](f)   # Dropout
        h = self.head[1](h)   # Linear(512, 128)
        h = self.head[2](h)   # ReLU
        return h

    def forward(self, x):
        return self.head(self.features(x))


# ── Model registry ───────────────────────────────────────

MODEL_CLASSES = {
    'v9m': ECGRepNetV9M,
    'conformer': ECGRepNetConformer,
    'baseline': BaselineCNN,
}

REPNET_MODELS = {'v9m', 'conformer'}
