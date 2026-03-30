"""Evaluation metrics for ECG rhythm classification."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from repnet.models import K, RHYTHM_CLASSES


def eval_auc(model, X, Y, is_repnet=True, batch_size=500):
    """Compute macro AUC for multi-label ECG classification.

    Processes in batches to avoid OOM on large datasets.
    """
    model.eval()
    with torch.no_grad():
        parts = []
        for i in range(0, len(X), batch_size):
            out = model(X[i:i + batch_size])
            parts.append(out[0] if is_repnet else out)
        lo = torch.cat(parts, 0)
        probs = torch.sigmoid(lo).cpu().numpy()
        Y_np = Y.cpu().numpy()
        aucs = [roc_auc_score(Y_np[:, i], probs[:, i])
                for i in range(K)
                if Y_np[:, i].sum() > 0 and Y_np[:, i].sum() < len(Y_np)]
        return np.mean(aucs) if aucs else 0.0


def compute_all_metrics(model, X, Y, has_bottleneck=False, device=None, batch_size=500):
    """Compute all metrics for a trained model.

    Returns dict with: auc, per_auc, em, ham, er, mono, ece, overconf,
    conf_c, conf_w, ent_c, ent_w.
    """
    if device is None:
        device = next(model.parameters()).device

    Y_np = Y.cpu().numpy()
    model.eval()

    with torch.no_grad():
        lo_list, h_list = [], []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            if has_bottleneck:
                lo, h = model(batch)
                lo_list.append(lo)
                h_list.append(h)
            else:
                lo = model(batch)
                lo_list.append(lo)
                if hasattr(model, 'penultimate'):
                    h_list.append(model.penultimate(batch))
                else:
                    h_list.append(model.features(batch))
        lo = torch.cat(lo_list, 0)
        H = torch.cat(h_list, 0)
        P = torch.sigmoid(lo).cpu().numpy()

    preds = (P > 0.5).astype(float)

    # Per-class and macro AUC
    per_auc = {}
    for i in range(K):
        if Y_np[:, i].sum() > 0 and Y_np[:, i].sum() < len(Y_np):
            per_auc[RHYTHM_CLASSES[i]] = round(roc_auc_score(Y_np[:, i], P[:, i]), 4)
    macro_auc = np.mean(list(per_auc.values())) if per_auc else 0

    # Exact match & Hamming
    em = (preds == Y_np).all(1).mean()
    ham = (preds == Y_np).mean()

    # eRank
    S = torch.linalg.svdvals(H.float().cpu())
    p = S / S.sum()
    p = p[p > 1e-10]
    er = (-(p * p.log()).sum()).exp().item()

    # MonoScore (raw bottleneck)
    ms = []
    for j in range(min(H.shape[1], 20)):
        top_idx = H[:, j].topk(min(50, len(H))).indices.cpu()
        labels = Y[top_idx].cpu().float()
        class_presence = labels.sum(0)
        total = class_presence.sum().clamp(min=1)
        ms.append((class_presence / total).max().item())
    mono = np.mean(ms)

    # ECE (per-class, 10 bins)
    ece = 0.0
    for i in range(K):
        for b in range(10):
            lo_b, hi_b = b / 10, (b + 1) / 10
            mask = (P[:, i] >= lo_b) & (P[:, i] < hi_b)
            if mask.sum() > 0:
                ece += mask.mean() * abs(Y_np[mask, i].mean() - P[mask, i].mean())
    ece /= K

    # Confidence & entropy
    max_prob = P.max(1)
    correct_all = (preds == Y_np).all(1)
    conf_c = max_prob[correct_all].mean() if correct_all.sum() > 0 else 0
    conf_w = max_prob[~correct_all].mean() if (~correct_all).sum() > 0 else 0
    high_conf = max_prob > 0.9
    overconf = (high_conf & ~correct_all).sum() / max(high_conf.sum(), 1)

    entropy = -(P * np.log(P + 1e-10) + (1 - P) * np.log(1 - P + 1e-10)).mean(1)
    ent_c = entropy[correct_all].mean() if correct_all.sum() > 0 else 0
    ent_w = entropy[~correct_all].mean() if (~correct_all).sum() > 0 else 0

    # Energy score: -log(sum(exp(logits))). More negative = more confident.
    energy = -torch.logsumexp(lo.float().cpu(), dim=1).numpy()

    return {
        'auc': macro_auc, 'per_auc': per_auc,
        'em': em, 'ham': ham,
        'er': er, 'mono': mono,
        'ece': ece, 'overconf': overconf,
        'conf_c': conf_c, 'conf_w': conf_w,
        'ent_c': ent_c, 'ent_w': ent_w,
        'energy_mean': float(energy.mean()),
        'feat_dim': H.shape[1],
    }


def save_metrics(metrics, path):
    """Save metrics dict to JSON."""
    def jsonable(v):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            return round(float(v), 4)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {k: jsonable(vv) for k, vv in v.items()}
        if isinstance(v, float):
            return round(v, 4)
        return v

    out = {k: jsonable(v) for k, v in metrics.items()}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def print_metrics(name, metrics):
    """Print formatted metrics summary."""
    print(flush=True)
    print('=' * 60, flush=True)
    print(name, flush=True)
    print('=' * 60, flush=True)
    print(f'  Macro AUC:       {metrics["auc"]:.4f}', flush=True)
    print(f'  Exact Match:     {metrics["em"]:.4f}', flush=True)
    print(f'  Hamming Acc:     {metrics["ham"]:.4f}', flush=True)
    print(f'  eRank:           {metrics["er"]:.2f}/{metrics["feat_dim"]}', flush=True)
    print(f'  MonoScore:       {metrics["mono"]:.4f}', flush=True)
    print(f'  ECE:             {metrics["ece"]:.5f}', flush=True)
    print(f'  Overconfidence:  {metrics["overconf"]:.4f}', flush=True)
    if 'per_auc' in metrics:
        print('  Per-class AUC:', flush=True)
        for cls in sorted(metrics['per_auc'], key=lambda c: -metrics['per_auc'][c]):
            print(f'    {cls:6s}: {metrics["per_auc"][cls]:.4f}', flush=True)
