"""OOD detection and noise robustness evaluation for ECG models.

Evaluates whether trained models can detect out-of-distribution inputs
and measures classification degradation under clinical noise.

OOD scoring methods:
  - Entropy: H(p) = -p*log(p) - (1-p)*log(1-p), averaged over classes
  - Energy:  -log(sum(exp(logits)))
  - Max logit: max(sigmoid(logits))
  - ETF OOD projection: cosine with K+1 ETF vertex (bottleneck models only)

OOD test sets:
  - Flat/leadoff: electrode disconnection (near-constant signal)
  - Gaussian: random noise, no cardiac structure
  - Muscle artifact: synthetic high-frequency bursts
  - EMI: 50/60 Hz powerline interference + harmonics
  - Corrupted ECG: real ECG + heavy noise overlay

Clinical noise (requires MIT-BIH NSTDB):
  - Baseline wander, muscle artifact, electrode motion at 0-24 dB SNR

Usage:
  python eval_ood.py --checkpoint best_v9m_dct.pt --data-dir data/ptb-xl
  python eval_ood.py --checkpoints best_v9m_dct.pt,best_baseline.pt --data-dir data/ptb-xl
  python eval_ood.py --checkpoint best_v9m_dct.pt --data-dir data/ptb-xl --nstdb-path data/nstdb
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from repnet.models import K, N_LEADS, BN_DIM, REPNET_MODELS
from repnet.init import compute_etf
from repnet.data import load_ptbxl, download_ptbxl
from repnet.eval import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── OOD data generation ──────────────────────────────────────────────────────

def generate_synthetic_ood(n_samples, seq_len=1000, sr=100, X_real=None,
                           seed=0):
    """Generate synthetic OOD ECG-like data for trustworthiness testing.

    All generation is seeded for reproducible evaluation.

    Args:
        n_samples: Number of OOD samples per type.
        seq_len: Sequence length (default 1000 = 10s at 100Hz).
        sr: Sample rate in Hz.
        X_real: Optional real ECG tensor for corrupted-ECG generation.
        seed: RNG seed for reproducible OOD data.

    Returns:
        Dict mapping OOD type name to (n_samples, N_LEADS, seq_len) tensor.
    """
    rng = torch.Generator().manual_seed(seed)
    t = torch.arange(seq_len, dtype=torch.float32) / sr

    ood_sets = {}

    # 1. Flat signals (electrode disconnection / lead-off)
    # Near-constant per lead with tiny noise — no cardiac structure.
    dc_offsets = torch.randn(n_samples, N_LEADS, 1, generator=rng)
    flat = dc_offsets.expand(-1, -1, seq_len).clone()
    flat += torch.randn(n_samples, N_LEADS, seq_len, generator=rng) * 0.01
    ood_sets['flat_leadoff'] = flat

    # 2. Gaussian noise (no cardiac structure)
    ood_sets['gaussian'] = torch.randn(n_samples, N_LEADS, seq_len,
                                       generator=rng)

    # 3. Muscle artifact (high-frequency bursts on baseline)
    muscle = torch.zeros(n_samples, N_LEADS, seq_len)
    for i in range(n_samples):
        n_bursts = torch.randint(3, 8, (1,), generator=rng).item()
        for _ in range(n_bursts):
            start = torch.randint(0, seq_len - 100, (1,),
                                  generator=rng).item()
            length = torch.randint(50, 200, (1,), generator=rng).item()
            end = min(start + length, seq_len)
            muscle[i, :, start:end] = torch.randn(
                N_LEADS, end - start, generator=rng) * 3.0
        # Add baseline wander
        muscle[i] += 0.2 * torch.sin(2 * np.pi * 0.3 * t).unsqueeze(0)
    ood_sets['muscle_artifact'] = muscle

    # 4. EMI (electromagnetic interference): 50/60 Hz + harmonics
    emi = torch.zeros(n_samples, N_LEADS, seq_len)
    for i in range(n_samples):
        freq = 50.0 if i % 2 == 0 else 60.0
        phase = torch.rand(N_LEADS, 1, generator=rng) * 2 * np.pi
        amp = 0.5 + torch.rand(N_LEADS, 1, generator=rng) * 2.0
        signal = amp * torch.sin(
            2 * np.pi * freq * t.unsqueeze(0) + phase)
        signal += 0.3 * amp * torch.sin(
            2 * np.pi * 2 * freq * t.unsqueeze(0) + phase)
        signal += 0.1 * amp * torch.sin(
            2 * np.pi * 3 * freq * t.unsqueeze(0) + phase)
        emi[i] = signal
    ood_sets['emi'] = emi

    # 5. Corrupted ECG (real ECG + heavy noise overlay)
    if X_real is not None and len(X_real) >= n_samples:
        corrupted = X_real[:n_samples].clone()
        noise_50hz = 2.0 * torch.sin(
            2 * np.pi * 50 * t).unsqueeze(0).unsqueeze(0)
        corrupted += noise_50hz.expand(n_samples, N_LEADS, -1)
        corrupted += torch.randn_like(corrupted) * 1.5
        ood_sets['corrupted_ecg'] = corrupted

    return ood_sets


# ── OOD scoring ──────────────────────────────────────────────────────────────

def compute_ood_scores(model, X, has_bottleneck=True, etf_ood_vertex=None,
                       batch_size=500):
    """Compute OOD detection scores for a batch of inputs.

    Args:
        model: Trained ECG model (eval mode).
        X: Input tensor (N, N_LEADS, seq_len).
        has_bottleneck: Whether model returns (logits, bottleneck_features).
        etf_ood_vertex: Optional ETF K+1 vertex for OOD projection scoring.
        batch_size: Inference batch size.

    Returns:
        Dict of score arrays, each shape (N,). Higher = more likely OOD
        for entropy, energy; lower for max_logit.
    """
    model.eval()
    all_logits, all_h = [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(DEVICE)
            if has_bottleneck:
                lo, h = model(batch)
                all_logits.append(lo.cpu())
                all_h.append(h.cpu())
            else:
                lo = model(batch)
                all_logits.append(lo.cpu())

    logits = torch.cat(all_logits, 0)
    H = torch.cat(all_h, 0) if all_h else None
    probs = torch.sigmoid(logits).numpy()

    scores = {}

    # Max logit: max(sigmoid(logit_k)). Higher = more confident = more ID.
    scores['max_logit'] = probs.max(axis=1)

    # Entropy: binary CE entropy per class, averaged. Higher = more uncertain.
    ent = -(probs * np.log(probs + 1e-10)
            + (1 - probs) * np.log(1 - probs + 1e-10))
    scores['entropy'] = ent.mean(axis=1)

    # Energy: -log(sum(exp(logits))). Higher (less negative) = more OOD.
    scores['energy'] = -torch.logsumexp(logits.float(), dim=1).numpy()

    # Bottleneck norm: L2 norm in bottleneck space. Extreme = OOD.
    if H is not None:
        scores['bn_norm'] = H.norm(dim=1).numpy()

    # ETF OOD projection: projection onto K+1 vertex. Higher = more OOD.
    if H is not None and etf_ood_vertex is not None:
        scores['ood_proj'] = (H @ etf_ood_vertex.unsqueeze(1)).squeeze(
            1).numpy()

    return scores


def compute_auroc(id_scores, ood_scores, score_name, higher_is_ood=True):
    """Compute AUROC for binary OOD detection using a given score.

    Args:
        id_scores: Score dict for in-distribution samples.
        ood_scores: Score dict for out-of-distribution samples.
        score_name: Key in score dicts to use.
        higher_is_ood: If True, higher score = more likely OOD.
            If False, negate scores before AUROC computation.

    Returns:
        AUROC float in [0, 1].
    """
    id_vals = id_scores[score_name]
    ood_vals = ood_scores[score_name]
    labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
    if higher_is_ood:
        combined = np.concatenate([id_vals, ood_vals])
    else:
        combined = np.concatenate([-id_vals, -ood_vals])
    return roc_auc_score(labels, combined)


# ── Clinical noise mixing ────────────────────────────────────────────────────

def load_nstdb_noise(nstdb_path):
    """Load real noise recordings from MIT-BIH Noise Stress Test Database.

    Returns dict mapping noise type to 1D numpy array.
    Noise types: baseline_wander, muscle_artifact_real, electrode_motion.
    """
    try:
        import wfdb
    except ImportError:
        print('WARNING: wfdb not installed. Cannot load NSTDB noise.', flush=True)
        print('  Install with: pip install wfdb', flush=True)
        return {}

    noise = {}
    for rec_name, label in [('bw', 'baseline_wander'),
                             ('ma', 'muscle_artifact_real'),
                             ('em', 'electrode_motion')]:
        rec_path = os.path.join(nstdb_path, rec_name)
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal  # (N_samples, n_channels)
            noise[label] = sig[:, 0].astype(np.float32)
            print(f'  Loaded {label}: {len(sig)} samples at {record.fs}Hz',
                  flush=True)
        except Exception as e:
            print(f'  Failed to load {rec_name}: {e}', flush=True)
    return noise


def mix_noise_at_snr(clean_ecg, noise_signal, snr_db, sr_clean=100,
                     sr_noise=360):
    """Mix clean ECG with real noise at a specified SNR level.

    Each lead gets a different segment of the noise recording (offset by
    lead index) to avoid unrealistic correlated noise across leads.

    SNR is calibrated per-sample: noise is scaled so that
    P_noise = P_signal / 10^(SNR/10).

    Args:
        clean_ecg: (N, n_leads, seq_len) tensor of clean ECGs.
        noise_signal: 1D numpy array of noise recording.
        snr_db: Target SNR in dB. 0 dB = equal power.
        sr_clean: Clean ECG sample rate (Hz).
        sr_noise: Noise recording sample rate (Hz).

    Returns:
        (N, n_leads, seq_len) tensor of noisy ECGs.
    """
    from scipy.signal import resample_poly
    from math import gcd

    if sr_noise != sr_clean:
        g = gcd(int(sr_clean), int(sr_noise))
        noise_resampled = resample_poly(
            noise_signal, int(sr_clean) // g, int(sr_noise) // g)
    else:
        noise_resampled = noise_signal.copy()

    N_seq = clean_ecg.shape[-1]
    n_leads = clean_ecg.shape[1]
    noise_len = len(noise_resampled)
    results = []

    for i in range(clean_ecg.shape[0]):
        noisy = clean_ecg[i].clone()
        sig_power = float(np.mean(clean_ecg[i].numpy() ** 2)) + 1e-10

        for ch in range(n_leads):
            offset = ((i * n_leads + ch) * N_seq) % max(noise_len - N_seq, 1)
            noise_seg = noise_resampled[offset:offset + N_seq]
            if len(noise_seg) < N_seq:
                noise_seg = np.tile(
                    noise_seg, (N_seq // len(noise_seg) + 1))[:N_seq]

            noise_power = float(np.mean(noise_seg ** 2)) + 1e-10
            target_noise_power = sig_power / (10 ** (snr_db / 10))
            scale = np.sqrt(target_noise_power / noise_power)

            noisy[ch] += torch.from_numpy(
                (noise_seg * scale).astype(np.float32))

        results.append(noisy)

    return torch.stack(results)


# ── Evaluation routines ──────────────────────────────────────────────────────

# Score configurations: (name, higher_is_ood)
SCORE_CONFIGS = [
    ('max_logit', False),   # lower max prob = more likely OOD
    ('entropy', True),      # higher entropy = more uncertain = more OOD
    ('energy', True),       # higher (less negative) energy = more OOD
]

BOTTLENECK_SCORES = [
    ('bn_norm', True),      # extreme bottleneck norm = OOD
    ('ood_proj', True),     # higher projection on OOD vertex = more OOD
]


def eval_synthetic_ood(model, model_name, X_te, has_bottleneck,
                       etf_ood_vertex, n_ood, seed=0):
    """Run synthetic OOD detection evaluation.

    Returns dict mapping OOD type to dict of {score_name: AUROC}.
    """
    n_id = min(n_ood, len(X_te))
    X_id = X_te[:n_id]

    ood_sets = generate_synthetic_ood(
        n_ood, seq_len=X_te.shape[2], X_real=X_te, seed=seed)

    id_scores = compute_ood_scores(
        model, X_id, has_bottleneck, etf_ood_vertex)

    score_configs = list(SCORE_CONFIGS)
    if has_bottleneck:
        score_configs.extend(BOTTLENECK_SCORES)

    results = {}
    print(f'\n{"="*60}', flush=True)
    print(f'Synthetic OOD Detection: {model_name}', flush=True)
    print(f'{"="*60}', flush=True)

    for ood_name, X_ood in ood_sets.items():
        ood_scores = compute_ood_scores(
            model, X_ood, has_bottleneck, etf_ood_vertex)

        ood_results = {}
        print(f'\n  {ood_name} (n={len(X_ood)}):', flush=True)
        for score_name, higher_is_ood in score_configs:
            if score_name not in ood_scores:
                continue
            auroc = compute_auroc(
                id_scores, ood_scores, score_name, higher_is_ood)
            ood_results[score_name] = round(float(auroc), 4)
            print(f'    {score_name:12s} AUROC: {auroc:.4f}', flush=True)

        results[ood_name] = ood_results

    return results


def eval_noise_robustness(model, model_name, X_te, Y_te, has_bottleneck,
                          nstdb_noise, snr_levels=(24, 18, 12, 6, 0)):
    """Evaluate classification AUC degradation under clinical noise.

    Returns dict with clean AUC and per-noise-type per-SNR AUC.
    """
    Y_np = Y_te.cpu().numpy()

    # Clean baseline
    model.eval()
    with torch.no_grad():
        lo_list = []
        for i in range(0, len(X_te), 500):
            batch = X_te[i:i + 500].to(DEVICE)
            out = model(batch)
            lo_list.append((out[0] if has_bottleneck else out).cpu())
        probs_clean = torch.sigmoid(torch.cat(lo_list, 0)).numpy()
        aucs = [roc_auc_score(Y_np[:, k], probs_clean[:, k])
                for k in range(K)
                if 0 < Y_np[:, k].sum() < len(Y_np)]
        clean_auc = float(np.mean(aucs))

    print(f'\n  {model_name}: clean AUC = {clean_auc:.4f}', flush=True)

    results = {'clean_auc': round(clean_auc, 4), 'noise': {}}

    for noise_name, noise_sig in nstdb_noise.items():
        results['noise'][noise_name] = {}
        for snr in snr_levels:
            X_noisy = mix_noise_at_snr(X_te, noise_sig, snr)
            with torch.no_grad():
                lo_list = []
                for i in range(0, len(X_noisy), 500):
                    batch = X_noisy[i:i + 500].to(DEVICE)
                    out = model(batch)
                    lo_list.append(
                        (out[0] if has_bottleneck else out).cpu())
                probs_noisy = torch.sigmoid(
                    torch.cat(lo_list, 0)).numpy()
                aucs_noisy = [
                    roc_auc_score(Y_np[:, k], probs_noisy[:, k])
                    for k in range(K)
                    if 0 < Y_np[:, k].sum() < len(Y_np)]
                noisy_auc = float(np.mean(aucs_noisy))

            drop = clean_auc - noisy_auc
            results['noise'][noise_name][f'snr_{snr}dB'] = {
                'auc': round(noisy_auc, 4),
                'drop': round(drop, 4),
            }
            print(f'    {noise_name} SNR={snr:2d}dB: '
                  f'{noisy_auc:.4f} (drop={drop:+.4f})', flush=True)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='OOD detection and noise robustness evaluation')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint', type=str,
                   help='Single checkpoint path')
    g.add_argument('--checkpoints', type=str,
                   help='Comma-separated checkpoint paths for comparison')
    p.add_argument('--model', type=str, default=None,
                   help='Model class (inferred from checkpoint if omitted)')
    p.add_argument('--data-dir', default='data/ptb-xl',
                   help='PTB-XL data directory')
    p.add_argument('--nstdb-path', type=str, default=None,
                   help='MIT-BIH NSTDB directory for clinical noise eval')
    p.add_argument('--n-ood', type=int, default=500,
                   help='Number of OOD samples per type')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for OOD data generation')
    p.add_argument('--output', type=str, default=None,
                   help='Output JSON path (default: results/ood_results.json)')
    return p.parse_args()


def main():
    args = parse_args()

    # Build checkpoint list
    if args.checkpoints:
        ckpt_paths = [p.strip() for p in args.checkpoints.split(',')]
    else:
        ckpt_paths = [args.checkpoint]

    # Load data
    print('Loading PTB-XL data...', flush=True)
    download_ptbxl(args.data_dir)
    _, _, _, _, X_te, Y_te = load_ptbxl(args.data_dir)

    # Compute ETF OOD vertex (K+1 row of the K+1 simplex in BN_DIM space)
    etf = compute_etf(K, BN_DIM)
    ood_vertex = etf[K]  # the (K+1)-th vertex, orthogonal to all K classes

    # Load clinical noise if available
    nstdb_noise = {}
    if args.nstdb_path:
        print('Loading NSTDB noise recordings...', flush=True)
        nstdb_noise = load_nstdb_noise(args.nstdb_path)
        if not nstdb_noise:
            print('WARNING: No NSTDB noise loaded. '
                  'Skipping clinical noise evaluation.', flush=True)

    all_results = {}

    for ckpt_path in ckpt_paths:
        print(f'\n\nLoading {ckpt_path}...', flush=True)
        try:
            model, model_name, ckpt = load_model(
                ckpt_path, args.model, DEVICE)
        except Exception as e:
            print(f'  Failed to load: {e}', flush=True)
            continue

        has_bn = model_name in REPNET_MODELS
        etf_v = ood_vertex if has_bn else None
        params = sum(p.numel() for p in model.parameters())
        print(f'  Model: {model_name} ({params:,} params)', flush=True)

        model_results = {}

        # 1. Synthetic OOD detection
        model_results['synthetic_ood'] = eval_synthetic_ood(
            model, model_name, X_te, has_bn, etf_v,
            args.n_ood, seed=args.seed)

        # 2. Noise robustness (if NSTDB available)
        if nstdb_noise:
            print(f'\n{"="*60}', flush=True)
            print(f'Noise Robustness: {model_name}', flush=True)
            print(f'{"="*60}', flush=True)
            model_results['noise_robustness'] = eval_noise_robustness(
                model, model_name, X_te, Y_te, has_bn, nstdb_noise)

        all_results[os.path.basename(ckpt_path)] = model_results

    # Summary
    if len(all_results) > 1:
        print(f'\n\n{"="*60}', flush=True)
        print('OOD COMPARISON (best AUROC per type)', flush=True)
        print(f'{"="*60}', flush=True)
        for name, results in all_results.items():
            print(f'\n  {name}:', flush=True)
            for ood_type, scores in results.get('synthetic_ood', {}).items():
                best = max(scores.values())
                best_method = max(scores, key=scores.get)
                print(f'    {ood_type:25s} {best:.4f} ({best_method})',
                      flush=True)

    # Save
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = args.output or os.path.join(results_dir, 'ood_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {output_path}', flush=True)


if __name__ == '__main__':
    main()
