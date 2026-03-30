"""GPU worker for basis comparison — load data once, compile once, loop runs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/basis_worker.py \
      --data-dir data/ptb-xl --runs-file /tmp/gpu0_runs.json
"""
import os
import sys
import time
import argparse
import json

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()  # Prevent NVIDIA nightly auto-compilation
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# Determinism
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from repnet.models import K, BN_REG_WEIGHT, MODEL_CLASSES
from repnet.eval import model_md5
from repnet.init import deterministic_init
from repnet.data import load_ptbxl, download_ptbxl
from repnet.batch import build_seeded_batches
from repnet.metrics import compute_all_metrics, print_metrics, save_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def train_one(name, seed, basis, det_init, epochs, bs, lr, class_weight,
              fixup_scale, use_compile,
              X_tr, Y_tr, X_va, Y_va, X_te, Y_te, results_dir):
    """Train a single run, reusing loaded data. Fresh model each run."""
    result_file = os.path.join(results_dir, f'{name}_metrics.json')
    if os.path.exists(result_file):
        print(f'SKIP (exists): {name}', flush=True)
        return

    # Fresh model each run — identical to per-process approach
    model = MODEL_CLASSES['conformer'](base_ch=40, etf_init=True).to(DEVICE)

    # Apply deterministic init if requested
    if det_init:
        deterministic_init(model, fixup_scale=fixup_scale,
                          mixed_bases=False, basis=basis)

    # torch.compile (inductor cache makes 2nd+ compile near-instant)
    if use_compile:
        try:
            model = torch.compile(model)
            model(X_tr[:2])  # warm-up
        except RuntimeError as e:
            print(f'torch.compile unavailable: {e}', flush=True)

    init_hash = model_md5(model)
    params = sum(p.numel() for p in model.parameters())
    print(f'\n[{name}] basis={basis} seed={seed} det_init={det_init} | '
          f'Init MD5: {init_hash}', flush=True)

    # Build batches for this seed
    batches = build_seeded_batches(Y_tr, bs, epochs, seed=seed)
    batches_per_epoch = len(batches) // epochs

    # Class-weighted loss
    pos_weight = None
    if class_weight == 'sqrt':
        class_counts = Y_tr.sum(dim=0).clamp(min=1).float()
        pos_weight = (Y_tr.shape[0] / class_counts).sqrt().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc = 0
    best_ep = -1
    ckpt_path = f'best_{name}.pt'
    t0 = time.time()
    log_entries = []
    batch_idx = 0

    for ep in range(epochs):
        model.train()
        epoch_loss = 0
        n = 0
        for bi in range(batches_per_epoch):
            idx = batches[batch_idx].to(DEVICE)
            batch_idx += 1
            out = model(X_tr[idx])
            lo = out[0] if isinstance(out, tuple) else out
            lo = lo.clamp(-50, 50)
            loss = F.binary_cross_entropy_with_logits(lo, Y_tr[idx],
                                                       pos_weight=pos_weight)
            # Bottleneck regularization
            if isinstance(out, tuple) and out[1].shape[1] > K:
                loss += BN_REG_WEIGHT * out[1][:, K:].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()

        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                parts = []
                for i in range(0, len(X_va), 500):
                    out = model(X_va[i:i+500])
                    parts.append(out[0] if isinstance(out, tuple) else out)
                lo = torch.cat(parts, 0)
                probs = torch.sigmoid(lo).cpu().numpy()
                Y_np = Y_va.cpu().numpy()
                aucs = [roc_auc_score(Y_np[:, i], probs[:, i])
                        for i in range(K)
                        if Y_np[:, i].sum() > 0 and Y_np[:, i].sum() < len(Y_np)]
                auc = np.mean(aucs) if aucs else 0.0

            if auc > best_auc:
                best_auc = auc
                best_ep = ep
                torch.save({
                    'model_state': model.state_dict(),
                    'model_class': 'conformer',
                    'epoch': ep,
                    'val_auc': best_auc,
                }, ckpt_path)
            entry = {'epoch': ep, 'val_auc': auc, 'loss': epoch_loss / max(n, 1),
                     'time': time.time() - t0}
            log_entries.append(entry)
            print(f'  Ep {ep}: val_AUC={auc:.3f} (best={best_auc:.3f}@{best_ep}) '
                  f'({time.time()-t0:.0f}s)', flush=True)

    # Load best and evaluate on test
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

    test_metrics = compute_all_metrics(model, X_te, Y_te, has_bottleneck=True)
    test_metrics['params'] = params
    test_metrics['best_val'] = best_auc
    test_metrics['best_ep'] = best_ep
    test_metrics['time'] = time.time() - t0
    test_metrics['init_md5'] = init_hash
    test_metrics['final_md5'] = model_md5(model)
    test_metrics['seed'] = seed
    test_metrics['batch_order'] = 'seeded'

    print_metrics(f'{name} TEST', test_metrics)
    save_metrics(test_metrics, result_file)
    print(f'Saved: {result_file}', flush=True)

    log_path = os.path.join(results_dir, f'{name}_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2, default=float)

    # Clean checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/ptb-xl')
    p.add_argument('--runs-file', required=True,
                   help='JSON file with list of {name, seed, basis, det_init}')
    p.add_argument('--epochs', type=int, default=85)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--fixup-scale', type=float, default=0.01)
    p.add_argument('--class-weight', default='sqrt')
    args = p.parse_args()

    # Load data ONCE — stays in GPU memory for all runs
    download_ptbxl(args.data_dir)
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = load_ptbxl(args.data_dir)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_va, Y_va = X_va.to(DEVICE), Y_va.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)
    print(f'Data loaded to {DEVICE}', flush=True)

    # torch.compile disabled — fresh model per run means recompile overhead,
    # and compile can cause subtle numerical issues (GPU 2 collapse, GPU 3 tuple error).
    # Eager mode is ~8s/epoch vs ~6s compiled — acceptable for correctness.
    use_compile = False
    print('Using eager mode (no torch.compile)', flush=True)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load run list
    with open(args.runs_file) as f:
        runs = json.load(f)

    print(f'\n=== Worker starting: {len(runs)} runs ===', flush=True)
    t_total = time.time()

    for i, run in enumerate(runs):
        print(f'\n--- Run {i+1}/{len(runs)} ---', flush=True)
        train_one(
            name=run['name'],
            seed=run['seed'],
            basis=run.get('basis', 'dct'),
            det_init=run.get('det_init', True),
            epochs=args.epochs,
            bs=args.bs,
            lr=args.lr,
            class_weight=args.class_weight,
            fixup_scale=args.fixup_scale,
            use_compile=use_compile,
            X_tr=X_tr, Y_tr=Y_tr,
            X_va=X_va, Y_va=Y_va,
            X_te=X_te, Y_te=Y_te,
            results_dir=results_dir,
        )

    elapsed = time.time() - t_total
    print(f'\n=== Worker done: {len(runs)} runs in {elapsed/60:.1f} min ===',
          flush=True)


if __name__ == '__main__':
    main()
