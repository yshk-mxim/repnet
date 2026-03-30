"""ECG RepNet training — paper reproduction.

Usage:
  python train_ecg.py --model v9m --data-dir ./data/ptb-xl
  python train_ecg.py --model conformer --batch-order golden
  python train_ecg.py --model baseline --data-dir ./data/ptb-xl
"""
import os
import sys
import time
import argparse
import json

import torch
import torch.nn.functional as F

# Determinism
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Add parent to path for repnet package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from repnet.models import (K, BN_REG_WEIGHT, MODEL_CLASSES, REPNET_MODELS)
from repnet.eval import model_md5
from repnet.init import deterministic_init
from repnet.data import load_ptbxl, download_ptbxl
from repnet.batch import build_seeded_batches, build_golden_ratio_batches
from repnet.metrics import eval_auc, compute_all_metrics, print_metrics, save_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser(description='RepNet ECG Training')
    p.add_argument('--model', default='v9m', choices=list(MODEL_CLASSES.keys()))
    p.add_argument('--data-dir', default='data/ptb-xl')
    p.add_argument('--name', default=None, help='Experiment name (default: model name)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=85)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--batch-order', default='seeded', choices=['seeded', 'golden'])
    p.add_argument('--det-init', action='store_true', default=True,
                   help='Use DCT initialization (default: True)')
    p.add_argument('--no-det-init', dest='det_init', action='store_false',
                   help='Use Kaiming initialization instead of DCT')
    p.add_argument('--fixup-scale', type=float, default=0.01)
    p.add_argument('--mixed-bases', action='store_true',
                   help='Use DCT/Hadamard/Hartley per stage')
    p.add_argument('--basis', default='dct',
                   choices=['dct', 'hadamard', 'hartley', 'sinusoidal'],
                   help='Basis for deterministic init (default: dct)')
    p.add_argument('--class-weight', default='sqrt', choices=['sqrt', 'none'])
    p.add_argument('--device', default=None)
    return p.parse_args()



def train(args, model, X_tr, Y_tr, X_va, Y_va, X_te, Y_te):
    is_repnet = args.model in REPNET_MODELS
    has_bn = is_repnet
    name = args.name or args.model
    ckpt_path = f'best_{name}.pt'
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Golden ratio or seeded batching
    use_golden = args.batch_order == 'golden'
    if not use_golden:
        batches = build_seeded_batches(Y_tr, args.bs, args.epochs, seed=args.seed)
        batches_per_epoch = len(batches) // args.epochs
    else:
        has_label = Y_tr.sum(dim=1) > 0
        batches_per_epoch = int(has_label.sum().item()) // args.bs

    params = sum(p.numel() for p in model.parameters())
    init_hash = model_md5(model)
    print(f'Model: {args.model} | Params: {params:,} | Device: {DEVICE}', flush=True)
    print(f'Batch order: {args.batch_order} | Init MD5: {init_hash}', flush=True)

    # Class-weighted loss
    pos_weight = None
    if args.class_weight == 'sqrt':
        class_counts = Y_tr.sum(dim=0).clamp(min=1).float()
        pos_weight = (Y_tr.shape[0] / class_counts).sqrt().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_auc = 0
    best_ep = -1
    t0 = time.time()
    log_entries = []
    batch_idx = 0

    for ep in range(args.epochs):
        if use_golden:
            epoch_batches = build_golden_ratio_batches(X_tr, Y_tr, args.bs, ep)
            batches_per_epoch = len(epoch_batches)

        model.train()
        epoch_loss = 0
        n = 0
        for bi in range(batches_per_epoch):
            if use_golden:
                idx = epoch_batches[bi].to(DEVICE)
            else:
                idx = batches[batch_idx].to(DEVICE)
                batch_idx += 1

            out = model(X_tr[idx])
            lo = out[0] if is_repnet else out
            lo = lo.clamp(-50, 50)
            loss = F.binary_cross_entropy_with_logits(lo, Y_tr[idx], pos_weight=pos_weight)

            if is_repnet:
                h = out[1]
                if h.shape[1] > K:
                    loss += BN_REG_WEIGHT * h[:, K:].pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            auc = eval_auc(model, X_va, Y_va, is_repnet=is_repnet)
            if auc > best_auc:
                best_auc = auc
                best_ep = ep
                torch.save({
                    'model_state': model.state_dict(),
                    'model_class': args.model,
                    'epoch': ep,
                    'val_auc': best_auc,
                    'args': vars(args),
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

    model.eval()
    test_metrics = compute_all_metrics(model, X_te, Y_te, has_bottleneck=has_bn)
    test_metrics['params'] = params
    test_metrics['best_val'] = best_auc
    test_metrics['best_ep'] = best_ep
    test_metrics['time'] = time.time() - t0
    test_metrics['init_md5'] = init_hash
    test_metrics['final_md5'] = model_md5(model)
    test_metrics['seed'] = args.seed
    test_metrics['batch_order'] = args.batch_order

    print_metrics(f'{name} TEST', test_metrics)
    metrics_path = os.path.join(results_dir, f'{name}_metrics.json')
    save_metrics(test_metrics, metrics_path)
    print(f'Saved: {metrics_path}', flush=True)

    log_path = os.path.join(results_dir, f'{name}_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2, default=float)

    return test_metrics


def main():
    args = parse_args()
    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)

    # Download and load data
    download_ptbxl(args.data_dir)
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te = load_ptbxl(args.data_dir)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)
    X_va, Y_va = X_va.to(DEVICE), Y_va.to(DEVICE)
    X_te, Y_te = X_te.to(DEVICE), Y_te.to(DEVICE)

    # Build model (Conformer uses base_ch=40 per paper)
    model_cls = MODEL_CLASSES[args.model]
    if args.model == 'conformer':
        model = model_cls(base_ch=40, etf_init=True).to(DEVICE)
    elif args.model in REPNET_MODELS:
        model = model_cls(etf_init=True).to(DEVICE)
    else:
        model = model_cls().to(DEVICE)

    # DCT initialization (applies to all architectures when --det-init is set)
    if args.det_init:
        deterministic_init(model, fixup_scale=args.fixup_scale,
                          mixed_bases=args.mixed_bases, basis=args.basis)
        print(f'Det init applied (basis={args.basis}, fixup={args.fixup_scale}, '
              f'mixed={args.mixed_bases})', flush=True)

    # torch.compile is opt-in: set TORCH_COMPILE=1 to enable.
    # Disabled by default — can cause numerical issues with multi-run workflows.
    if hasattr(torch, 'compile') and os.environ.get('TORCH_COMPILE') == '1':
        try:
            compiled = torch.compile(model)
            compiled(X_tr[:2])  # warm-up
            model = compiled
            print('torch.compile enabled', flush=True)
        except RuntimeError as e:
            print(f'torch.compile unavailable: {e}', flush=True)

    train(args, model, X_tr, Y_tr, X_va, Y_va, X_te, Y_te)


if __name__ == '__main__':
    main()
