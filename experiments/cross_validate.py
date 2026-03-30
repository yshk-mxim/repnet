"""3-fold cross-validation for variance decomposition.

Rotates test fold across folds 10, 9, 8 to estimate fold variance.
Reports: init variance (zero for DCT), batch variance, fold variance.

Usage:
  python cross_validate.py --model v9m --seeds 42 123 456
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from repnet.models import MODEL_CLASSES, REPNET_MODELS, K, BN_REG_WEIGHT
from repnet.init import deterministic_init
from repnet.data import load_ptbxl, download_ptbxl
from repnet.batch import build_seeded_batches
from repnet.metrics import eval_auc

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CV_SPLITS = [
    {'name': 'fold10', 'train': [1, 2, 3, 4, 5, 6, 7, 8], 'val': 9, 'test': 10},
    {'name': 'fold9',  'train': [1, 2, 3, 4, 5, 6, 7, 10], 'val': 8, 'test': 9},
    {'name': 'fold8',  'train': [1, 2, 3, 4, 5, 6, 7, 9], 'val': 10, 'test': 8},
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='v9m', choices=list(MODEL_CLASSES.keys()))
    p.add_argument('--data-dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ptb-xl'))
    p.add_argument('--epochs', type=int, default=85)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    p.add_argument('--mixed-bases', action='store_true',
                   help='Use DCT/Hadamard/Hartley per stage')
    return p.parse_args()



def train_one(args, seed, X_tr, Y_tr, X_te, Y_te):
    model_cls = MODEL_CLASSES[args.model]
    is_repnet = args.model in REPNET_MODELS

    if args.model == 'conformer':
        model = model_cls(base_ch=40, etf_init=True).to(DEVICE)
    elif is_repnet:
        model = model_cls(etf_init=True).to(DEVICE)
    else:
        model = model_cls().to(DEVICE)

    if is_repnet:
        deterministic_init(model, fixup_scale=0.01, mixed_bases=args.mixed_bases)

    batches = build_seeded_batches(Y_tr, args.bs, args.epochs, seed=seed)
    batches_per_epoch = len(batches) // args.epochs

    class_counts = Y_tr.sum(dim=0).clamp(min=1).float()
    pos_weight = (Y_tr.shape[0] / class_counts).sqrt().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    batch_idx = 0

    for ep in range(args.epochs):
        model.train()
        for bi in range(batches_per_epoch):
            idx = batches[batch_idx].to(DEVICE)
            batch_idx += 1
            out = model(X_tr[idx])
            lo = out[0] if is_repnet else out
            lo = lo.clamp(-50, 50)
            loss = F.binary_cross_entropy_with_logits(
                lo, Y_tr[idx], pos_weight=pos_weight)
            if is_repnet:
                h = out[1]
                if h.shape[1] > K:
                    loss += BN_REG_WEIGHT * h[:, K:].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    test_auc = eval_auc(model, X_te, Y_te, is_repnet)
    return test_auc


def main():
    args = parse_args()
    print(f'=== 3-Fold CV: {args.model}, {len(args.seeds)} seeds ===', flush=True)

    download_ptbxl(args.data_dir)
    # Load all data with fold info
    X_tr, Y_tr, X_va, Y_va, X_te, Y_te, folds = load_ptbxl(
        args.data_dir, return_folds=True)
    # Recombine all data for fold rotation
    X_all = torch.cat([X_tr, X_va, X_te], dim=0)
    Y_all = torch.cat([Y_tr, Y_va, Y_te], dim=0)

    all_results = []
    for split in CV_SPLITS:
        train_mask = torch.zeros(len(folds), dtype=torch.bool)
        for f in split['train']:
            train_mask |= (folds == f)
        test_mask = (folds == split['test'])

        X_tr_fold = X_all[train_mask].to(DEVICE)
        Y_tr_fold = Y_all[train_mask].to(DEVICE)
        X_te_fold = X_all[test_mask].to(DEVICE)
        Y_te_fold = Y_all[test_mask].to(DEVICE)

        fold_results = []
        for seed in args.seeds:
            t0 = time.time()
            auc = train_one(args, seed, X_tr_fold, Y_tr_fold,
                            X_te_fold, Y_te_fold)
            elapsed = time.time() - t0
            fold_results.append({
                'fold': split['name'], 'seed': seed,
                'test_auc': round(auc, 4), 'time': round(elapsed, 1)
            })
            print(f'  {split["name"]} seed {seed}: AUC={auc:.4f} ({elapsed:.0f}s)',
                  flush=True)
        all_results.extend(fold_results)

    # Compute variance decomposition
    aucs_by_fold = {}
    for r in all_results:
        aucs_by_fold.setdefault(r['fold'], []).append(r['test_auc'])

    fold_means = [np.mean(v) for v in aucs_by_fold.values()]
    all_aucs = [r['test_auc'] for r in all_results]

    summary = {
        'model': args.model,
        'n_folds': len(CV_SPLITS),
        'n_seeds': len(args.seeds),
        'cv_mean': round(np.mean(fold_means), 4),
        'cv_std': round(np.std(fold_means, ddof=1), 4) if len(fold_means) > 1 else 0.0,
        'overall_mean': round(np.mean(all_aucs), 4),
        'overall_std': round(np.std(all_aucs, ddof=1), 4) if len(all_aucs) > 1 else 0.0,
        'per_fold': {k: {'mean': round(np.mean(v), 4),
                         'std': round(np.std(v, ddof=1), 4) if len(v) > 1 else 0.0}
                     for k, v in aucs_by_fold.items()},
        'runs': all_results,
    }
    print(f'\nCV Mean: {summary["cv_mean"]:.4f} +/- {summary["cv_std"]:.4f}',
          flush=True)
    for fold_name, stats in summary['per_fold'].items():
        print(f'  {fold_name}: {stats["mean"]:.4f} +/- {stats["std"]:.4f}',
              flush=True)

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'cv_{args.model}_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
