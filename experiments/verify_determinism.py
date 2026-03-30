"""Verify bit-identical determinism: train twice, compare MD5.

Usage:
  python verify_determinism.py --model v9m --epochs 10
  python verify_determinism.py --model v9m --epochs 10 --batch-order golden
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from repnet.models import MODEL_CLASSES, REPNET_MODELS, K, BN_REG_WEIGHT
from repnet.eval import model_md5
from repnet.init import deterministic_init
from repnet.data import load_ptbxl, download_ptbxl
from repnet.batch import build_seeded_batches, build_golden_ratio_batches

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='v9m', choices=list(MODEL_CLASSES.keys()))
    p.add_argument('--data-dir', default='data/ptb-xl')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-order', default='seeded', choices=['seeded', 'golden'])
    return p.parse_args()




def train_run(args, X_tr, Y_tr):
    """Single training run. Returns init and final MD5."""
    model_cls = MODEL_CLASSES[args.model]
    is_repnet = args.model in REPNET_MODELS

    if args.model == 'conformer':
        model = model_cls(base_ch=40, etf_init=True).to(DEVICE)
        deterministic_init(model, fixup_scale=0.01)
    elif is_repnet:
        model = model_cls(etf_init=True).to(DEVICE)
        deterministic_init(model, fixup_scale=0.01)
    else:
        model = model_cls().to(DEVICE)

    init_md5 = model_md5(model)

    use_golden = args.batch_order == 'golden'
    if not use_golden:
        batches = build_seeded_batches(Y_tr, args.bs, args.epochs, seed=args.seed)
        batches_per_epoch = len(batches) // args.epochs

    # Class-weighted loss
    class_counts = Y_tr.sum(dim=0).clamp(min=1).float()
    pos_weight = (Y_tr.shape[0] / class_counts).sqrt().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_idx = 0

    for ep in range(args.epochs):
        if use_golden:
            epoch_batches = build_golden_ratio_batches(X_tr, Y_tr, args.bs, ep)
            batches_per_epoch = len(epoch_batches)

        model.train()
        for bi in range(batches_per_epoch):
            if use_golden:
                idx = epoch_batches[bi].to(DEVICE)
            else:
                idx = batches[batch_idx].to(DEVICE)
                batch_idx += 1

            out = model(X_tr[idx])
            lo = out[0] if is_repnet else out
            loss = F.binary_cross_entropy_with_logits(lo, Y_tr[idx], pos_weight=pos_weight)
            if is_repnet:
                h = out[1]
                if h.shape[1] > K:
                    loss += BN_REG_WEIGHT * h[:, K:].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return init_md5, model_md5(model)


def main():
    args = parse_args()
    print(f'=== Determinism Verification: {args.model}, {args.epochs} epochs ===', flush=True)
    print(f'Batch order: {args.batch_order}', flush=True)

    download_ptbxl(args.data_dir)
    X_tr, Y_tr, _, _, _, _ = load_ptbxl(args.data_dir)
    X_tr, Y_tr = X_tr.to(DEVICE), Y_tr.to(DEVICE)

    print('\n--- Run 1 ---', flush=True)
    init1, final1 = train_run(args, X_tr, Y_tr)
    print(f'Init MD5:  {init1}', flush=True)
    print(f'Final MD5: {final1}', flush=True)

    print('\n--- Run 2 ---', flush=True)
    init2, final2 = train_run(args, X_tr, Y_tr)
    print(f'Init MD5:  {init2}', flush=True)
    print(f'Final MD5: {final2}', flush=True)

    print('\n=== RESULT ===', flush=True)
    init_match = init1 == init2
    final_match = final1 == final2
    print(f'Init identical:  {init_match}', flush=True)
    print(f'Final identical: {final_match}', flush=True)

    if final_match:
        print('\nBIT-IDENTICAL TRAINING VERIFIED', flush=True)
    else:
        print('\nWARNING: MD5 MISMATCH — not deterministic', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
