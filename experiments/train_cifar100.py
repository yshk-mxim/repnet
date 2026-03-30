"""CIFAR-100 experiment — structured orthogonal init vs Kaiming.

Usage:
  python train_cifar100.py --init dct              # DCT init (default)
  python train_cifar100.py --init hadamard         # Hadamard init
  python train_cifar100.py --init hartley          # Hartley init
  python train_cifar100.py --init sinusoidal       # Sinusoidal init
  python train_cifar100.py --init kaiming --seed 42  # Kaiming baseline
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from repnet.models_cifar import ResNet18CIFAR
from repnet.eval import model_md5
from repnet.init import dct_init_2d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHI = (1 + 5**0.5) / 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--init', default='dct',
                   choices=['dct', 'hadamard', 'hartley', 'sinusoidal', 'kaiming'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default=None)
    return p.parse_args()




def main():
    args = parse_args()
    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    model = ResNet18CIFAR(num_classes=100).to(DEVICE)

    if args.init != 'kaiming':
        dct_init_2d(model, fixup_scale=0.01, basis=args.init)
    else:
        torch.manual_seed(args.seed)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    init_md5 = model_md5(model)
    print(f'=== CIFAR-100 ({args.init} init, seed={args.seed}) ===', flush=True)
    print(f'Init MD5: {init_md5}', flush=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=500, shuffle=False, num_workers=0)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0
    best_ep = -1
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        torch.manual_seed(args.seed * 10000 + ep * 1000)

        indices = np.arange(len(train_set))
        positions = np.mod(indices * PHI + ep * PHI * PHI, 1.0)
        order = np.argsort(positions).tolist()

        epoch_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.bs, sampler=order,
            num_workers=2, pin_memory=True,
        )

        for X_batch, Y_batch in epoch_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = F.cross_entropy(logits, Y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_ep = ep
            print(f'  Ep {ep}: test_acc={acc:.4f} (best={best_acc:.4f}@{best_ep}) '
                  f'({time.time()-t0:.0f}s)', flush=True)

    final_md5 = model_md5(model)
    print(f'Final MD5: {final_md5}', flush=True)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    result = {
        'dataset': 'cifar100',
        'init': args.init,
        'seed': args.seed,
        'best_acc': round(best_acc, 4),
        'best_ep': best_ep,
        'init_md5': init_md5,
        'final_md5': final_md5,
        'time': round(time.time() - t0, 1),
        'params': sum(p.numel() for p in model.parameters()),
    }
    out_path = os.path.join(results_dir, f'cifar100_{args.init}_s{args.seed}_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}', flush=True)


if __name__ == '__main__':
    main()
