"""MedMNIST experiments — DCT init vs Kaiming on medical image benchmarks.

Runs both bottleneck and non-bottleneck ResNet-18 variants on PathMNIST and DermaMNIST.
Applies golden ratio batching for deterministic ordering.

Usage:
  python train_medmnist.py --dataset pathmnist --init dct --seed 42
  python train_medmnist.py --dataset dermamnist --init kaiming --seed 42 --no-bottleneck
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
from torchvision import transforms
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from repnet.init import dct_init_2d
from repnet.eval import model_md5
from repnet.models import BN_REG_WEIGHT
from repnet.models_cifar import IntermediateBottleneck2D, BasicBlock

# Determinism: configured after argparse in main() via _setup_determinism().
_GOLDEN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHI = (1 + 5**0.5) / 2


class GoldenRatioSampler(torch.utils.data.Sampler):
    """Persistent sampler with golden ratio ordering — no DataLoader recreation."""
    def __init__(self, n_samples):
        self.n = n_samples
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = np.arange(self.n)
        positions = np.mod(indices * PHI + self.epoch * PHI * PHI, 1.0)
        return iter(np.argsort(positions).tolist())

    def __len__(self):
        return self.n

DATASET_INFO = {
    'pathmnist':   {'n_classes': 9,  'n_channels': 3, 'img_size': 28, 'multi_label': False},
    'dermamnist':  {'n_classes': 7,  'n_channels': 3, 'img_size': 28, 'multi_label': False},
    'bloodmnist':  {'n_classes': 8,  'n_channels': 3, 'img_size': 28, 'multi_label': False},
    'breastmnist': {'n_classes': 2,  'n_channels': 1, 'img_size': 28, 'multi_label': False},
    'retinamnist': {'n_classes': 5,  'n_channels': 3, 'img_size': 28, 'multi_label': False},
    'organamnist': {'n_classes': 11, 'n_channels': 1, 'img_size': 28, 'multi_label': False},
    'organcmnist': {'n_classes': 11, 'n_channels': 1, 'img_size': 28, 'multi_label': False},
    'chestmnist':  {'n_classes': 14, 'n_channels': 1, 'img_size': 28, 'multi_label': True},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='pathmnist',
                   choices=list(DATASET_INFO.keys()))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--init', default='dct',
                   choices=['dct', 'hadamard', 'hartley', 'sinusoidal', 'kaiming'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-bottleneck', action='store_true',
                   help='Use standard ResNet-18 without bottleneck')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--batch-order', default='golden', choices=['golden', 'seeded'],
                   help='Batch ordering: golden (deterministic) or seeded (random shuffle)')
    p.add_argument('--no-deterministic', action='store_true',
                   help='Disable deterministic mode (faster but non-reproducible)')
    return p.parse_args()




# --- Architecture ---
# IntermediateBottleneck2D and BasicBlock imported from repnet.models_cifar


class ResNet18MedMNIST(nn.Module):
    """ResNet-18 for 28x28 medical images. Modified stem (3x3, no maxpool).
    Optional intermediate bottleneck at each stage (like RepNet/CIFAR design).
    """
    def __init__(self, num_classes, n_channels=3, bn_dim=None):
        super().__init__()
        self.use_bottleneck = bn_dim is not None
        self.bn_dim = bn_dim
        # Modified stem for 28x28: 3x3 conv stride 1 (no 7x7 + maxpool)
        self.conv1 = nn.Conv2d(n_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        if self.use_bottleneck:
            self.ibn1 = IntermediateBottleneck2D(64, bn_dim)
            self.ibn2 = IntermediateBottleneck2D(128, bn_dim)
            self.ibn3 = IntermediateBottleneck2D(256, bn_dim)
            self.ibn4 = IntermediateBottleneck2D(512, bn_dim)
            self.bn_linear = nn.Linear(512, bn_dim)
            self.bn_act = nn.GELU()
            self.fc = nn.Linear(bn_dim, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.use_bottleneck:
            out = self.ibn1(out)
        out = self.layer2(out)
        if self.use_bottleneck:
            out = self.ibn2(out)
        out = self.layer3(out)
        if self.use_bottleneck:
            out = self.ibn3(out)
        out = self.layer4(out)
        if self.use_bottleneck:
            out = self.ibn4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        if self.use_bottleneck:
            h = self.bn_act(self.bn_linear(out))
            return self.fc(h), h
        else:
            return self.fc(out), out


# 2D DCT initialization imported from repnet.init


def _setup_determinism(golden):
    """Configure torch determinism settings based on batch order."""
    global _GOLDEN
    _GOLDEN = golden
    if _GOLDEN:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    _setup_determinism(args.batch_order == 'golden')
    info = DATASET_INFO[args.dataset]
    K = info['n_classes']
    multi_label = info.get('multi_label', False)
    bn_dim = None if args.no_bottleneck else (K + 2)
    arch_name = 'resnet18' if args.no_bottleneck else f'resnet18_bn{bn_dim}'

    print(f'=== MedMNIST: {args.dataset} | {arch_name} | {args.init} | seed={args.seed} ===',
          flush=True)

    # Load dataset via medmnist
    import medmnist
    from medmnist import INFO

    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # MedMNIST provides 28x28 PIL images
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * info['n_channels'], [0.5] * info['n_channels']),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * info['n_channels'], [0.5] * info['n_channels']),
    ])

    dataset_cls_name = INFO[args.dataset]['python_class']
    DataClass = getattr(medmnist, dataset_cls_name)

    train_set = DataClass(split='train', transform=transform_train,
                          download=True, root=data_dir)
    test_set = DataClass(split='test', transform=transform_test,
                         download=True, root=data_dir)
    print(f'Train: {len(train_set)}, Test: {len(test_set)}, Classes: {K}', flush=True)

    # Test loader (deterministic)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=500, shuffle=False, num_workers=0)

    # Batch ordering: golden ratio (deterministic) or seeded shuffle.
    # num_workers=0 for 28x28 images: data loading is negligible,
    # avoids memory/CPU contention when running multiple jobs in parallel.
    if args.batch_order == 'golden':
        sampler = GoldenRatioSampler(len(train_set))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.bs, sampler=sampler,
            num_workers=0, pin_memory=True,
        )
    else:
        g = torch.Generator().manual_seed(args.seed)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.bs, shuffle=True, generator=g,
            num_workers=0, pin_memory=True,
        )

    # Build model with channels_last for faster Conv2d
    model = ResNet18MedMNIST(
        num_classes=K, n_channels=info['n_channels'], bn_dim=bn_dim
    ).to(DEVICE, memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    print(f'{arch_name}: {params:,} params', flush=True)

    # Enable TF32 for seeded runs (tiny numerical diff, irrelevant for mean±std)
    if not _GOLDEN:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # torch.compile is opt-in: set TORCH_COMPILE=1 to enable.
    # Disabled by default — can cause numerical issues with multi-run workflows.
    if hasattr(torch, 'compile') and os.environ.get('TORCH_COMPILE') == '1':
        try:
            compiled = torch.compile(model)
            with torch.no_grad():
                compiled(torch.randn(1, info['n_channels'], 28, 28,
                                     device=DEVICE).to(memory_format=torch.channels_last))
            model = compiled
            print('torch.compile enabled', flush=True)
        except RuntimeError as e:
            print(f'torch.compile unavailable: {e}', flush=True)

    # Initialize
    if args.init != 'kaiming':
        dct_init_2d(model, fixup_scale=0.01, basis=args.init)
    else:
        torch.manual_seed(args.seed)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    init_md5 = model_md5(model)
    print(f'Init MD5: {init_md5}', flush=True)

    print(f'Optimizer: SGD lr={args.lr} momentum=0.9 wd=5e-4', flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0
    best_auc = 0
    best_per_class_auc = [0.0] * K
    best_ep = -1
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        if args.batch_order == 'golden':
            sampler.set_epoch(ep)
            # Seed augmentation RNG per epoch for bit-identical golden runs
            torch.manual_seed(ep * 1000)
        else:
            # Seeded runs: augmentation varies with seed (expected)
            torch.manual_seed(args.seed * 10000 + ep * 1000)

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, memory_format=torch.channels_last)
            if multi_label:
                Y_batch = Y_batch.float().to(DEVICE)
            else:
                # MedMNIST labels are (N, 1) shaped — squeeze to (N,)
                Y_batch = Y_batch.squeeze().long().to(DEVICE)
            logits, h = model(X_batch)
            if multi_label:
                loss = F.binary_cross_entropy_with_logits(logits, Y_batch)
            else:
                loss = F.cross_entropy(logits, Y_batch)
            # L2 regularize buffer dims beyond K (bottleneck only)
            if bn_dim is not None and h.shape[1] > K:
                loss += BN_REG_WEIGHT * h[:, K:].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            correct, total = 0, 0
            all_probs, all_labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(DEVICE, memory_format=torch.channels_last)
                    if multi_label:
                        y = y.float().to(DEVICE)
                    else:
                        y = y.squeeze().long().to(DEVICE)
                    logits = model(x)[0]
                    if multi_label:
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        correct += (preds == y).all(dim=1).sum().item()
                        total += y.size(0)
                        all_probs.append(torch.sigmoid(logits).cpu().numpy())
                        all_labels.append(y.cpu().numpy())
                    else:
                        correct += (logits.argmax(1) == y).sum().item()
                        total += y.size(0)
                        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                        all_labels.append(y.cpu().numpy())
            acc = correct / total
            y_true = np.concatenate(all_labels)
            probs = np.concatenate(all_probs)
            try:
                if multi_label:
                    auc = roc_auc_score(y_true, probs, average='macro')
                    per_class_auc = roc_auc_score(y_true, probs, average=None)
                elif K == 2:
                    auc = roc_auc_score(y_true, probs[:, 1])
                    per_class_auc = np.array([auc, auc])
                else:
                    auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
                    per_class_auc = roc_auc_score(y_true, probs, multi_class='ovr', average=None)
            except ValueError:
                auc = 0.0
                per_class_auc = np.zeros(K)
            if acc > best_acc:
                best_acc = acc
                best_auc = auc
                best_per_class_auc = [round(float(a), 4) for a in per_class_auc]
                best_ep = ep
            print(f'  Ep {ep}: acc={acc:.4f} auc={auc:.4f} (best={best_acc:.4f}@{best_ep}) '
                  f'({time.time()-t0:.0f}s)', flush=True)

    final_md5 = model_md5(model)

    result = {
        'dataset': args.dataset,
        'arch': arch_name,
        'init': args.init,
        'seed': args.seed,
        'bn_dim': bn_dim,
        'num_classes': K,
        'best_acc': round(best_acc, 4),
        'best_auc': round(best_auc, 4),
        'per_class_auc': best_per_class_auc,
        'batch_order': args.batch_order,
        'best_ep': best_ep,
        'params': params,
        'init_md5': init_md5,
        'final_md5': final_md5,
        'time': round(time.time() - t0, 1),
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    order_tag = 'golden' if args.batch_order == 'golden' else f's{args.seed}'
    out_file = os.path.join(
        results_dir, f'{args.dataset}_{arch_name}_{args.init}_{order_tag}_results.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_file}', flush=True)


if __name__ == '__main__':
    main()
