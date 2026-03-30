# Bit-Identical Medical Deep Learning via Structured Orthogonal Initialization

Reproducible code for *"Bit-Identical Medical Deep Learning via Structured Orthogonal Initialization"*.

**Paper:** [arXiv:2603.28040](https://arxiv.org/abs/2603.28040)

## Quick Start

```bash
pip install -e .                 # install repnet package
pip install -r requirements.txt  # install pinned dependencies

bash reproduce.sh --quick        # sanity check (~30 min on GPU)
bash reproduce.sh                # core experiments (~12h on 1x GPU)
```

## Structure

```
repnet/                        # Library
  init.py                      # DCT/Hadamard/Hartley/Sinusoidal/ETF initialization
  models.py                    # ECG models: Conformer, BaselineCNN
  models_cifar.py              # CIFAR/MedMNIST models: ResNet-18
  conformer_block.py           # Conformer attention block
  data.py                      # PTB-XL download and loading
  batch.py                     # Deterministic batch ordering (golden ratio + seeded)
  metrics.py                   # Evaluation metrics (macro AUC, ECE, per-class)
  eval.py                      # Model loading + MD5 hash verification

experiments/                   # Standalone experiment scripts
  train_ecg.py                 # ECG rhythm classification (Tables 2-6)
  train_medmnist.py            # MedMNIST medical image benchmarks (Table 7)
  train_cifar100.py            # CIFAR-100 DCT vs Kaiming (Table A1)
  cross_validate.py            # 3-fold cross-validation (Table 9)
  verify_determinism.py        # Bit-identical training verification

evidence/                      # Experimental results backing all paper tables
  compute_paper_numbers.py     # Reproduces all paper numbers from JSON evidence
  eval_demographics.py         # Sex-stratified demographic analysis
  manifest.json                # Maps each paper table/figure to evidence files

tests/                         # Unit tests
  test_init_determinism.py     # Verifies zero-seed deterministic initialization

reproduce.sh                   # One-command full reproduction
results/                       # Output directory for new experiment runs
```

## Key Experiments

| Experiment | Script | Paper table | What it shows |
|---|---|---|---|
| ECG Conformer | `train_ecg.py --model conformer` | Tables 2, 3, 6 | 1.83M param conv-attention hybrid |
| ECG Baseline | `train_ecg.py --model baseline` | Table 3 | 1.65M param xresnet-style CNN |
| Basis comparison | `train_ecg.py --basis {dct,hadamard,hartley,sinusoidal}` | Table 5 | All bases equivalent (Friedman p=0.48) |
| MedMNIST | `train_medmnist.py --dataset pathmnist` | Table 7 | Cross-domain MD5-verified training |
| CIFAR-100 | `train_cifar100.py` | Table A1 | DCT matches Kaiming (TOST p<0.001) |
| Determinism | `verify_determinism.py` | — | MD5-verified bit-identical runs |
| Cross-validation | `cross_validate.py` | Table 9 | Variance decomposition |

## Verifying Paper Numbers

```bash
# After running experiments (or using pre-computed evidence/):
python evidence/compute_paper_numbers.py
```

This script recomputes all statistics (means, stds, p-values, Friedman tests, Cohen's d, TOST equivalence) from the JSON evidence files and compares against paper claims.

## Options

```bash
# ECG training
python experiments/train_ecg.py \
  --model conformer \     # conformer | baseline
  --batch-order seeded \  # seeded | golden (seed-free deterministic)
  --seed 42 \             # batch ordering seed (ignored if golden)
  --basis dct \           # dct | hadamard | hartley | sinusoidal
  --mixed-bases \         # DCT/Hadamard/Hartley per stage
  --class-weight sqrt \   # sqrt | none
  --epochs 85 \
  --data-dir data/ptb-xl

# MedMNIST (7 datasets)
python experiments/train_medmnist.py \
  --dataset pathmnist \   # pathmnist | dermamnist | bloodmnist | ...
  --init dct \            # dct | kaiming
  --seed 42

# CIFAR-100
python experiments/train_cifar100.py \
  --init dct              # dct | kaiming
```

## Data

All datasets are automatically downloaded on first run:

- **PTB-XL** (~1.5 GB): ECG recordings from PhysioNet
- **MedMNIST** (~100 MB per dataset): Medical image benchmarks
- **CIFAR-100** (~170 MB): Standard image classification benchmark
- **MIT-BIH AFDB/NSTDB**: ECG databases from PhysioNet (for cross-dataset and noise evaluation)

## Requirements

- Python 3.10+
- PyTorch 2.12+ with CUDA support
- See `requirements.txt` for exact pinned versions
- GPU with 8+ GB VRAM (tested on RTX 3080, RTX 5090, A100, RTX Pro 6000)

## Tests

```bash
python tests/test_init_determinism.py
```

Verifies that structured initialization produces identical weights without any random seed — the core claim of the paper.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{shkolnikov2026bitidentical,
  title={Bit-Identical Medical Deep Learning via Structured Orthogonal Initialization},
  author={Shkolnikov, Yakov P.},
  journal={arXiv preprint arXiv:2603.28040},
  year={2026},
  url={https://arxiv.org/abs/2603.28040}
}
```
