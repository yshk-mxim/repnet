# Copy from ../repnet/ecg/train_ecg.py and adapt as experiment runner
#
# This script runs all ECG experiment configurations from the paper:
# - V9H + DCT init (headline)
# - V9H + DCT + sign constraint
# - Conformer + DCT
# - V9M (intermediate bottlenecks)
# - Baseline CNN
# - Ablations: no morph, no ETF, Kaiming init
#
# Usage:
#   python experiments/run_ecg.py --config configs/conformer_dct.yaml
#   python experiments/run_ecg.py --all
