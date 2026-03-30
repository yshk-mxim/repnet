# Copy from ../repnet/ecg/run_cifar10_bottleneck.py and clean up
#
# CIFAR-10 with information bottleneck (K+2=12 dim), matching ECG RepNet design.
# Proves the bottleneck architecture is domain-agnostic.
#
# Usage:
#   python experiments/run_cifar10_bottleneck.py --init dct --seed 42
#   python experiments/run_cifar10_bottleneck.py --init kaiming --seed 42
