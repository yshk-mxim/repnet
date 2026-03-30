# Copy from ../repnet/ecg/run_cifar_dct.py and clean up
#
# CIFAR-100 experiment: demonstrates DCT initialization is domain-agnostic.
# Compares DCT init vs Kaiming init on standard ResNet-18 for 32x32 images.
#
# Usage:
#   python experiments/run_cifar100.py --init dct --seed 42
#   python experiments/run_cifar100.py --init kaiming --seed 42
