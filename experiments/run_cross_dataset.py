# Copy from ../repnet/ecg/run_crossdataset_cpsc_chapman.py and clean up
#
# Cross-dataset evaluation: loads pre-trained ECG models and evaluates
# AFIB detection on CPSC2018 and Chapman-Shaoxing datasets.
# Tests generalization beyond PTB-XL training distribution.
#
# Usage:
#   python experiments/run_cross_dataset.py --checkpoint best_conformer_dct.pt
