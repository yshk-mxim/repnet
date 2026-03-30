# Copy from ../repnet/ecg/run_spatial_stability.py and clean up
#
# Spatial attribution stability analysis: compares GradCAM maps across seeds.
# DCT init (deterministic) should produce identical attributions;
# Kaiming init (random) produces different explanations per seed.
#
# Usage:
#   python experiments/run_spatial_stability.py
