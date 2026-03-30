# Copy from ../repnet/ecg/train_ecg.py and clean up
#
# Cleanup tasks:
# - Update imports to use repnet.models instead of models_ecg
# - Update imports to use repnet.data_loader, repnet.batch_builder, repnet.metrics
# - Remove sys.path.insert hack
# - Remove BUG FIX comments (keep the fixes, remove the commentary)
# - Remove Finding/BUG reference comments
# - Clean up semicolons in hot loops (opt.zero_grad(); loss.backward(); opt.step())
# - Run ruff --fix to remove unused imports (BN_DIM, N_LEADS, BaselineCNN)
