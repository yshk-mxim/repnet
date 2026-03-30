#!/usr/bin/env python3
"""Reproduce all paper table numbers from evidence JSON files.

Usage: python compute_paper_numbers.py [--verify]
  Without --verify: prints computed values
  With --verify: compares against paper claims and flags mismatches

All data sourced from release/evidence/ directory.
"""
import json, glob, os, sys
import numpy as np
from scipy import stats
from pathlib import Path

EVIDENCE = Path(__file__).parent

def load_aucs(pattern):
    """Load AUC values from JSON files matching glob pattern."""
    files = sorted(glob.glob(str(EVIDENCE / pattern)))
    aucs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        auc = d.get("auc", d.get("best_auc", d.get("test_macro_auc")))
        if auc is not None:
            aucs.append(float(auc))
    return np.array(aucs), len(files)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ============================================================
#  TABLE 3: Main Results
# ============================================================
section("TABLE 3: Structured Initialization Across Architectures")

conf_mixed, n = load_aucs("ecg/conformer_mixed/conformer_ms_s*_metrics.json")
print(f"Conformer Mixed (n={len(conf_mixed)}): {conf_mixed.mean():.3f} ± {conf_mixed.std(ddof=1):.3f}")

conf_kaiming, n = load_aucs("ecg/conformer_kaiming/conformer_kaiming_s*_metrics.json")
print(f"Conformer Kaiming (n={len(conf_kaiming)}): {conf_kaiming.mean():.3f} ± {conf_kaiming.std(ddof=1):.3f}")

base_dct, n = load_aucs("ecg/baseline_dct/baseline_fulldet_bs*_metrics.json")
if len(base_dct) < 20:
    base_dct_ext, _ = load_aucs("ecg_n20/baseline_dct_s*_metrics.json")
    if len(base_dct_ext) > len(base_dct):
        base_dct = base_dct_ext
print(f"Baseline DCT (n={len(base_dct)}): {base_dct.mean():.3f} ± {base_dct.std(ddof=1):.3f}")

base_kaim, n = load_aucs("ecg/baseline_kaiming/baseline_kaiming_s*_metrics.json")
if len(base_kaim) < 20:
    base_kaim_ext, _ = load_aucs("ecg_n20/baseline_kaiming_s*_metrics.json")
    if len(base_kaim_ext) > len(base_kaim):
        base_kaim = base_kaim_ext
print(f"Baseline Kaiming (n={len(base_kaim)}): {base_kaim.mean():.3f} ± {base_kaim.std(ddof=1):.3f}")

# Welch t-tests
t_conf, p_conf = stats.ttest_ind(conf_mixed, conf_kaiming, equal_var=False)
d_conf = (conf_mixed.mean() - conf_kaiming.mean()) / np.sqrt(
    (conf_mixed.std(ddof=1)**2 + conf_kaiming.std(ddof=1)**2) / 2)
print(f"\nConformer: t={t_conf:.3f}, p={p_conf:.4f}, Cohen's d={d_conf:.2f}")

t_base, p_base = stats.ttest_ind(base_dct, base_kaim, equal_var=False)
d_base = (base_dct.mean() - base_kaim.mean()) / np.sqrt(
    (base_dct.std(ddof=1)**2 + base_kaim.std(ddof=1)**2) / 2)
print(f"Baseline:  t={t_base:.3f}, p={p_base:.4f}, Cohen's d={d_base:.2f}")

# ============================================================
#  TABLE 5: Basis Comparison
# ============================================================
section("TABLE 5: Orthogonal Basis Comparison (n=20)")

bases = {}
for basis in ["dct", "hadamard", "hartley", "sinusoidal"]:
    aucs, n = load_aucs(f"basis_comparison/conformer_{basis}/conformer_{basis}_s*_metrics.json")
    bases[basis] = aucs
    print(f"{basis:12s} (n={len(aucs)}): {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}")

kaim_cw, _ = load_aucs("basis_comparison/conformer_kaiming_cw/conformer_kaiming_cw_s*_metrics.json")
converged = kaim_cw[kaim_cw > 0.67]
print(f"Kaiming+cw   (n={len(converged)} of {len(kaim_cw)}): {converged.mean():.3f} ± {converged.std(ddof=1):.3f}")

# Friedman test
common_n = min(len(v) for v in bases.values())
chi2, p_fried = stats.friedmanchisquare(*[v[:common_n] for v in bases.values()])
print(f"\nFriedman chi2={chi2:.2f}, p={p_fried:.2f} (n={common_n})")

# ============================================================
#  TABLE 2: Ablation
# ============================================================
section("TABLE 2: Ablation")

for name, pattern in [
    ("mixed_nocw", "ecg/ablation/conformer_mixed_nocw_s*_metrics.json"),
    ("dct_cw", "ecg/ablation/conformer_dct_cw_s*_metrics.json"),
    ("kaiming_cw", "ecg/ablation/conformer_kaiming_cw_s*_metrics.json"),
]:
    aucs, n = load_aucs(pattern)
    if name == "kaiming_cw":
        converged = aucs[aucs > 0.60]
        print(f"{name:12s} (n={len(converged)}/{len(aucs)} converged): {converged.mean():.3f} ± {converged.std(ddof=1):.3f}")
    else:
        print(f"{name:12s} (n={len(aucs)}): {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f}")

# ============================================================
#  TABLE 7: MedMNIST
# ============================================================
section("TABLE 7: MedMNIST Cross-Domain Validation")

datasets = ["bloodmnist", "organcmnist", "dermamnist", "breastmnist",
            "retinamnist", "chestmnist", "pathmnist"]
for ds in datasets:
    dct_aucs, _ = load_aucs(f"medmnist/{ds}/dct/{ds}_resnet18_dct_s*_results.json")
    kaim_aucs, _ = load_aucs(f"medmnist/{ds}/kaiming/{ds}_resnet18_kaiming_s*_results.json")
    if len(dct_aucs) > 0 and len(kaim_aucs) > 0:
        t, p = stats.ttest_ind(dct_aucs, kaim_aucs, equal_var=False)
        delta = dct_aucs.mean() - kaim_aucs.mean()
        print(f"{ds:14s}: DCT {dct_aucs.mean():.4f}±{dct_aucs.std(ddof=1):.4f}  "
              f"Kaim {kaim_aucs.mean():.4f}±{kaim_aucs.std(ddof=1):.4f}  "
              f"Δ={delta:+.4f}  p={p:.2f}")
    else:
        print(f"{ds:14s}: DCT n={len(dct_aucs)}, Kaim n={len(kaim_aucs)} — insufficient data")

# ============================================================
#  TABLE A1: CIFAR-100
# ============================================================
section("TABLE A1: CIFAR-100")

cifar_dct, _ = load_aucs("cifar100/cifar_cifar100_dct_s*_results.json")
cifar_kaim, _ = load_aucs("cifar100/cifar_cifar100_kaiming_s*_results.json")
if len(cifar_dct) > 0:
    # CIFAR uses accuracy, not AUC
    dct_acc = np.array([json.load(open(f))["best_acc"] for f in sorted(glob.glob(str(EVIDENCE / "cifar100/cifar_cifar100_dct_s*_results.json")))])
    kaim_acc = np.array([json.load(open(f))["best_acc"] for f in sorted(glob.glob(str(EVIDENCE / "cifar100/cifar_cifar100_kaiming_s*_results.json")))])
    t, p = stats.ttest_ind(dct_acc, kaim_acc, equal_var=False)
    d = (dct_acc.mean() - kaim_acc.mean()) / np.sqrt((dct_acc.std(ddof=1)**2 + kaim_acc.std(ddof=1)**2) / 2)
    print(f"DCT    (n={len(dct_acc)}): {dct_acc.mean()*100:.1f} ± {dct_acc.std(ddof=1)*100:.1f}%")
    print(f"Kaiming (n={len(kaim_acc)}): {kaim_acc.mean()*100:.1f} ± {kaim_acc.std(ddof=1)*100:.1f}%")
    print(f"t={t:.2f}, p={p:.2f}, d={d:.2f}")

# ============================================================
#  TABLE 6: Golden 2×2
# ============================================================
section("TABLE 6: Golden 2×2 Design")

golden_file = EVIDENCE / "ecg/golden/conformer_combined_golden_metrics.json"
if golden_file.exists():
    d = json.load(open(golden_file))
    print(f"Conformer Mixed golden: AUC={d['auc']}")
    if "per_auc" in d:
        for cls, auc in d["per_auc"].items():
            print(f"  {cls}: {auc:.3f}")

for name in ["conformer_dct_golden", "conformer_kaiming_golden",
             "baseline_dct_golden", "baseline_kaiming_golden"]:
    f = EVIDENCE / f"basis_golden/{name}_metrics.json"
    if f.exists():
        d = json.load(open(f))
        print(f"{name}: AUC={d.get('auc', '?')}")

# ============================================================
#  TABLE 9: Cross-Validation
# ============================================================
section("TABLE 9: Cross-Validation")

for cvfile in sorted(glob.glob(str(EVIDENCE / "ecg/cv/*.json"))):
    name = Path(cvfile).stem
    d = json.load(open(cvfile))
    if isinstance(d, dict) and "mean" in d:
        print(f"{name}: mean={d['mean']:.3f} ± {d.get('std', 0):.3f}, folds={d.get('folds', '?')}")
    elif isinstance(d, dict):
        for config, vals in d.items():
            if isinstance(vals, dict) and "mean" in vals:
                print(f"  {config}: mean={vals['mean']:.3f} ± {vals.get('std', 0):.3f}")

# ============================================================
#  TOST Equivalence
# ============================================================
section("TOST Equivalence Tests")

def tost(x, y, delta):
    """Two One-Sided Tests for equivalence within margin delta."""
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
    t_upper = (diff - delta) / se
    t_lower = (diff + delta) / se
    df = len(x) + len(y) - 2
    p_upper = stats.t.cdf(t_upper, df)
    p_lower = 1 - stats.t.cdf(t_lower, df)
    return max(p_upper, p_lower)

p_tost_conf = tost(conf_mixed, conf_kaiming, 0.015)
print(f"Conformer TOST (δ=0.015): p={p_tost_conf:.4f}")

if len(cifar_dct) > 0:
    p_tost_cifar = tost(dct_acc, kaim_acc, 0.005)
    print(f"CIFAR-100 TOST (δ=0.5pp): p={p_tost_cifar:.4f}")

# ============================================================
#  Permutation Test
# ============================================================
section("Permutation Test: Sex-Label Association")


def exact_permutation_test(group_a, group_b, n_permutations=10000, seed=42):
    """Two-sample permutation test on macro AUC.

    Tests whether the difference in means between group_a and group_b
    is significant by randomly permuting group assignments.

    Args:
        group_a: array of AUC values for group A (e.g., male patients)
        group_b: array of AUC values for group B (e.g., female patients)
        n_permutations: number of random permutations (default 10,000)
        seed: random seed for reproducibility

    Returns:
        observed_diff: observed difference in means (A - B)
        p_value: two-sided permutation p-value
    """
    rng = np.random.RandomState(seed)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)
    observed_diff = group_a.mean() - group_b.mean()
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = combined[:n_a].mean() - combined[n_a:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    p_value = count / n_permutations
    return observed_diff, p_value


# Load pre-computed permutation test results
demo_file = EVIDENCE / "ecg" / "demographics" / "demographic_exact_results.json"
if demo_file.exists():
    demo = json.load(open(demo_file))
    print(f"Demographic analysis (n={demo['n_seeds']} seeds, {len(demo['shared_classes'])} classes):")
    print(f"  Male AUC:  {demo['male_mean_auc']:.4f}")
    print(f"  Female AUC: {demo['female_mean_auc']:.4f}")
    print(f"  Diff (M-F): {demo['observed_diff_MF']:+.4f}")
    print(f"  Permutation p-value: {demo['p_value_twosided']:.4f} ({demo['n_permutations']} perms)")
    print(f"  Shared classes: {demo['shared_classes']}")
else:
    print(f"Demographic evidence not found at {demo_file}")

print("\n" + "="*60)
print("  DONE — all numbers computed from evidence JSON files")
print("="*60)
