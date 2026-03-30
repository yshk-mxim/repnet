"""Demographics n=20: all release DCT Conformer checkpoints."""
import torch, numpy as np, json, glob, sys, os
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/repnet/release"))
from repnet.models import ECGRepNetConformer
from repnet.data import load_ptbxl
from sklearn.metrics import roc_auc_score
import pandas as pd

DATA_DIR = os.path.expanduser("~/repnet/release/data/ptb-xl")
CKPT_DIR = os.path.expanduser("~/repnet/release")
_, _, _, _, X_te, Y_te = load_ptbxl(DATA_DIR)
meta = pd.read_csv(os.path.join(DATA_DIR, "ptbxl_database.csv"), index_col="ecg_id")
sex_te = meta.loc[meta["strat_fold"]==10, "sex"].values
male_mask = sex_te == 1
female_mask = sex_te == 0
Y_np = Y_te.numpy()
shared = [0,1,2,3,4,5,6,7,8]

ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "best_conformer_dct_s*.pt")))
print("Found %d checkpoints" % len(ckpts))
assert len(ckpts) == 20, "Expected 20 checkpoints, got %d" % len(ckpts)

results = []
all_probs = []
for ckpt in ckpts:
    model = ECGRepNetConformer(base_ch=40)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        out = model(X_te)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits).numpy()
    all_probs.append(probs)
    r = {}
    for label, mask in [("male", male_mask), ("female", female_mask)]:
        aucs = [roc_auc_score(Y_np[mask, c], probs[mask, c]) for c in shared]
        r[label] = round(np.mean(aucs), 4)
    r["diff"] = round(r["male"] - r["female"], 4)
    name = Path(ckpt).stem.replace("best_", "")
    print("  %s: M=%.4f F=%.4f d=%.4f" % (name, r["male"], r["female"], r["diff"]))
    results.append(r)

m = [r["male"] for r in results]
f = [r["female"] for r in results]
d = [r["diff"] for r in results]
print("\nMatched 9-class (n=%d)" % len(results))
print("Male:   %.4f +/- %.4f" % (np.mean(m), np.std(m)))
print("Female: %.4f +/- %.4f" % (np.mean(f), np.std(f)))
print("Diff:   %.4f +/- %.4f" % (np.mean(d), np.std(d)))

# Exact permutation test
obs = np.mean(d)
rng = np.random.default_rng(42)
n_test = len(sex_te)
n_male = male_mask.sum()
count = 0
for p in range(10000):
    pi = rng.permutation(n_test)
    pm = np.zeros(n_test, dtype=bool)
    pm[pi[:n_male]] = True
    pf = ~pm
    sd = []
    for probs in all_probs:
        ma = np.mean([roc_auc_score(Y_np[pm, c], probs[pm, c]) for c in shared])
        fa = np.mean([roc_auc_score(Y_np[pf, c], probs[pf, c]) for c in shared])
        sd.append(ma - fa)
    if abs(np.mean(sd)) >= abs(obs):
        count += 1
    if (p+1) % 2000 == 0:
        print("  %d/10000 permutations" % (p+1))
p_val = count / 10000

print("\n=== FINAL DEMOGRAPHICS n=20 ===")
print("Male:   %.4f +/- %.4f" % (np.mean(m), np.std(m)))
print("Female: %.4f +/- %.4f" % (np.mean(f), np.std(f)))
print("Diff:   %.4f" % obs)
print("Permutation p-value: %.4f" % p_val)

json.dump({
    "test": "exact_permutation", "n_permutations": 10000,
    "n_seeds": len(results), "shared_classes": ["SR","AFIB","STACH","SARRH","SBRAD","PACE","SVARR","BIGU","AFLT"],
    "observed_diff_MF": round(obs, 4), "p_value_twosided": round(p_val, 4),
    "male_mean_auc": round(np.mean(m), 4), "female_mean_auc": round(np.mean(f), 4),
    "per_seed_diffs": [round(x, 4) for x in d]
}, open(os.path.expanduser("~/repnet/release/results/demographic_n20_final.json"), "w"), indent=2)
print("Saved.")
