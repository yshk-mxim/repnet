"""PTB-XL data loading and download."""
import os
import ast
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch

from repnet.models import K, RHYTHM_CLASSES


def download_ptbxl(data_dir):
    """Download PTB-XL from PhysioNet if not already present.

    Uses urllib (stdlib) instead of wget to avoid system dependency.
    """
    marker = os.path.join(data_dir, 'ptbxl_database.csv')
    if os.path.exists(marker):
        print(f'PTB-XL already downloaded at {data_dir}', flush=True)
        return

    os.makedirs(data_dir, exist_ok=True)
    url = 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip'
    zip_path = os.path.join(data_dir, 'ptb-xl-1.0.3.zip')
    print(f'Downloading PTB-XL to {data_dir} ...', flush=True)
    urllib.request.urlretrieve(url, zip_path)
    print('Extracting...', flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Extract contents, stripping the top-level directory
        top_dir = None
        for member in zf.namelist():
            parts = member.split('/', 1)
            if top_dir is None and len(parts) > 1:
                top_dir = parts[0]
            if len(parts) > 1 and parts[1]:
                target = os.path.join(data_dir, parts[1])
                if member.endswith('/'):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
    os.remove(zip_path)
    print('Download complete.', flush=True)


def load_ptbxl(data_dir, sr=100, return_folds=False):
    """Load PTB-XL. Returns X_tr, Y_tr, X_va, Y_va, X_te, Y_te as tensors.

    Uses the standard 8/1/1 fold split from Wagner et al. (2020).
    Z-normalization fit on train only (no test leakage).
    If return_folds=True, also returns fold assignments as a tensor.
    """
    import wfdb

    Y_df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    Y_df.scp_codes = Y_df.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg = pd.read_csv(os.path.join(data_dir, 'scp_statements.csv'), index_col=0)
    rhythm_set = set(agg[agg.rhythm == 1.0].index.tolist())

    def get_labels(scp):
        labels = np.zeros(K, dtype=np.float32)
        for code in scp:
            if code in rhythm_set:
                idx = RHYTHM_CLASSES.index(code) if code in RHYTHM_CLASSES else -1
                if idx >= 0:
                    labels[idx] = 1.0
        return labels

    Y_df['rv'] = Y_df.scp_codes.apply(get_labels)

    print('Loading signals...', flush=True)
    sigs = []
    fns = Y_df.filename_lr.values if sr == 100 else Y_df.filename_hr.values
    for i, fn in enumerate(fns):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, fn))
        sigs.append(sig)
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{len(fns)}', flush=True)

    X = np.array(sigs, dtype=np.float32).transpose(0, 2, 1)  # (N, 12, 1000)
    Y = np.stack(Y_df['rv'].values)

    folds = Y_df.strat_fold.values

    # Z-normalization fit on TRAIN only
    X_train = X[folds <= 8]
    mu, std = X_train.mean(), X_train.std() + 1e-8
    X = (X - mu) / std

    result = (torch.tensor(X[folds <= 8]), torch.tensor(Y[folds <= 8]),
              torch.tensor(X[folds == 9]), torch.tensor(Y[folds == 9]),
              torch.tensor(X[folds == 10]), torch.tensor(Y[folds == 10]))
    if return_folds:
        return result + (torch.tensor(folds),)
    return result
