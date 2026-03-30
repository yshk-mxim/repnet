"""Model loading and evaluation utilities.

Loads trained checkpoints produced by experiments/train_ecg.py.
"""
import hashlib
import os
import torch

from repnet.models import MODEL_CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(ckpt_path, model_name=None, device=None):
    """Load a trained model from a checkpoint file.

    Supports wrapped checkpoints (dict with 'model_state' key, saved by
    train_ecg.py) and raw state dicts.

    Args:
        ckpt_path: Path to .pt checkpoint file.
        model_name: Model class name ('v9m', 'conformer', 'baseline').
            If None, inferred from checkpoint metadata or filename.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        (model, model_name, ckpt_dict) tuple.
    """
    if device is None:
        device = DEVICE

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    # Determine model class
    if model_name is None:
        model_name = ckpt.get('model_class') if isinstance(ckpt, dict) else None
    if model_name is None:
        base = os.path.basename(ckpt_path).lower()
        if 'conformer' in base:
            model_name = 'conformer'
        elif 'v9m' in base:
            model_name = 'v9m'
        elif 'baseline' in base:
            model_name = 'baseline'
        else:
            raise ValueError(
                f'Cannot infer model class from {ckpt_path}. '
                f'Use model_name= with one of: {list(MODEL_CLASSES.keys())}'
            )

    if model_name not in MODEL_CLASSES:
        raise ValueError(f'Unknown model: {model_name}. '
                         f'Available: {list(MODEL_CLASSES.keys())}')

    # Get constructor args from checkpoint or use defaults
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model_cls = MODEL_CLASSES[model_name]

    if model_name == 'baseline':
        model = model_cls(
            ch=ckpt_args.get('base_ch', 64),
            ks=ckpt_args.get('ks', 5),
        )
    elif model_name == 'conformer':
        model = model_cls(
            base_ch=ckpt_args.get('base_ch', 40),
            etf_init=False,  # will load trained weights
        )
    else:
        model = model_cls(etf_init=False)

    # Load state dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    else:
        state = ckpt
    model.load_state_dict(state)

    return model.to(device).eval(), model_name, ckpt


def model_md5(model):
    """Compute MD5 hash of all model parameters for determinism verification."""
    h = hashlib.md5(usedforsecurity=False)
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()
