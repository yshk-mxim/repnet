"""Verify structured initialization is deterministic and architecture-independent."""
import torch
from repnet.models import ECGRepNetConformer, BaselineCNN
from repnet.init import deterministic_init
from repnet.eval import model_md5


def test_init_determinism():
    """Same model + basis → same MD5 every time, no seed required.

    The full deterministic init pipeline:
    1. Construct model with etf_init=True (deterministic ETF heads)
    2. Call deterministic_init() (deterministic conv/linear weights)
    No random seed is needed — the initialization is fully deterministic.
    """
    for basis in ["dct", "hadamard", "hartley", "sinusoidal"]:
        m1 = ECGRepNetConformer(base_ch=40, etf_init=True)
        deterministic_init(m1, basis=basis)
        h1 = model_md5(m1)

        m2 = ECGRepNetConformer(base_ch=40, etf_init=True)
        deterministic_init(m2, basis=basis)
        h2 = model_md5(m2)

        assert h1 == h2, f"{basis}: init MD5 mismatch {h1} != {h2}"
        print(f"  {basis}: {h1[:16]}... OK")


def test_baseline_determinism():
    """Baseline CNN also deterministic, no seed required."""
    m1 = BaselineCNN()
    deterministic_init(m1, basis="dct")
    h1 = model_md5(m1)

    m2 = BaselineCNN()
    deterministic_init(m2, basis="dct")
    h2 = model_md5(m2)

    assert h1 == h2, f"Baseline: init MD5 mismatch {h1} != {h2}"
    print(f"  baseline: {h1[:16]}... OK")


def test_forward_pass():
    """Model produces valid output (no NaN, correct shape)."""
    m = ECGRepNetConformer(base_ch=40)
    deterministic_init(m, basis="dct")
    m.eval()
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        out = m(x)
        logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, 12), f"Wrong shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in output"
    print(f"  forward: shape={logits.shape}, range=[{logits.min():.2f}, {logits.max():.2f}] OK")


def test_different_bases_different_weights():
    """Different bases produce different initializations."""
    hashes = {}
    for basis in ["dct", "hadamard", "hartley", "sinusoidal"]:
        m = ECGRepNetConformer(base_ch=40)
        deterministic_init(m, basis=basis)
        hashes[basis] = model_md5(m)

    for b1 in hashes:
        for b2 in hashes:
            if b1 != b2:
                assert hashes[b1] != hashes[b2], f"{b1} == {b2}"
    print("  all 4 bases produce distinct initializations OK")


if __name__ == "__main__":
    print("test_init_determinism:")
    test_init_determinism()
    print("test_baseline_determinism:")
    test_baseline_determinism()
    print("test_forward_pass:")
    test_forward_pass()
    print("test_different_bases_different_weights:")
    test_different_bases_different_weights()
    print("\nAll tests passed.")
