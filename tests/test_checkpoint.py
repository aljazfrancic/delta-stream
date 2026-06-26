"""Round-trip the Branch B checkpoint save/load (Phase 4 foundation)."""

from __future__ import annotations

import torch

from layer_dynamics.checkpoint import load_ssm, save_ssm
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _toy(seed: int = 0, **kw) -> LowRankSSM:
    dims = SSMDims(n_layers=4, d_model=8, d_input=6, d_state=3, rank=2)
    return LowRankSSM(dims, seed=seed, **kw)


def test_save_load_reproduces_forward(tmp_path):
    model = _toy()
    u = torch.randn(5, 4, 6)
    expected = model(u)

    path = tmp_path / "ssm.pt"
    save_ssm(model, path, meta={"note": "toy"})
    reloaded = load_ssm(path)

    assert reloaded.dims == model.dims
    torch.testing.assert_close(reloaded(u), expected)


def test_roundtrip_preserves_flags(tmp_path):
    # learn_z0=False and use_state=False must survive the round-trip (they change forward).
    model = _toy(learn_z0=False, use_state=False)
    assert model.z0 is None
    path = tmp_path / "ssm.pt"
    save_ssm(model, path)
    reloaded = load_ssm(path)
    assert reloaded.z0 is None
    assert reloaded.use_state is False
    u = torch.randn(3, 4, 6)
    torch.testing.assert_close(reloaded(u), model(u))


def test_meta_persisted(tmp_path):
    path = tmp_path / "ssm.pt"
    save_ssm(_toy(), path, meta={"lambda_l1": 1.0, "seed": 0})
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["meta"]["lambda_l1"] == 1.0
