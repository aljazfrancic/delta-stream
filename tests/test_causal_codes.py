"""Tests for the depth-causal code inference (v1.1 check A; synthetic, CPU)."""

from __future__ import annotations

import pytest
import torch

from layer_dynamics.decompose import infer_inputs
from layer_dynamics.diagnostics.causal_codes import (
    PrefixSSM,
    feedthrough_image,
    infer_prefix_inputs,
    prefix_state,
)
from layer_dynamics.ssm import LowRankSSM, SSMDims

L, D, M, DZ, R, N = 5, 12, 20, 4, 3, 9


def _model(use_state: bool = True, seed: int = 0) -> LowRankSSM:
    # init_scale large enough that the state path is not numerically negligible
    return LowRankSSM(SSMDims(L, D, M, DZ, R), seed=seed, use_state=use_state,
                      init_scale=0.4).double()


def _data(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(N, L, M, generator=g, dtype=torch.float64)
    delta = torch.randn(N, L, D, generator=g, dtype=torch.float64)
    return u, delta


@pytest.mark.parametrize("use_state", [True, False])
def test_prefix_forward_at_full_length_is_the_model(use_state):
    model, (u, _) = _model(use_state), _data()
    with torch.no_grad():
        assert torch.equal(PrefixSSM(model, L)(u), model(u))


@pytest.mark.parametrize("use_state", [True, False])
@pytest.mark.parametrize("p", [1, 2, 4])
def test_prefix_forward_equals_the_first_p_outputs(use_state, p):
    """Δ̂_ℓ depends on u_0..u_ℓ only, so truncating the unroll is exact."""
    model, (u, _) = _model(use_state), _data()
    with torch.no_grad():
        got = PrefixSSM(model, p)(u[:, :p])
        want = model(u)[:, :p]
    assert torch.allclose(got, want, atol=1e-12)


def test_prefix_forward_rejects_bad_lengths():
    model, (u, _) = _model(), _data()
    with pytest.raises(ValueError):
        PrefixSSM(model, 0)
    with pytest.raises(ValueError):
        PrefixSSM(model, L + 1)
    with pytest.raises(ValueError):
        PrefixSSM(model, 2)(u)          # wrong number of input layers


def test_full_length_inference_is_the_recorded_encoder():
    """At p = L the truncated encoder is decompose.infer_inputs, batch for batch."""
    model, (u0, delta) = _model(), _data()
    kw = {"lambda_l1": 0.05, "n_steps": 15, "lipschitz": 1.0, "batch_size": 4}
    want = infer_inputs(model, u0, delta, **kw)
    got = infer_prefix_inputs(model, u0, delta, L, **kw)
    assert torch.equal(got, want)
    assert all(p.requires_grad for p in model.parameters())  # flags restored


def test_prefix_inference_cannot_see_later_layers():
    model, (u0, delta) = _model(), _data()
    p = 3
    kw = {"lambda_l1": 0.05, "n_steps": 15, "lipschitz": 1.0, "batch_size": 4}
    base = infer_prefix_inputs(model, u0, delta, p, **kw)
    u_alt, d_alt = u0.clone(), delta.clone()
    u_alt[:, p:] += 7.0
    d_alt[:, p:] -= 11.0
    assert torch.equal(infer_prefix_inputs(model, u_alt, d_alt, p, **kw), base)
    assert base.shape == (N, p, M)
    # ... and it does see the layers it is given
    d_alt2 = delta.clone()
    d_alt2[:, p - 1] += 5.0
    assert not torch.equal(infer_prefix_inputs(model, u0, d_alt2, p, **kw), base)


def test_prefix_inference_differs_from_truncating_the_full_solution():
    """Codes fit to the whole trajectory are not the codes fit to its first p layers:
    the later updates pull on the earlier codes through the state path."""
    model, (u0, delta) = _model(), _data()
    kw = {"lambda_l1": 0.05, "n_steps": 40, "lipschitz": 1.0, "batch_size": None}
    full = infer_prefix_inputs(model, u0, delta, L, **kw)
    pre = infer_prefix_inputs(model, u0, delta, 3, **kw)
    assert not torch.allclose(full[:, :3], pre, atol=1e-6)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_prefix_state_matches_the_model_trajectory(p):
    model, (u, _) = _model(), _data()
    assert torch.allclose(prefix_state(model, u[:, :p]), model.states(u)[:, p], atol=1e-12)


def test_prefix_state_bounds_and_baseline():
    model, (u, _) = _model(), _data()
    with pytest.raises(ValueError):
        prefix_state(model, u)                   # p = L: no transition out of the last layer
    with pytest.raises(ValueError):
        prefix_state(_model(use_state=False), u[:, :2])


def test_feedthrough_image_is_the_bottleneck_of_the_per_layer_write():
    model, (u, _) = _model(), _data()
    ell = 2
    img = feedthrough_image(model, u[:, ell], ell)
    assert img.shape == (N, R)
    with torch.no_grad():
        write = (u[:, ell] @ model.D2[ell]) @ model.D1[ell].T
        assert torch.allclose(img @ model.D1[ell].T, write, atol=1e-12)
