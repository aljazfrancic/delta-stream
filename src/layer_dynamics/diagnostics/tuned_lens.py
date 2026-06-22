"""Tuned-Lens-implied update — the Step-5 gate baseline (Phase 1 Step 3, part B).

The Tuned Lens (Belrose et al. 2023) learns a per-layer affine translator that
aligns an intermediate residual toward the final residual:
``r̂_ℓ = T_ℓ v_ℓ + c_ℓ`` (in residual space, before the unembed). We score its
*implied update* ``Δ̂^TL_ℓ = r̂_{ℓ+1} − r̂_ℓ`` as a predictor of the true
``Δ_ℓ``. Per the plan's last-layer convention, ``r̂_L ≡ v_L`` (T_L = I, c_L = 0),
so the comparison runs over all L pairs.

The public lens's translator is residual-form
(``transform_hidden(h, ℓ) = h + T̃_ℓ h + c_ℓ``, verified), and its translator ℓ
aligns hidden state ℓ — which is exactly our ``v_ℓ`` (``v_ℓ`` = HF
``hidden_states[ℓ]``). So ``transform_hidden`` applies directly to collected
``v_ℓ`` with no convention juggling.
"""

from __future__ import annotations

import numpy as np


def load_lens(resource_id: str):
    """Load the public Tuned Lens for ``resource_id`` (an HF model id).

    Returns the ``TunedLens`` whose ``transform_hidden(h, ℓ)`` yields r̂_ℓ.
    """
    from transformers import AutoModelForCausalLM
    from tuned_lens import TunedLens

    hf = AutoModelForCausalLM.from_pretrained(resource_id)
    return TunedLens.from_model_and_pretrained(hf, lens_resource_id=resource_id)


def tl_implied_deltas(lens, rs: np.ndarray) -> np.ndarray:
    """Δ̂^TL_ℓ = r̂_{ℓ+1} − r̂_ℓ for ℓ in 0..L-1, from residual states ``rs``.

    ``rs`` is ``[N, L+1, D]`` (v_0..v_L). Output ``[N, L, D]``.
    """
    import torch

    n, n_states, d = rs.shape
    n_layers = n_states - 1
    v = torch.from_numpy(rs).float()

    rhat = torch.empty_like(v)
    with torch.no_grad():
        for ell in range(n_layers):
            rhat[:, ell] = lens.transform_hidden(v[:, ell], ell)
        rhat[:, n_layers] = v[:, n_layers]  # r̂_L ≡ v_L (T_L = I, c_L = 0)

    return (rhat[:, 1:] - rhat[:, :-1]).numpy()
