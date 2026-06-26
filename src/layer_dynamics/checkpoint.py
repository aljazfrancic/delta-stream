"""Persist / reload a trained Branch B :class:`~layer_dynamics.ssm.LowRankSSM`.

Phase 3 fit the decomposition but never saved it (Gate-3 caveat 7: "fits ~16 min
each to regenerate"). Phase 4 needs *one canonical* trained SSM that the baseline
comparison, the stability metrics, and the causal existence proof all read — refitting
per script would be both slow and non-reproducible (the seed-stability degeneracy means
two fits at the same seed are bitwise-identical but two *scripts* re-deriving "the" fit
is asking for drift). So the fit is done once, here, and reloaded everywhere else.

The checkpoint is a plain ``torch.save`` dict: the model ``state_dict`` plus the
:class:`~layer_dynamics.ssm.SSMDims` and the two construction flags (``learn_z0``,
``use_state``) needed to rebuild the module before loading weights. Kept deliberately
small — the inferred inputs ``u`` are saved separately as ``.npy`` by the caller (they
are large and not part of the model).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from layer_dynamics.ssm import LowRankSSM, SSMDims


def save_ssm(model: LowRankSSM, path: str | Path, *, meta: dict | None = None) -> None:
    """Write ``model`` (weights + shape/flags) to ``path`` (a ``.pt`` file)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dims": asdict(model.dims),
            "learn_z0": model.z0 is not None,
            "use_state": model.use_state,
            "meta": meta or {},
        },
        path,
    )


def load_ssm(path: str | Path, *, device: str | torch.device = "cpu") -> LowRankSSM:
    """Rebuild a :class:`LowRankSSM` from a checkpoint written by :func:`save_ssm`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LowRankSSM(
        SSMDims(**ckpt["dims"]),
        learn_z0=ckpt["learn_z0"],
        use_state=ckpt["use_state"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
