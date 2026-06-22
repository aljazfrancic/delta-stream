"""Deterministic seeding across the libraries the pipeline uses.

Kept dependency-light: numpy and torch are seeded only if importable, so the
run-logging backbone (which imports this) works in a bare environment before
the scientific stack is installed.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) from a single integer.

    Args:
        seed: the run seed; also written into each run's ``meta.json``.
        deterministic: if True, request deterministic cuDNN kernels. This can
            slow GPU work but keeps Phase 3 decompositions reproducible across
            re-runs of the same seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
