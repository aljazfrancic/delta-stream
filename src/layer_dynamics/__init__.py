"""layer_dynamics — structured cross-layer propagation of features in transformers.

The reusable backbone for the four-phase experimental protocol described in
``docs/implementation-plan/``. Phase 1 Step 0 only requires the run-logging
utilities exported here; later phases add modules alongside them.
"""

from layer_dynamics.runlog import RunContext, init_run
from layer_dynamics.seeding import set_seed

__all__ = ["RunContext", "init_run", "set_seed", "__version__"]

__version__ = "0.1.0"
