#!/usr/bin/env python3
"""Smoke test of the run-logging contract (Phase 1 Step 0).

Does no science: it exercises the ``--config / --out / --seed`` interface,
initialises a run directory, and writes a trivial result so the provenance
machinery (config.yaml, meta.json, run.log, tables/) can be verified end to
end without the heavy scientific stack installed.

    python scripts/smoke_run.py --desc smoke-test --config configs/pythia-70m.yaml
"""

from __future__ import annotations

import argparse

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    run.log("smoke run started")

    # Stand-in for a real result: a deterministic value derived from the seed.
    result = {"seed_echo": args.seed, "answer": (args.seed * 7 + 3) % 100}
    run.save_json("tables/result.json", result)

    run.log(f"wrote result {result}")
    run.log("smoke run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
