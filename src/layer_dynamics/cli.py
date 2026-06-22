"""Shared CLI scaffolding for run scripts.

Every script under ``scripts/`` takes the same three arguments — ``--config``,
``--out``, ``--seed`` — so :func:`add_common_args` defines them once. The
``layer-dynamics`` console entry point (:func:`main`) is a thin status command
used to confirm the package is importable and to print provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer_dynamics import __version__
from layer_dynamics.runlog import git_info, make_out_dir, package_versions


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--config / --out / --seed`` arguments shared by run scripts."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file for the run (optional).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run output directory. Defaults to runs/<date>-<desc>/ when --desc is given.",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default=None,
        help="Short slug used to build a default --out of runs/<date>-<desc>/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    return parser


def resolve_out(args: argparse.Namespace) -> Path:
    """Resolve the output directory from ``--out`` or ``--desc``."""
    if args.out is not None:
        return args.out
    if args.desc is not None:
        return make_out_dir(args.desc)
    raise SystemExit("one of --out or --desc is required")


def main(argv: list[str] | None = None) -> int:
    """``layer-dynamics`` console command: print version and provenance."""
    parser = argparse.ArgumentParser(prog="layer-dynamics", description=__doc__)
    parser.add_argument("--version", action="version", version=f"layer-dynamics {__version__}")
    parser.parse_args(argv)

    git = git_info()
    commit = (git["commit"] or "no-git")[:12]
    print(f"layer-dynamics {__version__}  (commit {commit}, dirty={git['dirty']})")
    print("tracked packages:")
    for name, version in package_versions().items():
        print(f"  {name:16s} {version or '— not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
