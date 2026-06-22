"""Run logging backbone.

Every experiment script in this project follows the same contract (Phase 1
Step 0): it is invoked with ``--config config.yaml --out runs/<date>-<desc>/``
and, before doing any work, records *exactly* what was run — the resolved
config, the git commit, whether the tree was dirty, the random seed, and the
versions of the libraries whose numerics the result depends on.

This module provides that contract as :func:`init_run`, returning a
:class:`RunContext` that owns the output directory and the standard
``figures/`` / ``tables/`` subdirs used by the gate deliverables.

Notebooks are never the source of truth: a result that cannot be regenerated
from a script + its logged ``meta.json`` does not count.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from layer_dynamics.seeding import set_seed

# Packages whose versions are recorded in every run's meta.json. Kept to the
# numerically load-bearing parts of the stack; missing ones are recorded as
# null rather than raising, so the backbone works in a partial environment.
_TRACKED_PACKAGES = (
    "torch",
    "transformer-lens",
    "datasets",
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "h5py",
    "pandas",
)


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Best-effort git provenance: commit, branch, and dirty flag.

    Returns a dict with ``commit``/``branch``/``dirty`` set to ``None`` when not
    in a git repo (e.g. an extracted release tarball) rather than raising — a
    run outside version control is still a valid run, it just has no commit.
    """
    cwd = str(repo) if repo is not None else None

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return out.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        # None => not a git repo; "" (no changes) => clean; otherwise dirty.
        "dirty": None if status is None else bool(status),
    }


def package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str | None]:
    """Resolve installed versions for ``names`` (``None`` if not installed)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass
class RunContext:
    """Owns a single run's output directory and standard subdirectories.

    Created by :func:`init_run`; do not construct directly. Provides the
    ``figures/`` and ``tables/`` layout every gate deliverable expects, an
    append-only ``run.log``, and helpers to persist intermediate artefacts so
    figures regenerate from saved numerics rather than from a live session.
    """

    dir: Path
    config: dict[str, Any]
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def figures(self) -> Path:
        return self._subdir("figures")

    @property
    def tables(self) -> Path:
        return self._subdir("tables")

    def _subdir(self, name: str) -> Path:
        path = self.dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, *parts: str) -> Path:
        """Path inside the run dir, creating parent directories as needed."""
        path = self.dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(self, message: str) -> None:
        """Append a timestamped line to ``run.log`` and echo to stderr."""
        stamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        line = f"{stamp}  {message}"
        with (self.dir / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, file=sys.stderr)

    def save_json(self, name: str, obj: Any) -> Path:
        path = self.path(name)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        return path

    def save_yaml(self, name: str, obj: Any) -> Path:
        path = self.path(name)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(obj, fh, sort_keys=False)
        return path


def load_config(path: Path | str | None) -> dict[str, Any]:
    """Load a YAML config into a dict (empty dict when ``path`` is None)."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def make_out_dir(desc: str, *, runs_root: Path | str = "runs", date: str | None = None) -> Path:
    """Build the conventional ``runs/<date>-<desc>/`` path (not yet created)."""
    day = date or _dt.date.today().isoformat()
    safe = "-".join(desc.strip().split())
    return Path(runs_root) / f"{day}-{safe}"


def init_run(
    out: Path | str,
    *,
    config: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
    seed: int = 0,
    deterministic: bool = True,
    repo: Path | None = None,
) -> RunContext:
    """Initialise a run directory and record full provenance before work starts.

    Resolves the config (either passed in or loaded from ``config_path``), seeds
    the RNGs, creates ``out/``, and writes:

    - ``config.yaml`` — the exact resolved config the run used.
    - ``meta.json`` — timestamp, seed, git commit/branch/dirty, command line,
      Python version, platform, and tracked package versions.

    Returns the :class:`RunContext` for the rest of the script to write into.
    """
    if config is not None and config_path is not None:
        raise ValueError("pass either config or config_path, not both")
    resolved = dict(config) if config is not None else load_config(config_path)

    set_seed(seed, deterministic=deterministic)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "deterministic": deterministic,
        "git": git_info(repo),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if config_path is not None:
        meta["config_path"] = str(config_path)

    ctx = RunContext(dir=out_dir, config=resolved, seed=seed, meta=meta)
    ctx.save_yaml("config.yaml", resolved)
    ctx.save_json("meta.json", meta)

    git = meta["git"]
    commit = (git["commit"] or "no-git")[:12]
    dirty = " (dirty)" if git["dirty"] else ""
    ctx.log(f"run initialised  commit={commit}{dirty}  seed={seed}  out={out_dir}")
    return ctx
