"""Tests for the run-logging backbone (Phase 1 Step 0)."""

from __future__ import annotations

import json

from layer_dynamics.runlog import (
    git_info,
    init_run,
    load_config,
    make_out_dir,
    package_versions,
)


def test_make_out_dir_shape():
    out = make_out_dir("phase1 diagnostic", runs_root="runs", date="2026-06-20")
    assert out.as_posix() == "runs/2026-06-20-phase1-diagnostic"


def test_load_config_roundtrip(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  name: pythia-410m\nseed_note: x\n")
    loaded = load_config(cfg)
    assert loaded["model"]["name"] == "pythia-410m"
    assert load_config(None) == {}


def test_git_info_keys():
    info = git_info()
    assert set(info) == {"commit", "branch", "dirty"}


def test_package_versions_unknown_is_none():
    versions = package_versions(("definitely-not-a-real-package-xyz",))
    assert versions["definitely-not-a-real-package-xyz"] is None


def test_init_run_writes_provenance(tmp_path):
    out = tmp_path / "run"
    run = init_run(out, config={"model": {"name": "pythia-410m"}}, seed=42)

    assert run.seed == 42
    assert (out / "config.yaml").exists()
    assert (out / "meta.json").exists()
    assert (out / "run.log").exists()

    meta = json.loads((out / "meta.json").read_text())
    assert meta["seed"] == 42
    assert set(meta["git"]) == {"commit", "branch", "dirty"}
    assert "packages" in meta

    # Standard subdirs are created on demand.
    run.save_json("tables/r.json", {"a": 1})
    assert (out / "tables" / "r.json").exists()
    assert run.figures.is_dir()


def test_init_run_rejects_double_config(tmp_path):
    try:
        init_run(tmp_path / "r", config={}, config_path="x.yaml")
    except ValueError:
        return
    raise AssertionError("expected ValueError when both config and config_path given")
