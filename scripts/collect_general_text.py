#!/usr/bin/env python3
"""Collect residual-stream trajectories for generic (non-factual) text (Phase 4 E4).

The E4 generality control: re-run the Phase-1 H1 test (AR-on-Δ + Granger +
scrambled null) on **generic text** instead of factual-recall prompts, to decide
whether the cross-layer Δ structure is specific to factual recall or general to
transformer depth. This script builds the matched residual store; the analysis is
then the *identical* ``scripts/step5_ar.py`` pointed at the resulting HDF5 with
``--h5``.

Generic windows are drawn from the Pile (``NeelNanda/pile-10k``; Pythia-410m-
deduped's training distribution) and **length-matched** to the factual set — same
N, same multiset of content-token lengths, one window per distinct document at a
random in-document offset — so the only thing that differs from Phase 1 is
factual-recall vs generic text, not context length. The residual is read at the
final content token (BOS prepended), exactly as the factual store.

    python scripts/collect_general_text.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-e4-collect-410md

Writes ``data/<model>-general/residual_stream.h5`` with the same schema as the
factual store (``relation_class``/``relation`` filled with "general") plus the
window texts/lengths for provenance.

Sanity check: for 5 random windows, ln_final + unembed of v_L must recover the
model's own argmax next-token at that position (the generic analog of the factual
top-1 check — validates the extraction without needing a gold label).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from layer_dynamics.activations import collect_trajectories_from_ids, recovers_top1
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.data.general_text import length_matched_windows
from layer_dynamics.device import describe
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run

PILE = "NeelNanda/pile-10k"


def _content_lengths(model, prompts: list[str]) -> list[int]:
    """Factual content-token lengths (no special tokens) — the multiset to match."""
    return [len(model.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]


def _pile_doc_ids(model, n_docs: int, char_cap: int, tok_cap: int) -> list[list[int]]:
    """Tokenize the first ``n_docs`` Pile docs to content-id lists (capped length)."""
    from datasets import load_dataset

    ds = load_dataset(PILE, split="train")
    docs: list[list[int]] = []
    for i in range(min(n_docs, len(ds))):
        text = ds[i]["text"][:char_cap]
        ids = model.tokenizer(text, add_special_tokens=False)["input_ids"][:tok_cap]
        if ids:
            docs.append(list(ids))
    return docs


def _argmax_next_token(model, ids: list[int]) -> int:
    """The model's actual argmax next-token id for a content-id window (BOS prepended)."""
    import torch

    bos = int(model.to_tokens("", prepend_bos=True)[0, 0])
    toks = torch.tensor([[bos, *ids]], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(toks, return_type="logits")
    return int(logits[0, -1].argmax())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-docs", type=int, default=4000,
                        help="Pile documents to tokenize as the window pool.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg, ccfg = cfg["model"], cfg.get("collection", {})

    parquet = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    run.log(f"loaded {len(df)} factual examples from {parquet} (length-match target)")

    model = load_model(mcfg["name"], device=mcfg.get("device", "auto"), dtype=mcfg["dtype"])
    run.log(f"model {mcfg['name']} on {describe(next(model.parameters()).device.type)}")
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model

    target_lengths = _content_lengths(model, df["prompt"].tolist())
    run.log(f"factual length multiset: min {min(target_lengths)} / median "
            f"{int(np.median(target_lengths))} / max {max(target_lengths)}")

    docs = _pile_doc_ids(model, args.n_docs, char_cap=2000, tok_cap=256)
    run.log(f"tokenized {len(docs)} Pile docs (>= {max(target_lengths)} tokens: "
            f"{sum(len(d) >= max(target_lengths) for d in docs)})")

    windows = length_matched_windows(docs, target_lengths, seed=args.seed)
    assert [len(w) for w in windows] == target_lengths
    run.log(f"built {len(windows)} length-matched generic windows")

    traj = collect_trajectories_from_ids(
        model, windows, batch_size=int(ccfg.get("batch_size", 8))
    )
    assert traj.shape == (len(windows), n_layers + 1, d_model)
    run.log(f"collected residual_stream {traj.shape} ({traj.nbytes / 2**20:.0f} MiB)")

    # Sanity: ln_final + unembed of v_L recovers the model's own argmax next-token.
    rng = np.random.default_rng(args.seed)
    checks = []
    for i in rng.choice(len(windows), size=min(5, len(windows)), replace=False):
        rec = recovers_top1(model, traj[i, -1])
        expected = _argmax_next_token(model, windows[i])
        checks.append((int(i), rec, expected, rec == expected))
    n_ok = sum(c[3] for c in checks)
    for i, rec, exp, ok in checks:
        run.log(f"  sanity ex {i}: ln_final+unembed(v_L)={rec} vs model argmax={exp} "
                f"{'OK' if ok else 'MISMATCH'}")
    if n_ok != len(checks):
        run.log("WARNING: sanity check failed — investigate before trusting activations")

    texts = [model.tokenizer.decode(w) for w in windows]
    out_h5 = Path("data") / f"{mcfg['name']}-general" / "residual_stream.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("residual_stream", data=traj, compression="gzip", compression_opts=4)
        f.create_dataset("example_id", data=np.arange(len(windows), dtype=np.int64))
        # Placeholders so the downstream loader's schema is satisfied; generic text
        # has no relation labels.
        f.create_dataset("relation_class",
                         data=np.array(["general"] * len(windows), dtype=object), dtype=str_dt)
        f.create_dataset("relation",
                         data=np.array(["general"] * len(windows), dtype=object), dtype=str_dt)
        f.create_dataset("window_text", data=np.array(texts, dtype=object), dtype=str_dt)
        f.create_dataset("window_len", data=np.array(target_lengths, dtype=np.int64))
        f.attrs.update(
            model=mcfg["name"], source=PILE, n_examples=len(windows),
            n_layers=n_layers, d_model=d_model,
            token_position="final_generic_token", seed=args.seed,
        )
    run.log(f"wrote {out_h5}")

    run.save_json("tables/collect_general_stats.json", {
        "source": PILE, "n_examples": len(windows), "n_docs_pool": len(docs),
        "n_layers": n_layers, "d_model": d_model, "shape": list(traj.shape),
        "mib": round(traj.nbytes / 2**20, 1), "sanity_ok": f"{n_ok}/{len(checks)}",
        "length_min": int(min(target_lengths)), "length_median": int(np.median(target_lengths)),
        "length_max": int(max(target_lengths)), "hdf5": str(out_h5),
        "sample_windows": texts[:5],
    })
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
