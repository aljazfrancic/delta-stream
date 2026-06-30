#!/usr/bin/env python3
"""Embedding-only source ablation — the Phase-3 Step-2 pre-registered lower bound (E5).

[phase-3 Step 2] registers a variant with sources only at ``s_0`` (all ``u_ℓ = 0`` for
ℓ > 0), "reported as a lower bound": per [proposal §4.3] (Geva et al. 2023) subject- and
relation-enrichment features are *built up* across early-to-middle layers, not present at
the embedding, so a model that may only inject sources at layer 0 (and must propagate them
forward through the state chain) should under-fit late-layer updates. This is the dual of
the Gate-3 ``use_state=False`` ablation: that one keeps all per-layer sources but kills
cross-layer propagation; this one keeps propagation but kills all sources past ℓ0.

Both the full and embedding-only SSMs are fit on the **same frozen mlsae sources** with the
same Adam recipe as the E3 CLT comparison, so the gap isolates exactly the value of the
per-layer sources. Reports overall held-out R² and the **per-layer** R² breakdown (the
late-layer collapse is the pre-registered prediction).

    python scripts/phase4_embed_only.py --config configs/pythia-410m-deduped.yaml \
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz --desc ph4-e5-embed-410md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _fit(model, u_tr, d_tr, *, lambda_reg, epochs, lr, bs, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = u_tr.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(u_tr[idx].to(device))
            loss = ((d_tr[idx].to(device) - pred) ** 2).sum() + lambda_reg * model.operator_fro2()
            loss.backward()
            opt.step()


def _r2_overall_and_per_layer(model, u, d, *, bs, device):
    """Pooled R² and per-layer R², both vs the per-depth mean (project convention)."""
    mean = d.mean(dim=0, keepdim=True).to(device)
    L = d.shape[1]
    res = torch.zeros(L, device=device)
    tot = torch.zeros(L, device=device)
    with torch.no_grad():
        for i in range(0, u.shape[0], bs):
            sl = slice(i, i + bs)
            db = d[sl].to(device)
            err = (db - model(u[sl].to(device))) ** 2
            res += err.sum(dim=(0, 2))
            tot += ((db - mean) ** 2).sum(dim=(0, 2))
    per_layer = (1.0 - res / tot).cpu().numpy()
    overall = float(1.0 - res.sum().item() / tot.sum().item())
    return overall, per_layer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--sources-npz", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[0, 3, 7, 11, 15, 19, 23],
                        help="Source-prefix cutoffs ℓ_max: keep sources for ℓ≤ℓ_max, zero the "
                             "rest. ℓ_max=0 is the embedding-only lower bound; ℓ_max=L-1 is full.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.sources_npz)
    u = torch.tensor(z["u"], dtype=torch.float32)
    rs, _, _ = load_residual_stream(Path("data") / mcfg["name"] / "residual_stream.h5")
    delta = torch.tensor(deltas(rs), dtype=torch.float32)
    N, L, M = u.shape
    D = delta.shape[2]
    r = args.rank
    tr, te = train_test_split_idx(N, test_frac=0.2, seed=args.seed)
    run.log(f"sources {tuple(u.shape)}, delta {tuple(delta.shape)}, rank={r}, device={device}")

    # Source-prefix sweep: for each cutoff ℓ_max, keep sources for ℓ≤ℓ_max and zero the
    # rest, then fit the SSM (which may still propagate the kept sources forward through
    # the state chain). ℓ_max=0 is the pre-registered embedding-only lower bound; ℓ_max=L-1
    # is the full model. The buildup curve shows whether late-layer sources are load-bearing.
    fit_kw = dict(lambda_reg=args.lambda_reg, epochs=args.epochs, lr=args.lr,
                  bs=args.batch_size, device=device)
    cutoffs = sorted({min(c, L - 1) for c in args.cutoffs})
    rows, per_layer = [], {}
    for c in cutoffs:
        src = u.clone()
        if c + 1 < L:
            src[:, c + 1 :, :] = 0.0
        model = LowRankSSM(SSMDims(L, D, M, r, r), use_state=True, seed=args.seed).to(device)
        _fit(model, src[tr], delta[tr], **fit_kw)
        tr_r2, _ = _r2_overall_and_per_layer(model, src[tr], delta[tr],
                                             bs=args.batch_size, device=device)
        te_r2, te_per = _r2_overall_and_per_layer(model, src[te], delta[te],
                                                  bs=args.batch_size, device=device)
        rows.append({"cutoff": c, "n_source_layers": c + 1, "train_r2": tr_r2, "test_r2": te_r2})
        per_layer[c] = te_per.tolist()
        run.log(f"  [ℓ_max={c:2d}] sources ℓ≤{c}: train R²={tr_r2:+.4f} test R²={te_r2:+.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full = max(rows, key=lambda x: x["cutoff"])
    embed = min(rows, key=lambda x: x["cutoff"])
    summary = {
        "rank": r, "n_layers": L, "sources": args.sources_npz.name, "cutoffs": cutoffs,
        "rows": rows, "test_per_layer_r2_by_cutoff": per_layer,
        "full_test_r2": full["test_r2"], "embed_only_test_r2": embed["test_r2"],
        "drop_full_minus_embed": round(full["test_r2"] - embed["test_r2"], 4),
        "interpretation": (
            "Source-prefix sweep: reconstruction R² as a function of how many layers' "
            "sources are available (ℓ≤ℓ_max), with the SSM free to propagate them forward. "
            "ℓ_max=0 is the pre-registered embedding-only lower bound — and since v_0 has no "
            "SAE (u_0=0 in every available dictionary), it is the example-independent floor: "
            "with no source to inject, the input-free state trajectory is the same for every "
            "example. The curve's rise with ℓ_max measures how much each band of per-layer "
            "sources is load-bearing; if it only saturates near ℓ_max=L-1, late-layer sources "
            "(Geva et al.'s built-up features) carry real, non-propagable reconstruction value. "
            "Complement of the use_state=False ablation (keeps sources, kills propagation)."
        ),
    }
    run.save_json("tables/embed_only.json", summary)
    _plot(run, rows, np.array(per_layer[full["cutoff"]]), np.array(per_layer[embed["cutoff"]]))

    print("\n=== Source-prefix sweep (frozen mlsae sources, rank "
          f"{r}) ===")
    for row in rows:
        print(f"  ℓ_max={row['cutoff']:2d} ({row['n_source_layers']:2d} source layers): "
              f"test R²={row['test_r2']:+.4f}")
    print(f"  embedding-only (ℓ_max=0) {summary['embed_only_test_r2']:+.4f}  →  "
          f"full (ℓ_max={L-1}) {summary['full_test_r2']:+.4f}  "
          f"(buildup {summary['drop_full_minus_embed']:+.4f})")
    return 0


def _plot(run, rows, full_pl, embed_pl) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    cut = [r["cutoff"] for r in rows]
    ax1.plot(cut, [r["test_r2"] for r in rows], "-o", color="tab:blue")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("source-prefix cutoff ℓ_max (sources kept for ℓ≤ℓ_max)")
    ax1.set_ylabel("held-out Δ reconstruction R²")
    ax1.set_title("Reconstruction builds up as per-layer sources are added")
    ax1.grid(alpha=0.3)
    ax2.plot(range(len(full_pl)), full_pl, "-o", label="full (all sources)")
    ax2.plot(range(len(embed_pl)), embed_pl, "-s", label="embedding-only (ℓ_max=0)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("layer ℓ"); ax2.set_ylabel("held-out Δ_ℓ R²")
    ax2.set_title("Per-layer R²: endpoints")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(run.path("figures/source_prefix_sweep.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
