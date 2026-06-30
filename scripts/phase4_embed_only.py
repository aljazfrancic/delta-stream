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

    # Embedding-only sources: zero every layer past ℓ0 (the lower-bound variant).
    u_embed = u.clone()
    u_embed[:, 1:, :] = 0.0

    fit_kw = dict(lambda_reg=args.lambda_reg, epochs=args.epochs, lr=args.lr,
                  bs=args.batch_size, device=device)
    out = {}
    for name, src in (("full", u), ("embed_only", u_embed)):
        model = LowRankSSM(SSMDims(L, D, M, r, r), use_state=True, seed=args.seed).to(device)
        _fit(model, src[tr], delta[tr], **fit_kw)
        tr_r2, _ = _r2_overall_and_per_layer(model, src[tr], delta[tr],
                                             bs=args.batch_size, device=device)
        te_r2, te_per = _r2_overall_and_per_layer(model, src[te], delta[te],
                                                  bs=args.batch_size, device=device)
        out[name] = {"train_r2": tr_r2, "test_r2": te_r2, "test_per_layer_r2": te_per.tolist()}
        run.log(f"  [{name:10s}] train R²={tr_r2:+.4f} test R²={te_r2:+.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_pl = np.array(out["full"]["test_per_layer_r2"])
    embed_pl = np.array(out["embed_only"]["test_per_layer_r2"])
    early = slice(0, L // 3)
    late = slice(2 * L // 3, L)
    summary = {
        "rank": r, "n_layers": L, "sources": args.sources_npz.name,
        "full_test_r2": out["full"]["test_r2"], "embed_only_test_r2": out["embed_only"]["test_r2"],
        "drop": round(out["full"]["test_r2"] - out["embed_only"]["test_r2"], 4),
        "full_per_layer_r2": out["full"]["test_per_layer_r2"],
        "embed_only_per_layer_r2": out["embed_only"]["test_per_layer_r2"],
        "embed_only_early_mean_r2": float(embed_pl[early].mean()),
        "embed_only_late_mean_r2": float(embed_pl[late].mean()),
        "full_early_mean_r2": float(full_pl[early].mean()),
        "full_late_mean_r2": float(full_pl[late].mean()),
        "interpretation": (
            "Embedding-only sources (s_0 only, propagated through the state chain) are a "
            "lower bound: if per-layer sources are load-bearing, removing them collapses "
            "reconstruction, most at LATE layers where Geva et al.'s built-up features "
            "live. The complement of the use_state=False ablation (which keeps sources, "
            "kills propagation)."
        ),
    }
    run.save_json("tables/embed_only.json", summary)
    _plot(run, full_pl, embed_pl)

    print("\n=== Embedding-only source ablation (frozen mlsae sources, rank "
          f"{r}) ===")
    print(f"  full sources:    test R² {summary['full_test_r2']:+.4f}  "
          f"(early {summary['full_early_mean_r2']:+.3f} / "
          f"late {summary['full_late_mean_r2']:+.3f})")
    print(f"  embedding-only:  test R² {summary['embed_only_test_r2']:+.4f}  "
          f"(early {summary['embed_only_early_mean_r2']:+.3f} / "
          f"late {summary['embed_only_late_mean_r2']:+.3f})")
    print(f"  drop from removing per-layer sources: {summary['drop']:+.4f}")
    return 0


def _plot(run, full_pl, embed_pl) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(full_pl)), full_pl, "-o", label="full per-layer sources")
    ax.plot(range(len(embed_pl)), embed_pl, "-s", label="embedding-only (s_0)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("layer ℓ"); ax.set_ylabel("held-out Δ_ℓ R²")
    ax.set_title("Embedding-only source ablation: late-layer collapse")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run.path("figures/embed_only_per_layer.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
