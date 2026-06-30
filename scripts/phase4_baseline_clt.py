#!/usr/bin/env python3
"""Phase 4 Step-1 Baseline D(b) — residual-write CLT vs the SSM, on-box (E3).

The *primary* §4.2 conceptual comparison, finally with a number. CLT places a **free**
low-rank decoder at every (source, target) layer pair ``ℓ' ≤ ℓ`` — the unconstrained
``O(L²)`` parameterisation our Branch-B state-space model is meant to compress. We fit
three decoders of the **same cached per-layer sparse sources** ``u`` (mlsae, frozen —
no LASSO, so the only thing that differs is the decoder parameterisation), each at the
same rank and each swept over ``λ_reg`` to its **best held-out** R²:

  * **per_layer** — diagonal-only (no cross-layer term): ``Δ̂_ℓ = b_ℓ + D_ℓ u_ℓ``
    (``LowRankSSM use_state=False``).
  * **ssm** — our first-order depth-indexed state-space decoder (``use_state=True``).
  * **clt_full** — the unconstrained lower-triangular CLT (``ResidualWriteCLT``).

R² is the project convention (pooled ``1 − ss_res/ss_tot`` vs the per-depth mean).
Reports held-out R², the train→test gap (overfit), and the **effective parameter
count** of each, so the §4.2 "shared-kernel compression ratio vs CLT's O(L²)" is visible.

    python scripts/phase4_baseline_clt.py --config configs/pythia-410m-deduped.yaml \
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz --desc ph4-e3-clt-410md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.clt import ResidualWriteCLT
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _free(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _r2(model, u, d, *, bs, device) -> float:
    """Pooled R² vs the per-depth mean of ``d`` (matches ``decompose.evaluate``).

    ``u``/``d`` are kept on CPU and streamed per minibatch — the full ``[N,L,M]``
    source tensor never sits on the 3 GB GPU alongside the model + Adam state.
    """
    mean = d.mean(dim=0, keepdim=True).to(device)
    ss_res = ss_tot = 0.0
    with torch.no_grad():
        for i in range(0, u.shape[0], bs):
            sl = slice(i, i + bs)
            db = d[sl].to(device)
            ss_res += ((db - model(u[sl].to(device))) ** 2).sum().item()
            ss_tot += ((db - mean) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_frozen(model, u_tr, d_tr, *, lambda_reg, epochs, lr, bs, device, log) -> None:
    """Adam-fit a decoder against frozen sources ``u_tr`` (no LASSO input phase).

    Sources/targets live on CPU and are moved minibatch-by-minibatch (the unconstrained
    CLT is 52M params; pinning the 530 MB source tensor on-GPU too would OOM the box).
    """
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
    log(f"    fit done ({epochs} epochs, λ_reg={lambda_reg:g})")


def _ssm_params(L, D, M, dz, r, *, use_state: bool) -> int:
    """Effective (graph-reachable) parameter count of LowRankSSM at the given flag."""
    per_layer = L * (D + M) * r + L * D  # D_ℓ factors + b_ℓ
    if not use_state:
        return per_layer
    state = 2 * (L - 1) * dz * r + (L - 1) * (dz + M) * r + L * (D + dz) * r
    return per_layer + state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--sources-npz", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reg-grid", type=float, nargs="+",
                        default=[1e-4, 1e-3, 1e-2, 1e-1])
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.sources_npz)
    u = torch.tensor(z["u"], dtype=torch.float32)
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    delta = torch.tensor(deltas(rs), dtype=torch.float32)
    N, L, M = u.shape
    D = delta.shape[2]
    r = args.rank
    run.log(f"sources {tuple(u.shape)} from {args.sources_npz.name}, delta {tuple(delta.shape)}, "
            f"rank={r}, device={device}")

    tr, te = train_test_split_idx(N, test_frac=0.2, seed=args.seed)
    u_tr, u_te = u[tr], u[te]            # kept on CPU; streamed per minibatch
    d_tr, d_te = delta[tr], delta[te]
    run.log(f"split: {len(tr)} train / {len(te)} test (same as Gate-3)")

    builders = {
        "per_layer": (lambda: LowRankSSM(SSMDims(L, D, M, r, r), use_state=False, seed=args.seed),
                      _ssm_params(L, D, M, r, r, use_state=False)),
        "ssm": (lambda: LowRankSSM(SSMDims(L, D, M, r, r), use_state=True, seed=args.seed),
                _ssm_params(L, D, M, r, r, use_state=True)),
        "clt_full": (lambda: ResidualWriteCLT(L, D, M, r, seed=args.seed),
                     L * (L + 1) // 2 * (D + M) * r + L * D),
    }

    rows = []
    for name, (ctor, n_params) in builders.items():
        run.log(f"[{name}] n_params={n_params/1e6:.2f}M — λ_reg sweep {args.reg_grid}")
        best = None
        for lam in args.reg_grid:
            model = ctor().to(device)
            _fit_frozen(model, u_tr, d_tr, lambda_reg=lam, epochs=args.epochs,
                        lr=args.lr, bs=args.batch_size, device=device, log=run.log)
            tr_r2 = _r2(model, u_tr, d_tr, bs=args.batch_size, device=device)
            te_r2 = _r2(model, u_te, d_te, bs=args.batch_size, device=device)
            run.log(f"  [{name}] λ_reg={lam:g}: train R²={tr_r2:+.4f} test R²={te_r2:+.4f}")
            if best is None or te_r2 > best["test_r2"]:
                best = {"lambda_reg": lam, "train_r2": tr_r2, "test_r2": te_r2}
            _free(model)
        rows.append({"model": name, "n_params": n_params, "n_params_M": round(n_params / 1e6, 2),
                     **best, "overfit_gap": round(best["train_r2"] - best["test_r2"], 4)})

    ssm = next(r for r in rows if r["model"] == "ssm")
    clt = next(r for r in rows if r["model"] == "clt_full")
    summary = {
        "rank": r, "n_layers": L, "d_model": D, "n_sources": M,
        "sources": args.sources_npz.name, "epochs": args.epochs, "reg_grid": args.reg_grid,
        "rows": rows,
        "ssm_vs_clt_test_r2_gap": round(ssm["test_r2"] - clt["test_r2"], 4),
        "clt_param_ratio_vs_ssm": round(clt["n_params"] / ssm["n_params"], 2),
        "reference_gate3_ssm_lasso_test_r2": 0.857,
        "interpretation": (
            "All three decode the SAME frozen per-layer mlsae sources at rank r, so the "
            "only difference is the decoder parameterisation. The unconstrained O(L²) CLT "
            "is strictly more expressive in-sample; whether it beats the state-space SSM "
            "HELD-OUT (vs its larger overfit) is the §4.2 question. Positive "
            "ssm_vs_clt_test_r2_gap with clt_param_ratio_vs_ssm > 1 = parsimony win."
        ),
    }
    run.save_json("tables/baseline_clt.json", summary)
    _plot(run, rows)

    print("\n=== Residual-write CLT vs SSM vs per-layer (frozen mlsae sources, rank "
          f"{r}) ===")
    print(f"{'model':<12}{'params':>10}{'best λ_reg':>12}{'train R²':>11}{'test R²':>10}"
          f"{'overfit':>9}")
    for row in rows:
        print(f"{row['model']:<12}{row['n_params_M']:>8.2f}M{row['lambda_reg']:>12g}"
              f"{row['train_r2']:>+11.4f}{row['test_r2']:>+10.4f}{row['overfit_gap']:>+9.4f}")
    print(f"\nSSM−CLT held-out R² gap: {summary['ssm_vs_clt_test_r2_gap']:+.4f}  "
          f"(CLT uses {summary['clt_param_ratio_vs_ssm']}× the SSM's params)")
    return 0


def _plot(run, rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for row in rows:
        ax.scatter(row["n_params"] / 1e6, row["test_r2"], s=80)
        ax.annotate(f"{row['model']}\n(train {row['train_r2']:.3f})",
                    (row["n_params"] / 1e6, row["test_r2"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("effective parameters (millions, log)")
    ax.set_ylabel("held-out Δ reconstruction R²")
    ax.set_title("Residual-write CLT vs state-space SSM (frozen sources, matched rank)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run.path("figures/clt_vs_ssm.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
