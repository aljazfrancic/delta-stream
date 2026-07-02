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
from layer_dynamics.diagnostics.bootstrap import paired_gap_ci, per_example_ss
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


def _fit_frozen(model, u_tr, d_tr, *, lambda_reg, epochs, lr, bs, device, log,
                eval_every: int = 0, eval_cb=None) -> None:
    """Adam-fit a decoder against frozen sources ``u_tr`` (no LASSO input phase).

    Sources/targets live on CPU and are moved minibatch-by-minibatch (the unconstrained
    CLT is 52M params; pinning the 530 MB source tensor on-GPU too would OOM the box).
    ``eval_every``/``eval_cb`` hook the R2 convergence curve (review item R2): every
    ``eval_every`` epochs (and at the end) ``eval_cb(epoch_done)`` is called.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = u_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(u_tr[idx].to(device))
            loss = ((d_tr[idx].to(device) - pred) ** 2).sum() + lambda_reg * model.operator_fro2()
            loss.backward()
            opt.step()
        if eval_cb is not None and eval_every > 0 and (ep + 1) % eval_every == 0:
            eval_cb(ep + 1)
    log(f"    fit done ({epochs} epochs, λ_reg={lambda_reg:g})")


def _per_example_ss(model, u_te, d_te, *, bs, device):
    """Per-example (ss_res, ss_tot) on the test split for the paired bootstrap (R0).

    ``ss_tot`` uses the per-depth mean of the test Δ — same convention as ``_r2``.
    """
    mean = d_te.mean(dim=0, keepdim=True)
    preds = []
    with torch.no_grad():
        for i in range(0, u_te.shape[0], bs):
            sl = slice(i, i + bs)
            preds.append(model(u_te[sl].to(device)).cpu())
    pred = torch.cat(preds).numpy()
    return per_example_ss(pred, d_te.numpy(), mean.numpy())


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
    parser.add_argument("--sources-npz", type=Path, default=None,
                        help="Cached compressed sources; omit to build on the fly "
                             "from --sae-repo (the 70m EleutherAI lane).")
    parser.add_argument("--sae-repo", type=str, default=None,
                        help="Build full-width sources from this SAE repo when no "
                             "--sources-npz is given (same path as gate3_pareto).")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reg-grid", type=float, nargs="+",
                        default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--models", type=str, nargs="+",
                        default=["per_layer", "ssm", "clt_full"],
                        help="Subset of decoder families to fit (R3 runs ssm+clt only).")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Subsample the train split to this many examples "
                             "(R3 sample-efficiency curve); test split unchanged.")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Log train/test R² every this many epochs "
                             "(R2 convergence curve; 0 = off).")
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Paired-bootstrap resamples for the gap CIs (R0).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    delta = torch.tensor(deltas(rs), dtype=torch.float32)
    if args.sources_npz is not None:
        z = np.load(args.sources_npz)
        u = torch.tensor(z["u"], dtype=torch.float32)
        src_name = args.sources_npz.name
    elif args.sae_repo is not None:  # 70m lane: full-width sources built on the fly
        from layer_dynamics.saes import build_sources

        u = torch.tensor(build_sources(rs, args.sae_repo), dtype=torch.float32)
        src_name = args.sae_repo
    else:
        raise SystemExit("need --sources-npz or --sae-repo")
    N, L, M = u.shape
    if delta.shape[1] > L:  # partial SAE coverage (e.g. Qwen 12L prefix of a 24L model)
        run.log(f"partial SAE coverage -> truncating delta to first {L} layers")
        delta = delta[:, :L]
    D = delta.shape[2]
    r = args.rank
    run.log(f"sources {tuple(u.shape)} from {src_name}, delta {tuple(delta.shape)}, "
            f"rank={r}, device={device}")

    tr, te = train_test_split_idx(N, test_frac=0.2, seed=args.seed)
    if args.n_train is not None:  # R3: tr is already a random permutation → prefix is random
        tr = tr[: args.n_train]
    u_tr, u_te = u[tr], u[te]            # kept on CPU; streamed per minibatch
    d_tr, d_te = delta[tr], delta[te]
    run.log(f"split: {len(tr)} train / {len(te)} test (same as Gate-3"
            + (f", train subsampled to {len(tr)}" if args.n_train else "") + ")")

    builders = {
        "per_layer": (lambda: LowRankSSM(SSMDims(L, D, M, r, r), use_state=False, seed=args.seed),
                      _ssm_params(L, D, M, r, r, use_state=False)),
        "ssm": (lambda: LowRankSSM(SSMDims(L, D, M, r, r), use_state=True, seed=args.seed),
                _ssm_params(L, D, M, r, r, use_state=True)),
        "clt_full": (lambda: ResidualWriteCLT(L, D, M, r, seed=args.seed),
                     L * (L + 1) // 2 * (D + M) * r + L * D),
    }
    builders = {k: v for k, v in builders.items() if k in args.models}

    rows = []
    conv_rows: list[dict] = []
    ss_arrays: dict[str, np.ndarray] = {}
    ss_tot_arr = None
    for name, (ctor, n_params) in builders.items():
        run.log(f"[{name}] n_params={n_params/1e6:.2f}M — λ_reg sweep {args.reg_grid}")
        best = None
        for lam in args.reg_grid:
            model = ctor().to(device)

            def _conv_cb(ep, *, _m=model, _name=name, _lam=lam):
                trr = _r2(_m, u_tr, d_tr, bs=args.batch_size, device=device)
                ter = _r2(_m, u_te, d_te, bs=args.batch_size, device=device)
                conv_rows.append({"model": _name, "lambda_reg": _lam, "epoch": ep,
                                  "train_r2": trr, "test_r2": ter})
                run.log(f"    [{_name}] λ={_lam:g} ep{ep}: train {trr:+.4f} test {ter:+.4f}")

            _fit_frozen(model, u_tr, d_tr, lambda_reg=lam, epochs=args.epochs,
                        lr=args.lr, bs=args.batch_size, device=device, log=run.log,
                        eval_every=args.eval_every,
                        eval_cb=_conv_cb if args.eval_every > 0 else None)
            tr_r2 = _r2(model, u_tr, d_tr, bs=args.batch_size, device=device)
            te_r2 = _r2(model, u_te, d_te, bs=args.batch_size, device=device)
            run.log(f"  [{name}] λ_reg={lam:g}: train R²={tr_r2:+.4f} test R²={te_r2:+.4f}")
            if best is None or te_r2 > best["test_r2"]:
                best = {"lambda_reg": lam, "train_r2": tr_r2, "test_r2": te_r2}
                ss_res, ss_tot = _per_example_ss(model, u_te, d_te,
                                                 bs=args.batch_size, device=device)
                ss_arrays[name] = ss_res
                ss_tot_arr = ss_tot
            _free(model)
        rows.append({"model": name, "n_params": n_params, "n_params_M": round(n_params / 1e6, 2),
                     **best, "overfit_gap": round(best["train_r2"] - best["test_r2"], 4)})

    # R0: paired bootstrap CIs over test examples for every fitted pair
    cis = {}
    pairs = [("ssm", "clt_full"), ("ssm", "per_layer"), ("clt_full", "per_layer")]
    for a, b in pairs:
        if a in ss_arrays and b in ss_arrays:
            cis[f"{a}_minus_{b}"] = paired_gap_ci(
                ss_arrays[a], ss_arrays[b], ss_tot_arr, n_boot=args.n_boot, seed=args.seed)
    np.savez_compressed(
        run.path("tables/per_example_ss.npz"),
        test_idx=te, ss_tot=ss_tot_arr,
        **{f"ss_res_{k}": v for k, v in ss_arrays.items()})

    by_name = {row["model"]: row for row in rows}
    ssm, clt = by_name.get("ssm"), by_name.get("clt_full")
    summary = {
        "rank": r, "n_layers": L, "d_model": D, "n_sources": M,
        "sources": src_name, "epochs": args.epochs, "reg_grid": args.reg_grid,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "rows": rows,
        "paired_bootstrap_cis": cis,
        "reference_gate3_ssm_lasso_test_r2": 0.857,
        "interpretation": (
            "All families decode the SAME frozen per-layer sources at rank r, so the "
            "only difference is the decoder parameterisation. The unconstrained O(L²) CLT "
            "is strictly more expressive in-sample; whether it beats the state-space SSM "
            "HELD-OUT (vs its larger overfit) is the §4.2 question. Positive "
            "ssm_vs_clt_test_r2_gap with clt_param_ratio_vs_ssm > 1 = parsimony win."
        ),
    }
    if ssm and clt:
        summary["ssm_vs_clt_test_r2_gap"] = round(ssm["test_r2"] - clt["test_r2"], 4)
        summary["clt_param_ratio_vs_ssm"] = round(clt["n_params"] / ssm["n_params"], 2)
    run.save_json("tables/baseline_clt.json", summary)
    if conv_rows:
        import pandas as pd

        pd.DataFrame(conv_rows).to_csv(run.path("tables/convergence.csv"), index=False)
    _plot(run, rows)

    print("\n=== Residual-write CLT vs SSM vs per-layer (frozen mlsae sources, rank "
          f"{r}) ===")
    print(f"{'model':<12}{'params':>10}{'best λ_reg':>12}{'train R²':>11}{'test R²':>10}"
          f"{'overfit':>9}")
    for row in rows:
        print(f"{row['model']:<12}{row['n_params_M']:>8.2f}M{row['lambda_reg']:>12g}"
              f"{row['train_r2']:>+11.4f}{row['test_r2']:>+10.4f}{row['overfit_gap']:>+9.4f}")
    if "ssm_vs_clt_test_r2_gap" in summary:
        print(f"\nSSM−CLT held-out R² gap: {summary['ssm_vs_clt_test_r2_gap']:+.4f}  "
              f"(CLT uses {summary['clt_param_ratio_vs_ssm']}× the SSM's params)")
    for key, ci in cis.items():
        star = " *" if ci["significant"] else ""
        print(f"CI [{key}]: gap {ci['gap']:+.4f}  95% [{ci['ci_lo']:+.4f}, "
              f"{ci['ci_hi']:+.4f}]{star}")
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
    fig.savefig(run.path("figures/clt_vs_ssm.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
