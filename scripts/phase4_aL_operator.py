#!/usr/bin/env python3
"""Characterise the learned depth-propagation operator A_ℓ (Phase 4 — positive result).

Phase 4's causal existence proof landed NULL (recall is a distributed, mass-driven
code), so the decomposition's demonstrated value is reconstruction. This adds the
*positive* structural readout that balances it: a description of the cross-layer
state-transition operator the SSM learned. ``A_ℓ = A1_ℓ A2_ℓᵀ`` maps ``z_ℓ→z_{ℓ+1}``.

**Corrected 2026-07-09.** The original version of this script asserted that ρ(A_ℓ) is
gauge-invariant. It is not: the depth-indexed SSM's symmetry is a *per-layer*
``T_ℓ ∈ GL(d_z)`` (``A_ℓ → T_{ℓ+1} A_ℓ T_ℓ⁻¹``), so only a global ``T`` acts by
similarity. The script now (a) labels ρ as a property of the fitted min-Frobenius
gauge, (b) leads with the genuinely invariant state-path kernel and its memory
constant τ_inv, and (c) runs a gauge audit that perturbs the checkpoint with an
output-preserving scalar gauge and reports what moves. Headline questions:

  1. How fast does the *realised* state path attenuate with depth gap (invariant)?
  2. Does that memory horizon explain the reconstruction k*≈1 (Step 2)?
  3. Is the fitted spectrum reproducible across seeds and dictionaries — and by a
     statistic stronger than the near-vacuous sorted-modulus cosine?

Runs on the saved Branch-B checkpoints — no model forward, seconds on CPU.

    python scripts/phase4_aL_operator.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-aL-operator-410md

Outputs: tables/{spectrum_by_depth,state_path_decay,seed_stability}.csv,
tables/aL_stats.json, figures/{eig_plane,rho_by_depth,modulus_hist,state_decay}.svg.
"""

from __future__ import annotations

import argparse

import numpy as np

from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.operator_spectrum import (
    apply_scalar_gauge,
    spectrum_by_depth,
    spectrum_cosine,
    spectrum_rel_diff,
    state_path_decay,
    state_path_tau,
)
from layer_dynamics.runlog import init_run

L1_CKPT = "runs/2026-06-30-ph4-e2-checkpoint-l1-410md/checkpoints/ssm_seed0.pt"
MLSAE_CKPTS = [
    "runs/2026-06-26-ph4-checkpoint-410md/checkpoints/ssm_seed0.pt",
    "runs/2026-06-29-ph4-gauge-410md/checkpoints/ssm_seed1.pt",
    "runs/2026-06-29-ph4-gauge-410md/checkpoints/ssm_seed2.pt",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--l1-ckpt", type=str, default=L1_CKPT,
                        help="Primary (clean L1-sourced) SSM checkpoint.")
    parser.add_argument("--mlsae-ckpts", type=str, nargs="+", default=MLSAE_CKPTS,
                        help="mlsae-sourced SSM checkpoints (seeds 0/1/2) for stability.")
    parser.add_argument("--max-gap", type=int, default=8,
                        help="Max depth gap for the state-path memory kernel.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)

    import pandas as pd

    # ---- Primary: the clean L1-sourced operator ----
    l1 = load_ssm(args.l1_ckpt)
    spec = spectrum_by_depth(l1)
    decay = state_path_decay(l1, max_gap=args.max_gap)
    tau_inv = state_path_tau(decay)
    run.log(f"L1 operator: d_z={spec['d_state']}, {spec['n_layers_trans']} transitions")
    run.log("  [INVARIANT] state-path memory kernel ‖C(∏A)B‖_F ratio to gap1: "
            f"{np.round(decay['ratio_to_gap1'], 3).tolist()}")
    run.log(f"  [INVARIANT] gap2/gap1 = {decay['decay_gap2_over_gap1']:.3f}; memory "
            f"τ_inv = {tau_inv:.2f} layers  (vs measured k*≈1: Step-2 lag≥2 adds only "
            "+0.018 R² median)")
    run.log(f"  [GAUGE-DEPENDENT] spectral radius ρ(A_ℓ) in the fitted min-Frobenius "
            f"gauge: median {spec['rho_median']:.3f}, max {spec['rho_max']:.3f}, geomean "
            f"{spec['rho_geomean']:.3f}; layers with ρ≥1: "
            f"{spec['max_over_unit'] * spec['n_layers_trans']:.0f}/{spec['n_layers_trans']}")

    # ---- Gauge audit: an output-preserving per-layer scalar gauge moves ρ, not the
    # state path. This is the check the original (wrong) gauge-invariance claim lacked.
    gauged = load_ssm(args.l1_ckpt)
    rng = np.random.default_rng(args.seed)
    apply_scalar_gauge(gauged, np.exp(rng.normal(0.0, 1.0, size=gauged.dims.n_layers)))
    g_spec = spectrum_by_depth(gauged)
    g_decay = state_path_decay(gauged, max_gap=args.max_gap)
    gauge_audit = {
        "rho_median": [spec["rho_median"], g_spec["rho_median"]],
        "rho_max": [spec["rho_max"], g_spec["rho_max"]],
        "noncontractive_layers": [
            int(round(spec["max_over_unit"] * spec["n_layers_trans"])),
            int(round(g_spec["max_over_unit"] * g_spec["n_layers_trans"])),
        ],
        "state_path_ratio_max_abs_change": float(
            np.max(np.abs(decay["ratio_to_gap1"] - g_decay["ratio_to_gap1"]))),
        "tau_inv": [tau_inv, state_path_tau(g_decay)],
    }
    run.log(f"  gauge audit (output-preserving scalar gauge): ρ_median "
            f"{gauge_audit['rho_median'][0]:.3f}→{gauge_audit['rho_median'][1]:.3f}, "
            f"ρ_max {gauge_audit['rho_max'][0]:.3f}→{gauge_audit['rho_max'][1]:.3f}, "
            f"non-contractive {gauge_audit['noncontractive_layers'][0]}→"
            f"{gauge_audit['noncontractive_layers'][1]}; state-path ratio unchanged to "
            f"{gauge_audit['state_path_ratio_max_abs_change']:.2e}")

    pd.DataFrame({
        "layer": np.arange(spec["n_layers_trans"]),
        "rho": spec["rho"], "mean_mod": spec["mean_mod"], "frac_complex": spec["frac_complex"],
    }).to_csv(run.path("tables/spectrum_by_depth.csv"), index=False)
    pd.DataFrame({
        "gap": decay["gaps"], "mean_fro": decay["mean_fro"],
        "median_fro": decay["median_fro"], "ratio_to_gap1": decay["ratio_to_gap1"],
    }).to_csv(run.path("tables/state_path_decay.csv"), index=False)

    # ---- Cross-seed stability (mlsae 0/1/2) + cross-dictionary (L1 vs mlsae) ----
    mlsae_models = [load_ssm(p) for p in args.mlsae_ckpts]
    mlsae_specs = [spectrum_by_depth(m) for m in mlsae_models]
    mlsae_decays = [state_path_decay(m, max_gap=args.max_gap) for m in mlsae_models]
    seed_rows = []
    pair_cos, pair_rel = [], []
    for i in range(len(mlsae_specs)):
        for k in range(i + 1, len(mlsae_specs)):
            c = spectrum_cosine(mlsae_specs[i]["moduli_sorted"], mlsae_specs[k]["moduli_sorted"])
            rd = spectrum_rel_diff(mlsae_specs[i]["moduli_sorted"],
                                   mlsae_specs[k]["moduli_sorted"])
            rc = float(np.corrcoef(mlsae_specs[i]["rho"], mlsae_specs[k]["rho"])[0, 1])
            pair_cos.append(c); pair_rel.append(rd)
            seed_rows.append({"pair": f"mlsae_s{i}_s{k}", "modulus_cosine": c,
                              "modulus_rel_diff": rd, "rho_corr": rc})
    l1_vs_mlsae = spectrum_cosine(spec["moduli_sorted"], mlsae_specs[0]["moduli_sorted"])
    l1_rel = spectrum_rel_diff(spec["moduli_sorted"], mlsae_specs[0]["moduli_sorted"])
    l1_rho_corr = float(np.corrcoef(spec["rho"], mlsae_specs[0]["rho"])[0, 1])
    seed_rows.append({"pair": "L1_vs_mlsae_s0", "modulus_cosine": l1_vs_mlsae,
                      "modulus_rel_diff": l1_rel, "rho_corr": l1_rho_corr})
    pd.DataFrame(seed_rows).to_csv(run.path("tables/seed_stability.csv"), index=False)

    # A null for the cosine: two independent sorted |N(0,1)| spectra of the same length.
    n_mod = spec["moduli_sorted"].size
    cos_null = float(np.mean([
        spectrum_cosine(np.sort(np.abs(rng.normal(size=n_mod)))[::-1],
                        np.sort(np.abs(rng.normal(size=n_mod)))[::-1]) for _ in range(20)]))
    gap2 = [d["decay_gap2_over_gap1"] for d in mlsae_decays] + [decay["decay_gap2_over_gap1"]]
    taus = [state_path_tau(d) for d in mlsae_decays] + [tau_inv]
    run.log(f"  cross-seed modulus cosine (mlsae 0/1/2): mean {np.mean(pair_cos):.3f} "
            f"— but the RANDOM-SPECTRA null is {cos_null:.3f}, so quote the median "
            f"relative difference instead: {np.mean(pair_rel):.3f}")
    run.log(f"  cross-dictionary L1↔mlsae: modulus cosine {l1_vs_mlsae:.3f}, rel-diff "
            f"{l1_rel:.3f}, ρ-profile corr {l1_rho_corr:.3f}")
    run.log(f"  [INVARIANT] gap2/gap1 across seeds+dictionaries: "
            f"{np.round(gap2, 3).tolist()}; τ_inv {np.round(taus, 2).tolist()} layers")

    stats = {
        "l1_ckpt": args.l1_ckpt, "mlsae_ckpts": args.mlsae_ckpts,
        "d_state": spec["d_state"], "n_transitions": spec["n_layers_trans"],
        "rho_median": spec["rho_median"], "rho_max": spec["rho_max"],
        "rho_geomean": spec["rho_geomean"], "layers_noncontractive": float(spec["max_over_unit"]),
        "mean_mod_median": float(np.median(spec["mean_mod"])),
        "frac_complex_mean": float(np.mean(spec["frac_complex"])),
        "state_path_ratio_to_gap1": decay["ratio_to_gap1"].tolist(),
        "state_path_gap2_over_gap1": decay["decay_gap2_over_gap1"],
        "state_path_tau_layers_invariant": tau_inv,
        "state_path_gap2_all_fits": [float(g) for g in gap2],
        "state_path_tau_all_fits": [float(t) for t in taus],
        "gauge_audit": gauge_audit,
        "cross_seed_modulus_cosine_mean": float(np.mean(pair_cos)),
        "cross_seed_modulus_cosine_pairs": [float(c) for c in pair_cos],
        "cross_seed_modulus_cosine_random_null": cos_null,
        "cross_seed_modulus_rel_diff_mean": float(np.mean(pair_rel)),
        "l1_vs_mlsae_modulus_cosine": l1_vs_mlsae,
        "l1_vs_mlsae_modulus_rel_diff": l1_rel, "l1_vs_mlsae_rho_corr": l1_rho_corr,
        "interpretation": (
            "The gauge-invariant result: the realised state path ‖C(∏A)B‖_F falls by "
            "~2× per layer of depth gap (gap2/gap1 ≈ 0.42–0.49 across seeds and both "
            "dictionaries), an e-folding memory of τ_inv ≈ 1.1–1.2 layers — the "
            "operator-level mechanism consistent with the reconstruction k*≈1 (Step 2). "
            "ρ(A_ℓ)<1 at every layer is ALSO true, but only of the min-Frobenius "
            "representative the λ_reg penalty selects: the depth-indexed SSM's gauge is "
            "per-layer (A_ℓ → T_{ℓ+1} A_ℓ T_ℓ⁻¹), and the gauge audit above turns 0/23 "
            "non-contractive layers into 5/23 without changing a single prediction. "
            "Likewise the 0.999 sorted-modulus cosine is near-vacuous (random spectra "
            "score the same); the honest cross-seed statistic is the ~10% median "
            "relative difference."
        ),
    }
    run.save_json("tables/aL_stats.json", stats)
    _plots(run, spec, decay, mlsae_specs)

    print(f"\n=== A_ℓ operator (L1-sourced, d_z={spec['d_state']}, "
          f"{spec['n_layers_trans']} transitions) ===")
    print(f"  [invariant] state-path decay gap2/gap1 = {decay['decay_gap2_over_gap1']:.3f}, "
          f"τ_inv = {tau_inv:.2f} layers  → mechanism for k*≈1")
    print(f"  [fitted gauge] ρ: median {spec['rho_median']:.3f}  max {spec['rho_max']:.3f}  "
          f"geomean {spec['rho_geomean']:.3f}  noncontractive layers "
          f"{spec['max_over_unit'] * spec['n_layers_trans']:.0f}"
          f"  → under a scalar gauge: {gauge_audit['noncontractive_layers'][1]}")
    print(f"  cross-seed modulus rel-diff {np.mean(pair_rel):.3f} (cosine "
          f"{np.mean(pair_cos):.3f}, random-spectra null {cos_null:.3f})")
    return 0


def _plots(run, spec, decay, mlsae_specs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = spec["n_layers_trans"]

    # 1. Eigenvalues in the complex plane, coloured by depth, with the unit circle.
    fig, ax = plt.subplots(figsize=(5.2, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.7, alpha=0.6)
    depth = np.repeat(np.arange(n), spec["d_state"])
    sc = ax.scatter(spec["eig_real"].reshape(-1), spec["eig_imag"].reshape(-1),
                    c=depth, cmap="viridis", s=8, alpha=0.7)
    ax.set_aspect("equal"); ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("Re λ"); ax.set_ylabel("Im λ")
    ax.set_title("A_ℓ eigenvalues, fitted min-Frobenius gauge")
    fig.colorbar(sc, ax=ax, label="depth ℓ", shrink=0.8)
    fig.tight_layout()
    fig.savefig(run.path("figures/eig_plane.svg"), bbox_inches="tight")
    plt.close(fig)

    # 2. Spectral radius by depth (L1 + mlsae seed band).
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(n), spec["rho"], "-o", color="tab:red", label="L1 (clean)")
    mrho = np.array([s["rho"] for s in mlsae_specs])
    ax.fill_between(np.arange(n), mrho.min(0), mrho.max(0), color="tab:blue", alpha=0.2,
                    label="mlsae seeds 0/1/2 range")
    ax.plot(np.arange(n), mrho.mean(0), "--", color="tab:blue", lw=1, label="mlsae mean")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("transition layer ℓ"); ax.set_ylabel("spectral radius ρ(A_ℓ)")
    ax.set_title("ρ(A_ℓ)<1 everywhere — in the fitted gauge (not gauge-invariant)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run.path("figures/rho_by_depth.svg"), bbox_inches="tight")
    plt.close(fig)

    # 3. Modulus histogram (pooled).
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spec["moduli_sorted"], bins=40, color="tab:purple", alpha=0.8)
    ax.axvline(spec["rho_median"], color="k", ls="--", lw=1,
               label=f"median ρ {spec['rho_median']:.2f}")
    ax.set_xlabel("|λ|"); ax.set_ylabel("count")
    ax.set_title("Eigenvalue moduli (most modes strongly contractive)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run.path("figures/modulus_hist.svg"), bbox_inches="tight")
    plt.close(fig)

    # 4. State-path memory kernel by gap.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(decay["gaps"], decay["ratio_to_gap1"], "-o", color="tab:green")
    ax.set_xlabel("depth gap g = ℓ − j (source→observation)")
    ax.set_ylabel("‖C(∏A)B‖_F  (ratio to gap 1)")
    ax.set_title("State memory kernel decays fast → k*≈1")
    fig.tight_layout()
    fig.savefig(run.path("figures/state_decay.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
