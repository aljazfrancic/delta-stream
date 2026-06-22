# Phase 3 architecture decision (locked at Gate 2)

**Branch B — first-order, fully depth-indexed state-space model.**

Phase 3 fits the depth dynamics as a state-space model
$`z_{\ell+1} = A_\ell z_\ell + B_\ell u_\ell`$ for $`\ell \in \{0,\dots,L-2\}`$ with the
observation/decode $`\Delta_\ell \approx b_\ell + C_\ell z_\ell + D_\ell u_\ell`$ for
$`\ell \in \{0,\dots,L-1\}`$ (the feedthrough $`D_\ell`$ is the Branch-A $`W_0`$
analogue; $`b_\ell`$ is the unpenalised per-layer bias absorbing $`\bar\Delta_\ell`$;
$`z_0`$ free or zero). The per-layer operators $`A_\ell,B_\ell,C_\ell,D_\ell`$ are
**free per layer (NOT depth-smoothed / interpolated across control depths)** —
Gate-2 Step 1 found the AR kernel is *pervasively* depth-dependent (cross-band
held-out R² is negative in every direction and rank-robust; between-band kernel
distance ≈ √2 with the three pairwise distances ≈ equal, i.e. no smooth drift and no
clean segment boundary), so the smoothness the interpolation assumes is unsupported;
the depth-smoothed variant is retained only as an ablation if a finer per-depth
kernel-similarity analysis later reveals exploitable smoothness. Each operator is
**low-rank** via factorisation $`A_\ell = A_\ell^{(1)}(A_\ell^{(2)})^\top`$ (and
analogously $`B,C,D`$), rank swept $`r \in \{16,32,64,128\}`$ — **extend the sweep to
256 if it has not saturated**, since Gate-2 Step 1's r-vs-2r spot-check promoted the
AR working rank 128→256. The SSM is **first order** in the depth latent — no kernel
length $`K`$ to sweep — which Gate-2 Step 2's $`k^\ast \approx 1`$ directly supports;
the latent-state dimension $`d_z`$ is the main capacity knob (start $`d_z \sim r`$).
Inputs $`u_\ell`$ initialise from per-layer SAE latents of $`v_\ell`$ (Phase 3 Step 2);
$`u_\ell`$ carries the L1 sparsity (per-example coupled LASSO across depths). Optimise
the consolidated loss $`\mathcal{L}_B`$ (Phase 3 Step 1) by the alternating pattern —
one Adam sweep over parameters $`\{A_\ell,B_\ell,C_\ell,D_\ell,b_\ell,z_0\}`$ with
$`\{u_\ell\}`$ fixed (autodiff through the unrolled SSM), then one proximal-gradient
LASSO pass over $`\{u_\ell\}`$ with parameters fixed — never co-optimised in one call.
Debug on Pythia-70m, primary run on Pythia-410m.

**Parsimony caveat.** Because H2 failed, this is proposal §4.2's "H2 fails → CLT's free
parameters are load-bearing" branch: the fully depth-indexed Branch B is *larger* than
a stationary kernel by design, so do **not** drop the Branch-A parameter-match
criterion's intent silently — report Branch B's parameter count honestly against CLT in
Phase 4 and do not equate a reconstruction win with a parsimony win.
