"""Phase 1 diagnostics: ridge predictability, CCA, AR/Granger."""

from layer_dynamics.diagnostics.autoregress import (
    ARFit,
    fit_ar_pooled,
    fit_delta_pca,
    granger_lag_grid,
    pca_project,
    scramble_per_depth,
    wilks_rao_f,
)
from layer_dynamics.diagnostics.cca import fit_cca, regularized_cca, shrink_cov
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.lowrank_ridge import fit_lowrank_ridge, pooled_r2

__all__ = [
    "ARFit",
    "deltas",
    "fit_ar_pooled",
    "fit_cca",
    "fit_delta_pca",
    "fit_lowrank_ridge",
    "granger_lag_grid",
    "load_residual_stream",
    "pca_project",
    "pooled_r2",
    "regularized_cca",
    "scramble_per_depth",
    "shrink_cov",
    "train_test_split_idx",
    "wilks_rao_f",
]
