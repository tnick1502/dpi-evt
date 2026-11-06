"""Feature matrices shared by the internal and external data adapters."""
from typing import Dict
import numpy as np
import pandas as pd
from dpi_evt.constants import LOAD_NAMES, SOIL_NAMES

def build_feature_matrices(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    delta_cycles: np.ndarray,
    csr: np.ndarray,
    observations: Dict[str, np.ndarray],
    prefix_len: int,
    max_cycle_reference: float = 3000.0,
) -> Dict[str, object]:
    """Build static, prefix-summary and sequential feature matrices.

    Static inputs combine soil/loading properties and one-hot categories. The prefix
    summary contains mean, last value, peak, log-cycle slope, variance and coverage.
    Sequence inputs contain CSR, normalized log cycles and increments, and masked
    pressure observations.

    :param soil_df: soil-property table
    :param load_df: loading-property table
    :param cycles: cycle grid, shape (n, seq_len)
    :param delta_cycles: cycle increments, shape (n, seq_len)
    :param csr: CSR history, shape (n, seq_len)
    :param observations: prefix observations and masks
    :param prefix_len: observed prefix length
    :return: feature matrices and their column names
    """
    n = len(soil_df)
    soil_onehot = np.eye(len(SOIL_NAMES), dtype=np.float32)[soil_df["class_id"].to_numpy()]
    mode_onehot = np.eye(len(LOAD_NAMES), dtype=np.float32)[load_df["mode_id"].to_numpy()]

    prefix_mask = observations["prefix_mask"]
    prefix_obs = observations["prefix_obs"]
    observed_prefix_count = prefix_mask.sum(axis=1)
    prefix_count = np.clip(observed_prefix_count, 1.0, None)
    prefix_mean = (prefix_obs.sum(axis=1) / prefix_count).astype(np.float32)
    last_prefix_idx = np.maximum(prefix_count.astype(int) - 1, 0)
    first_prefix = prefix_obs[:, 0]
    last_prefix = prefix_obs[np.arange(n), last_prefix_idx]
    prefix_peak = prefix_obs.max(axis=1)
    prefix_var = (
        (np.square(prefix_obs - prefix_mean[:, None]) * prefix_mask).sum(axis=1) / prefix_count
    ).astype(np.float32)
    delta_log_n = np.log1p(cycles[np.arange(n), last_prefix_idx]) - np.log1p(cycles[:, 0])
    prefix_slope = ((last_prefix - first_prefix) / np.maximum(delta_log_n, 1e-3)).astype(np.float32)
    prefix_coverage = np.minimum(observed_prefix_count / prefix_len, 1.0).astype(np.float32)

    static_feature_names = (
        [
            "e",
            "D_r",
            "I_p",
            "V_s",
            "xi",
            "sigma_eff",
            "log10_permeability",
            # Additional geotechnical inputs for the CRR property model.
            "fines_content",
            "clay_fraction",
            "log10_Cu",
            "K0",
            "Vs1",
            "static_shear_ratio",
            "crr_alpha",
            "crr_betta",
            "CSR_base",
            "frequency",
            "amp_scale",
            # N_max reflects record duration and is excluded from static inputs.
            # Normalize with the common max_cycle_reference horizon.
        ]
        + [f"soil_{name}" for name in SOIL_NAMES]
        + [f"mode_{name}" for name in LOAD_NAMES]
        # Missing-measurement flags; absent columns default to zero.
        + ["miss_e", "miss_Ip", "miss_K0", "miss_vs", "miss_gran"]
    )

    static_features = np.column_stack(
        [
            soil_df["e"].to_numpy(),
            soil_df["D_r"].to_numpy(),
            soil_df["I_p"].to_numpy(),
            soil_df["V_s"].to_numpy(),
            soil_df["xi"].to_numpy(),
            soil_df["sigma_eff"].to_numpy(),
            np.log10(soil_df["permeability"].to_numpy()),
            soil_df["fines_content"].to_numpy(),
            soil_df["clay_fraction"].to_numpy(),
            np.log10(np.maximum(soil_df["Cu"].to_numpy(), 1.0)),
            soil_df["K0"].to_numpy(),
            soil_df["Vs1"].to_numpy(),
            soil_df["static_shear_ratio"].to_numpy(),
            soil_df["crr_alpha"].to_numpy(),
            soil_df["crr_betta"].to_numpy(),
            load_df["CSR_base"].to_numpy(),
            load_df["frequency"].to_numpy(),
            load_df["amp_scale"].to_numpy(),
            soil_onehot,
            mode_onehot,
            # Missingness flags; absent columns default to zero.
            *[(soil_df[c].to_numpy() if c in soil_df.columns else np.zeros(n, np.float32))
              for c in ("miss_e", "miss_Ip", "miss_K0", "miss_vs", "miss_gran")],
        ]
    ).astype(np.float32)

    prefix_summary_names = [
        "prefix_mean",
        "prefix_last",
        "prefix_peak",
        "prefix_slope",
        "prefix_var",
        "prefix_coverage",
    ]
    prefix_summary = np.column_stack(
        [
            prefix_mean,
            last_prefix.astype(np.float32),
            prefix_peak.astype(np.float32),
            prefix_slope,
            prefix_var,
            prefix_coverage,
        ]
    ).astype(np.float32)

    # Normalize cycle features by the common max_cycle_reference
    # to avoid encoding record duration at every time step.
    _H = float(max_cycle_reference)
    log_cycle_norm = np.log1p(cycles) / np.log1p(_H)
    delta_cycle_norm = delta_cycles / max(_H, 1.0)
    seq_inputs = np.stack(
        [
            csr,
            log_cycle_norm.astype(np.float32),
            delta_cycle_norm.astype(np.float32),
            observations["prefix_obs"],
            observations["prefix_mask"],
        ],
        axis=-1,
    ).astype(np.float32)
    seq_feature_names = ["CSR", "log_cycle_norm", "delta_cycle_norm", "prefix_obs", "prefix_mask"]

    return {
        "static_features": static_features,
        "static_feature_names": static_feature_names,
        "prefix_summary": prefix_summary,
        "prefix_summary_names": prefix_summary_names,
        "seq_inputs": seq_inputs,
        "seq_feature_names": seq_feature_names,
    }
