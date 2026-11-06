"""Prepare laboratory measurements for model inputs.

Accept soil properties, loading, PPR(N), event time and status. Add derived
properties and assemble arrays, metadata and features in the package format.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from dpi_evt.config import ExperimentConfig
from dpi_evt.constants import LOAD_NAMES, SOIL_NAMES
from dpi_evt.data.observed import derive_observed_targets
from dpi_evt.data.splits import make_benchmark_splits
from dpi_evt.data.features import build_feature_matrices
from dpi_evt.physics.crr_physical import RESPONSE_TYPES, compute_crr_components

__all__ = ["compute_crr_features", "enrich_crr_breakdown", "build_observed_prefix",
           "build_population_from_experiments", "ensure_analysis_columns"]


def _col(df: pd.DataFrame, name: str, default) -> np.ndarray:
    """Return a column as an array, or fill it with a default when absent."""
    if name in df.columns:
        return df[name].to_numpy()
    return np.full(len(df), default, dtype=float)


# Reference D50 in mm for missing grain-size curves, by GOST type 1-9;
# e_min/e_max when loose/dense reference states were not measured.
_TYPE_D50 = {1: 3.0, 2: 0.7, 3: 0.4, 4: 0.15, 5: 0.08, 6: 0.03, 7: 0.012, 8: 0.002, 9: 0.05}
_TYPE_EMIN = {1: 0.40, 2: 0.45, 3: 0.50, 4: 0.55, 5: 0.55, 6: 0.50, 7: 0.55, 8: 0.60, 9: 1.50}
_TYPE_EMAX = {1: 0.75, 2: 0.80, 3: 0.85, 4: 0.95, 5: 1.05, 6: 0.95, 7: 1.10, 8: 1.35, 9: 3.00}
_GRAN_COLS = ["gran_10", "gran_5", "gran_2", "gran_1", "gran_05", "gran_025",
              "gran_01", "gran_005", "gran_001", "gran_0002", "gran_0000"]


def ensure_analysis_columns(soil_df: pd.DataFrame, load_df: pd.DataFrame,
                            crr_obs_mask: Optional[np.ndarray] = None) -> None:
    """Add analysis columns to the soil and loading tables in place.

    Compute D10/D50/D60 and PLAXIS classes from the grain-size curve using the
    DigitRock log-interpolation and D50-bin conventions. Other derived columns use
    available properties or GOST-type reference values. Latent truth fields such as
    risk_score_true are not added.

    :param soil_df: soil properties, modified in place
    :param load_df: loading properties, modified in place
    :param crr_obs_mask: measured-CRR availability for has_measured_crr
    :return: None
    """
    from dpi_evt.data.grainsize import FRACTION_KEYS, plaxis_classification

    n = len(soil_df)
    cls = soil_df["class_id"].astype(int).to_numpy() if "class_id" in soil_df.columns else np.zeros(n, int)
    tg = np.clip(cls + 1, 1, 9)

    def setdefault(df: pd.DataFrame, col: str, values) -> None:
        if col not in df.columns:
            df[col] = values

    setdefault(soil_df, "soil_name_en", [SOIL_NAMES[int(i)] for i in cls])

    # Derive D10/D50/D60/Cu/PLAXIS class from the grain-size curve.
    # If unavailable, use the GOST-type D50 estimate with the same class bins.
    gran_cols = [f"gran_{k}" for k in FRACTION_KEYS]
    for col in gran_cols:
        setdefault(soil_df, col, np.zeros(n))
    fractions = soil_df[gran_cols].to_numpy(dtype=float)
    pc = plaxis_classification(fractions)
    has_gran = fractions.sum(axis=1) > 1.0 # Measured grain-size curve is available.
    type_d50 = np.array([_TYPE_D50.get(int(t), 0.01) for t in tg])
    d50 = np.where(has_gran, pc["D50"].astype(float), type_d50)
    d10 = np.where(has_gran, pc["D10"].astype(float), type_d50 / 5.0)
    d60 = np.where(has_gran, pc["D60"].astype(float), type_d50 * 1.5)
    plaxis = np.select([d50 > 10.0, d50 > 2.0, d50 > 0.25, d50 > 0.075],
                       ["very coarse", "coarse", "medium", "fine"], default="very fine")
    setdefault(soil_df, "D10", d10)
    setdefault(soil_df, "D50", d50)
    setdefault(soil_df, "D60", d60)
    setdefault(soil_df, "plaxis_class", plaxis.astype(object))
    cu_prev = _col(soil_df, "Cu", 5.0)
    soil_df["Cu"] = np.where(has_gran, pc["Cu"].astype(float), cu_prev)

    setdefault(soil_df, "e_min", np.array([_TYPE_EMIN.get(int(t), 0.5) for t in tg]))
    setdefault(soil_df, "e_max", np.array([_TYPE_EMAX.get(int(t), 1.0) for t in tg]))
    e_arr = _col(soil_df, "e", 0.7)
    setdefault(soil_df, "n_porosity", e_arr / (1.0 + np.maximum(e_arr, 1e-3)) * 100.0)
    setdefault(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0)
    setdefault(soil_df, "saturation", np.ones(n))
    setdefault(soil_df, "B_value", np.full(n, 0.95))
    setdefault(soil_df, "aging_years", np.ones(n))
    setdefault(soil_df, "cementation_index", np.zeros(n))

    # Optional physical/chemical properties default to NaN.
    for col in ["rs", "rd", "r", "W", "Wl", "Wp", "ground_water_depth",
                "cohesion", "phi", "E_modulus", "calcite", "dolomite", "insoluble_residue"]:
        setdefault(soil_df, col, np.full(n, np.nan))

    # Availability of a measured CRR curve from a resistance-test series.
    has_crr = np.zeros(n, bool) if crr_obs_mask is None else (np.asarray(crr_obs_mask) > 0)
    setdefault(soil_df, "has_measured_crr", has_crr)

    # Loading-shape parameters; laboratory inputs use stationary values here.
    setdefault(load_df, "phase", np.zeros(n))
    setdefault(load_df, "burst_1", np.zeros(n))
    setdefault(load_df, "burst_2", np.zeros(n))


def compute_crr_features(soil_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute CRR alpha and beta from soil properties.

    Use the property-based CRR model under reference loading, filling missing
    properties with the defined defaults.

    :param soil_df: Dr, Ip, fines, clay, effective stress, OCR, K0, Vs1 and other properties
    :return: arrays crr_alpha, crr_betta and crr_ref
    """
    n = len(soil_df)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5),
        Ip=_col(soil_df, "I_p", 0.0),
        Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0),
        fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0),
        sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0),
        K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0),
        aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0),
        B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=np.ones(n),
        loading_mode_factor=np.ones(n),
    )
    return {"crr_alpha": crr["alpha"].astype(np.float32),
            "crr_betta": crr["betta"].astype(np.float32),
            "crr_ref": crr["crr_ref"].astype(np.float32)}


def enrich_crr_breakdown(soil_df: pd.DataFrame, load_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add the property-model CRR decomposition to the soil table.

    Include crr_alpha, crr_betta, crr_ref, crr_cycle_slope, crr_state_proxy,
    response_type and crr_f_* factors. Use load_df.frequency for the frequency factor
    when available; otherwise use the 1 Hz reference. The loading-mode factor stays
    at its reference value 1, while the model receives mode as a category and through
    the dynamic CRR head.

    :param soil_df: soil-property table
    :param load_df: optional loading table with frequency
    :return: the input table with added CRR columns
    """
    n = len(soil_df)
    if load_df is not None and "frequency" in load_df.columns:
        frequency = np.asarray(load_df["frequency"], dtype=np.float64)
    else:
        frequency = np.ones(n)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5), Ip=_col(soil_df, "I_p", 0.0), Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0), fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0), sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0), K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0), aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0), B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=frequency, loading_mode_factor=np.ones(n),
    )
    soil_df["crr_alpha"] = crr["alpha"].astype(np.float32)
    soil_df["crr_betta"] = crr["betta"].astype(np.float32)
    soil_df["crr_ref"] = crr["crr_ref"].astype(np.float32)
    soil_df["crr_cycle_slope"] = crr["cycle_slope"].astype(np.float32)
    soil_df["crr_state_proxy"] = crr["state_proxy"].astype(np.float32)
    soil_df["response_type"] = np.array(RESPONSE_TYPES, dtype=object)[crr["response_idx"]]
    for key, value in crr.items():
        if key.startswith("f_"):
            soil_df[f"crr_{key}"] = value.astype(np.float32)
    return soil_df


def strict_pre_onset_prefix_mask(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    prefix_len: int,
    *,
    strict: bool = True,
    onset_threshold: float = 0.95,
    margin: int = 1,
    min_len: int = 3,
) -> np.ndarray:
    """Construct a prefix mask ending before the first threshold crossing.

    With strict=True, the last index is below onset_idx. margin shortens the window
    subject to min_len, which is also bounded by onset. Without strict truncation,
    use the first prefix_len valid nodes. Strict prefix length is outcome-dependent.

    :param r_obs: measured PPR, shape (n, seq_len)
    :param valid_mask: measurement mask, shape (n, seq_len)
    :param prefix_len: maximum prefix length
    :param strict: whether to truncate before onset
    :param onset_threshold: ru onset threshold
    :param margin: additional pre-onset gap in grid nodes
    :param min_len: desired minimum, subject to ending before onset
    :return: float32 binary mask, shape (n, seq_len)
    """
    n, seq_len = r_obs.shape
    idx = np.arange(seq_len)[None, :]
    if not strict:
        return ((idx < prefix_len) & (valid_mask > 0)).astype(np.float32)
    onset_hit = (r_obs >= onset_threshold) & (valid_mask > 0)
    has_onset = onset_hit.any(axis=1)
    onset_idx = np.where(has_onset, onset_hit.argmax(axis=1), seq_len).astype(int)
    cut = onset_idx - int(margin) # Last permitted index is cut-1.
    floor = np.minimum(int(min_len), np.where(has_onset, onset_idx, prefix_len))
    cut = np.maximum(cut, floor) # Apply min_len without passing onset.
    cut = np.clip(cut, 0, prefix_len) # Bound by prefix_len.
    mask = (idx < cut[:, None]) & (valid_mask > 0)
    return mask.astype(np.float32)


def landmark_prefix_mask(cycles: np.ndarray, valid_mask: np.ndarray, landmark_cycles: float,
                         prefix_len: Optional[int] = None) -> np.ndarray:
    """Mask observations through landmark_cycles, inclusive.

    The physical-cycle window contains at most prefix_len grid nodes. The same mask
    defines prefix_obs, sequence inputs and prefix_summary, so coverage is <=1.

    :param cycles: specimen cycle grids, shape (n, seq_len)
    :param valid_mask: measurement mask, shape (n, seq_len)
    :param landmark_cycles: physical landmark N0
    :param prefix_len: optional upper bound on the node count
    :return: float32 binary mask, shape (n, seq_len)
    """
    m = (np.asarray(cycles) <= float(landmark_cycles)) & (valid_mask > 0)
    if prefix_len is not None:
        step_idx = np.arange(m.shape[1])[None, :]
        m = m & (step_idx < int(prefix_len)) # At most prefix_len grid nodes.
    return m.astype(np.float32)


def build_observed_prefix(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    prefix_len: int,
    *,
    strict_preonset: bool = True,
    onset_threshold: float = 0.95,
    margin: int = 1,
    min_len: int = 3,
    landmark_cycles: Optional[float] = None,
    cycles: Optional[np.ndarray] = None,
):
    """Construct a prefix from measured PPR without adding noise.

    The default strict_preonset mode truncates before the threshold crossing.

    :param r_obs: measured PPR, shape (n, seq_len)
    :param valid_mask: measurement mask, shape (n, seq_len)
    :param prefix_len: prefix length
    :param strict_preonset: truncate before onset
    :param onset_threshold: ru onset threshold
    :param margin: pre-onset gap in grid nodes
    :param min_len: minimum length subject to onset truncation
    :return: prefix_obs and prefix_mask arrays
    """
    if landmark_cycles is not None and cycles is not None:
        # Landmark window: physical cycles <=N0, capped by prefix_len,
        # shared across model branches with prefix_coverage<=1.
        prefix_mask = landmark_prefix_mask(cycles, valid_mask, landmark_cycles, prefix_len=prefix_len)
    else:
        prefix_mask = strict_pre_onset_prefix_mask(
            r_obs, valid_mask, prefix_len, strict=strict_preonset,
            onset_threshold=onset_threshold, margin=margin, min_len=min_len,
        )
    prefix_obs = (r_obs * prefix_mask).astype(np.float32)
    return {"prefix_obs": prefix_obs, "prefix_mask": prefix_mask}


def build_population_from_experiments(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    csr: np.ndarray,
    r_measured: np.ndarray,
    valid_mask: np.ndarray,
    liq_label: np.ndarray,
    n_liq: np.ndarray,
    config: ExperimentConfig,
    prefix_source: Optional[np.ndarray] = None,
    crr_obs: Optional[np.ndarray] = None,
    crr_obs_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Assemble a prepared dataset from observed laboratory measurements.

    The result supports data.io.save_population_artifact and
    data.splits.prepare_benchmark_dataset. Soil inputs include class_id/type_ground,
    soil_type and the physical properties used by the features. Compute missing
    crr_alpha/crr_betta from those properties. Loading inputs include CSR_base,
    frequency, amp_scale, N_max, nonstationarity, load_mode and mode_id.
    N_max and nonstationarity remain diagnostic metadata, outside the static input.

    :param soil_df: specimen soil-property table
    :param load_df: specimen loading-property table
    :param cycles: cycle grids, shape (n, seq_len)
    :param csr: CSR at each query node, shape (n, seq_len)
    :param r_measured: measured target PPR, shape (n, seq_len)
    :param valid_mask: measurement mask, shape (n, seq_len)
    :param liq_label: binary event labels, shape (n,)
    :param n_liq: observed event or censoring times, shape (n,)
    :param config: lengths, normalization and subset settings
    :param prefix_source: optional causal prefix curve; landmark inputs must use only
        observations through N0, while the target may use full-record smoothing
    :param crr_obs: optional measured CRR curves, shape (n, seq_len)
    :param crr_obs_mask: specimen-level CRR availability, shape (n,)
    :return: dataset dictionary of observed and derived fields
    """
    soil_df = soil_df.reset_index(drop=True).copy()
    load_df = load_df.reset_index(drop=True).copy()
    n, seq_len = r_measured.shape

    # Derived classification columns.
    if "class_id" not in soil_df.columns and "type_ground" in soil_df.columns:
        soil_df["class_id"] = soil_df["type_ground"].astype(int) - 1
    if "soil_type" not in soil_df.columns:
        soil_df["soil_type"] = [SOIL_NAMES[int(i)] for i in soil_df["class_id"]]
    if "mode_id" not in load_df.columns and "load_mode" in load_df.columns:
        load_df["mode_id"] = load_df["load_mode"].map({name: i for i, name in enumerate(LOAD_NAMES)})

    # Property-model CRR decomposition for features and diagnostics.
    if "crr_alpha" not in soil_df.columns or "crr_betta" not in soil_df.columns:
        soil_df = enrich_crr_breakdown(soil_df, load_df)
    if "Cu" not in soil_df.columns:
        soil_df["Cu"] = 5.0
    if "Vs1" not in soil_df.columns:
        soil_df["Vs1"] = _col(soil_df, "V_s", 180.0)

    # Columns used by the common analysis/evaluation interface.
    ensure_analysis_columns(soil_df, load_df, crr_obs_mask)

    delta_cycles = np.diff(np.concatenate([np.zeros((n, 1)), cycles], axis=1), axis=1).astype(np.float32)
    # landmark: physical-cycle window through N0 for all records;
    # fixed_k: first K grid nodes; preonset: truncation before onset,
    # with outcome-dependent length only in preonset mode.
    _mode = getattr(config, "prefix_mode", "preonset")
    _fixed_k = _mode == "fixed_k"
    _landmark = _mode == "landmark"
    _pref_window = int(getattr(config, "prefix_fixed_k", 6)) if _fixed_k else config.prefix_len
    prefix_values = r_measured if prefix_source is None else np.asarray(prefix_source)
    if prefix_values.shape != r_measured.shape:
        raise ValueError("prefix_source must have the same shape as r_measured")
    observations = build_observed_prefix(
        prefix_values.astype(np.float32), valid_mask.astype(np.float32), _pref_window,
        strict_preonset=(False if (_fixed_k or _landmark) else getattr(config, "prefix_strict_preonset", True)),
        onset_threshold=getattr(config, "prefix_onset_threshold", config.liq_threshold),
        margin=getattr(config, "prefix_onset_margin", 1),
        min_len=getattr(config, "prefix_min_len", 3),
        landmark_cycles=(float(getattr(config, "prefix_landmark_cycles", 20.0)) if _landmark else None),
        cycles=(cycles.astype(np.float32) if _landmark else None),
    )
    features = build_feature_matrices(soil_df, load_df, cycles.astype(np.float32), delta_cycles,
                                      csr.astype(np.float32), observations, config.prefix_len,
                                      max_cycle_reference=float(config.max_cycle_reference))

    meta = pd.concat([soil_df, load_df], axis=1)
    meta["liq_label"] = np.asarray(liq_label).astype(int)
    meta["N_liq_true"] = np.asarray(n_liq).astype(np.float32)
    meta["PPR_max_true"] = r_measured.max(axis=1)
    meta["CSR_max"] = csr.max(axis=1)
    # Full CSR(N) is a prescribed operator-controlled loading program.
    # Record that exogenous-input contract in the artifact.
    # Exclude post-hoc CSR_max/nonstationarity summaries from model features.
    meta["csr_program_known_a_priori"] = True

    # Auxiliary targets derived from measured PPR.
    obs_targets = derive_observed_targets(r_measured.astype(np.float32), valid_mask.astype(np.float32))

    benchmark = make_benchmark_splits(meta, min(config.benchmark_subset, n), config.seed, config)

    population = {
        "meta": meta,
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles,
        "csr": csr.astype(np.float32),
        "r_obs": r_measured.astype(np.float32),
        "valid_mask": valid_mask.astype(np.float32),
        "prefix_mask": observations["prefix_mask"],
        "prefix_obs": observations["prefix_obs"],
        "liq_label": np.asarray(liq_label).astype(np.float32),
        "n_liq_true": np.asarray(n_liq).astype(np.float32),
        "g_obs": obs_targets["g_obs"],
        "risk_proxy": obs_targets["risk_proxy"],
        "static_features": features["static_features"],
        "static_feature_names": features["static_feature_names"],
        "prefix_summary": features["prefix_summary"],
        "prefix_summary_names": features["prefix_summary_names"],
        "seq_inputs": features["seq_inputs"],
        "seq_feature_names": features["seq_feature_names"],
        "benchmark": benchmark,
    }
    # Optional measured CRR(N), for example from a six-specimen series.
    if crr_obs is not None:
        population["crr_obs"] = np.asarray(crr_obs, dtype=np.float32)
        population["crr_obs_mask"] = (np.ones(n, dtype=np.float32) if crr_obs_mask is None
                                      else np.asarray(crr_obs_mask, dtype=np.float32))
    return population
