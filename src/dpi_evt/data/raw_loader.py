"""Read laboratory archives: DigitRock pickles and .xls property statements.

Extract pressure/loading histories and specimen properties, read statements,
and fit group-level CRR(N). build_real_objects_population assembles the dataset
through build_population_from_experiments. data.ppr_envelope constructs envelopes;
data.real_adapter adds derived properties and features.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dpi_evt.config import ExperimentConfig, LIQ_THRESHOLD
from dpi_evt.data.ppr_envelope import (causal_monotone_smooth, extract_cycle_amplitude,
                                               extract_upper_envelope,
                                               forecasting_cycles, monotone_smooth,
                                               smooth_ppr_trajectory)
from dpi_evt.data.real_adapter import build_population_from_experiments


def _cycle_bins(cycles) -> np.ndarray:
    """Assign an integer cycle to each peak, allowing for rounding error.

    Adjacent peaks can occur at k+0.9 and (k+1)+0.1 as within-cycle phase changes.
    Use floor(cyc) for adjacency; epsilon handles values such as 1.9999999 for 2.0.
    """
    return np.floor(np.asarray(cycles, dtype=float) + 1e-6).astype(np.int64)


def _consecutive_cycles(cycles) -> bool:
    """Check that adjacent peaks belong to consecutive integer cycles, with step 1."""
    if cycles is None:
        return True
    bins = _cycle_bins(cycles)
    return bins.size <= 1 or bool(np.all(np.diff(bins) == 1))


def sustained_first_crossing(curve, thr: float = LIQ_THRESHOLD, sustain: int = 3,
                             cycles=None) -> int:
    """Find the start of a full sustained-threshold window on raw cycle peaks.

    All sustain values must be >= thr. Supplied cycle numbers must be consecutive
    after integer binning; see _cycle_bins. An incomplete terminal window does not
    confirm onset and is marked by terminal_onset_ambiguous. Targets use the complete
    record; prefixes use measurements available through the landmark.

    :param curve: raw per-cycle pressure peaks, one-dimensional
    :param thr: pressure threshold, usually LIQ_THRESHOLD
    :param sustain: required consecutive cycle count above threshold
    :param cycles: physical cycle coordinates, if available
    :return: first sustained-crossing index, or -1 if absent
    """
    c = np.asarray(curve)
    cyc = np.arange(c.size, dtype=float) if cycles is None else np.asarray(cycles, dtype=float)
    if cyc.shape != c.shape:
        raise ValueError("cycles and curve must have the same shape")
    above = c >= thr
    n = c.size
    s = int(max(1, sustain))
    for i in range(n):
        if not above[i]:
            continue
        j = i + s
        if j <= n and bool(above[i:j].all()):
            if cycles is None or _consecutive_cycles(cyc[i:j]):
                return i
    return -1


def terminal_onset_ambiguous(curve, thr: float = LIQ_THRESHOLD, sustain: int = 3,
                             cycles=None, min_tail: int = 2) -> bool:
    """Check for a terminal threshold exceedance shorter than the sustained window."""
    c = np.asarray(curve)
    cyc = np.arange(c.size, dtype=float) if cycles is None else np.asarray(cycles, dtype=float)
    if c.shape != cyc.shape or c.size == 0:
        return False
    above = c >= thr
    start = c.size
    while start > 0 and above[start - 1]:
        start -= 1
    tail = c.size - start
    s = max(int(sustain), 1)
    if not (max(int(min_tail), 1) <= tail < s):
        return False
    if cycles is not None and tail > 1 and not _consecutive_cycles(cyc[start:]):
        return False
    return True


def _peak_on_grid(cyc, signal, grid, points_in_cycle=None) -> np.ndarray:
    """Interpolate per-cycle peaks of a signal to grid.

    Used for deviator q(N) and strain epsilon(N). Values retain their stress/strain
    amplitude and are not clipped to the PPR interval [0, 1].
    """
    cp, pk = extract_upper_envelope(cyc, signal, points_in_cycle)
    if cp.size < 2:
        return np.zeros(len(grid), np.float32)
    return np.interp(np.asarray(grid, float), cp, pk).astype(np.float32)

__all__ = [
    "RealUnpickler", "load_pickle", "gv",
    "fines_clay_cu", "dr_proxy", "extract_test",
    "find_object_pickles", "load_object", "discover_objects",
    "read_statement", "fit_alpha_betta", "build_crr_obs",
    "find_cloud_root", "DEFAULT_TEST_TYPES", "TYPE_TO_MODE", "sustained_first_crossing",
    "terminal_onset_ambiguous",
    "build_real_objects_population", "build_cohort_manifest", "canonical_site_id",
]


def build_cohort_manifest(population: Dict[str, object], raw_count: Optional[int] = None) -> Dict[str, object]:
    """Summarize the extracted records and the eligible landmark risk set.

    :param population: dataset from build_real_objects_population
    :param raw_count: extracted record count; defaults to cohort_filter_counts
    :return: raw/eligible counts, classes, objects, loading modes and CRR coverage
    """
    meta = population["meta"]; lab = np.asarray(population["liq_label"])
    filt = dict(population.get("cohort_filter_counts", {}))
    raw_n = int(raw_count if raw_count is not None else filt.get("raw_specimens", len(meta)))
    excluded_event = int(filt.get("excluded_event_before_N0", 0))
    excluded_censored = int(filt.get("excluded_censored_before_N0", 0))
    crrm = population.get("crr_obs_mask")
    crr_n = int((np.asarray(crrm) > 0.5).sum()) if crrm is not None else 0
    crr_obj = int(meta.loc[np.asarray(crrm) > 0.5, "object"].nunique()) if crrm is not None else 0
    crr_site = int(meta.loc[np.asarray(crrm) > 0.5, "site_id"].nunique())\
        if crrm is not None and "site_id" in meta else crr_obj
    events = np.asarray(population.get("n_liq_true", []), dtype=float)[lab > 0.5]
    return {
        "raw_specimens": raw_n,
        "analytic_risk_set": int(len(meta)),
        "excluded_before_N0_total": int(raw_n - len(meta)),
        "excluded_event_before_N0": excluded_event,
        "excluded_censored_before_N0": excluded_censored,
        "n_liq": int((lab > 0.5).sum()), "n_nonliq": int((lab < 0.5).sum()),
        "n_objects": int(meta["object"].nunique()),
        "n_sites": int(meta["site_id"].nunique()) if "site_id" in meta else int(meta["object"].nunique()),
        "raw_objects": int(filt.get("raw_objects", meta["object"].nunique())),
        "modes": {str(k): int(v) for k, v in meta["load_mode"].value_counts().items()},
        "crr_samples": crr_n, "crr_objects": crr_obj, "crr_sites": crr_site,
        "terminal_onset_ambiguous_raw": int(filt.get("terminal_onset_ambiguous", 0)),
        "terminal_onset_ambiguous_analytic": int(meta.get(
            "onset_terminal_ambiguous", pd.Series(np.zeros(len(meta), dtype=int))).sum()),
        "audit_prefix_crossed_but_target_after_N0": int(
            filt.get("audit_prefix_crossed_but_target_after_N0", 0)),
        "audit_target_by_N0_but_prefix_not_crossed": int(
            filt.get("audit_target_by_N0_but_prefix_not_crossed", 0)),
        "event_nliq_quantiles": ({str(q): float(np.quantile(events, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
                                 if events.size else {}),
        "site_object_map": ({str(site): sorted(map(str, grp["object"].unique()))
                             for site, grp in meta.groupby("site_id")}
                            if "site_id" in meta else {}),
    }

# DigitRock grain-size fractions and characteristic sizes in mm.
GRAN = ["10", "5", "2", "1", "05", "025", "01", "005", "001", "0002", "0000"]
GRAN_SIZE = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.002, 0.001]
GRAN_COLS = ["gran_10", "gran_5", "gran_2", "gran_1", "gran_05", "gran_025",
             "gran_01", "gran_005", "gran_001", "gran_0002", "gran_0000"]
# Order-of-magnitude permeability by GOST type 1-9, in m/s.
PERM_BY_TYPE = {1: 1e-4, 2: 5e-5, 3: 1e-5, 4: 5e-6, 5: 1e-6, 6: 1e-7, 7: 1e-8, 8: 1e-9, 9: 1e-7}
FINES_BY_TYPE = {1: 3, 2: 4, 3: 5, 4: 8, 5: 20, 6: 45, 7: 75, 8: 90, 9: 60}
CLAY_BY_TYPE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 8, 7: 20, 8: 40, 9: 15}
# Loading-type folder name to model loading category.
TYPE_TO_MODE = {"Потенциал разжижения": "seismic", "Сейсморазжижение": "seismic",
                "Сейсмо": "seismic", "Штормовое разжижение": "storm"}
DEFAULT_TEST_TYPES = ["Потенциал разжижения", "Штормовое разжижение", "Сейсмо"]


# Pickle loading.

class _Stub:
    """Placeholder for an unavailable DigitRock class; accepts state in __dict__."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class RealUnpickler(pickle.Unpickler):
    """Read DigitRock pickles without importing the DigitRock package."""

    def find_class(self, module, name):
        if module.startswith("numpy") or module in ("builtins", "collections", "_codecs", "copy_reg"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def load_pickle(path: str):
    """Read a data/handler pickle without DigitRock dependencies."""
    with open(path, "rb") as f:
        return RealUnpickler(f).load()


def gv(obj, name, default=None):
    """Read an attribute, unwrapping a result object through _value/value when present."""
    d = getattr(obj, "__dict__", {})
    if name not in d:
        return default
    v = d[name]
    if hasattr(v, "__dict__"):
        vd = v.__dict__
        if "_value" in vd:
            return vd["_value"]
        if "value" in vd:
            return vd["value"]
    return v


# Soil properties.

def fines_clay_cu(phys, type_ground: int, Ip) -> Tuple[float, float, float]:
    """Return fines_content (%), clay_fraction (%) and Cu from grain size or a GOST-type estimate."""
    gd = {k: gv(phys, f"granulometric_{k}", None) for k in GRAN}
    if any(v not in (None, "") for v in gd.values()):
        v = {k: (float(gd[k]) if gd[k] not in (None, "") else 0.0) for k in GRAN}
        fines = v["005"] + v["001"] + v["0002"] + v["0000"] # <= 0.05 mm.
        clay = v["0000"] # < 0.002 mm.
        frac_pass = np.clip(np.cumsum([v[k] for k in GRAN][::-1])[::-1], 0, 100)
        d = np.array(GRAN_SIZE, float); order = np.argsort(frac_pass)
        try:
            D10 = np.interp(10, frac_pass[order], d[order])
            D60 = np.interp(60, frac_pass[order], d[order])
            Cu = float(np.clip(D60 / max(D10, 1e-4), 1.0, 200.0))
        except Exception:
            Cu = 5.0
        return float(fines), float(clay), Cu
    clay = float(np.clip(Ip * 1.4, 2, 60)) if Ip else CLAY_BY_TYPE.get(int(type_ground), 5)
    return float(FINES_BY_TYPE.get(int(type_ground), 30)), float(clay), 5.0


def dr_proxy(e) -> float:
    """Estimate a relative-density proxy from void ratio, from dense e=0.40 to loose e=1.10."""
    if e is None:
        return 0.5
    return float(np.clip((1.10 - float(e)) / (1.10 - 0.40), 0.0, 1.0))


# Extract one specimen.

def extract_test(data_obj, handler_obj, test_type: str, seq_len: int,
                 landmark_n0: Optional[float] = None, landmark_k: Optional[int] = None,
                 horizon_default: Optional[float] = None, onset_sustain_cycles: int = 3,
                 query_prefix_horizon: Optional[float] = None):
    """Extract specimen properties, loading and PPR(N) arrays.

    :param data_obj: data object containing cycles/PPR
    :param handler_obj: handler containing _test_params/_test_result
    :param test_type: loading-type folder name
    :param seq_len: target PPR grid length
    :return: (soil, load, arrays, liq_label, n_liq), or None if data are unavailable
    """
    cyc = gv(data_obj, "cycles"); ppr = gv(data_obj, "PPR")
    if cyc is None or ppr is None or len(cyc) < 3:
        return None
    tp = gv(handler_obj, "_test_params")
    phys = gv(tp, "physical") if tp is not None else None
    tr = gv(handler_obj, "_test_result")
    if tp is None or phys is None:
        return None

    # Smoothed upper envelope of the cyclic PPR signal.
    pic = gv(tp, "points_in_cycle", None)
    coarse_grid, coarse_r, coarse_mask, _cp, ppr_peaks = smooth_ppr_trajectory(
        cyc, ppr, seq_len, points_in_cycle=pic, return_peaks=True
    )
    # Interpolate the landmark grid from the full-resolution cycle envelope.
    # Resampling a coarse grid again loses early measurements and can mix
    # post-landmark values into the observed prefix.
    ppr_sm_pc = monotone_smooth(ppr_peaks) if len(ppr_peaks) else np.zeros(0, float)
    ppr_causal_pc = causal_monotone_smooth(ppr_peaks) if len(ppr_peaks) else np.zeros(0, float)
    peak = float(ppr_peaks.max()) if len(ppr_peaks) else float(np.nanmax(np.nan_to_num(ppr)))

    # Loading metadata, query horizon and observed duration.
    sigma_1 = gv(tp, "sigma_1", 100.0) or 100.0
    t = gv(tp, "t", None)
    csr = float(t) / float(sigma_1) if t else 0.2
    n_total = max(int(np.floor(np.nanmax(cyc))), 1)
    # last_obs is the actual final observed cycle and the censoring time.
    # It is excluded from the query grid and sequence inputs because termination
    # at onset would disclose the target through record duration.
    last_obs = float(_cp[-1]) if len(_cp) else float(
        coarse_grid[coarse_mask > 0].max() if np.any(coarse_mask > 0) else coarse_grid[-1]
    )
    # Preserve cycles_count as duration metadata.
    # Define the input-grid endpoint separately through grid_horizon.
    cycles_count = gv(tp, "cycles_count", None)
    n_max_planned = float(cycles_count) if (cycles_count and float(cycles_count) > 0) else None
    n_max = n_max_planned if n_max_planned is not None else float(n_total)
    # End the query grid at the common horizon_default.
    # Individual record duration does not set the prediction horizon.
    grid_horizon = float(horizon_default) if (horizon_default and horizon_default > 0) else float(n_total)

    # The target is the first sustained crossing of ru>=LIQ_THRESHOLD
    # on raw cycle peaks, requiring a full sustain-cycle window.
    # Do not use the isotonic envelope for this rule: after crossing,
    # it retains the threshold level. An incomplete terminal window is marked
    # onset_terminal_ambiguous. This is a retrospective target; build the prefix
    # separately from measurements available through N0. fail_cycle is unused.
    def _sustained_first_crossing(curve, cycle_values):
        return sustained_first_crossing(curve, thr=LIQ_THRESHOLD,
                                        sustain=int(max(1, onset_sustain_cycles)), cycles=cycle_values)
    _cross_i = _sustained_first_crossing(ppr_peaks, _cp) if len(ppr_peaks) else -1
    onset_terminal_ambiguous = int(terminal_onset_ambiguous(
        ppr_peaks, thr=LIQ_THRESHOLD, sustain=int(max(1, onset_sustain_cycles)), cycles=_cp,
    )) if len(ppr_peaks) else 0
    if _cross_i >= 0 and len(_cp):
        liq = 1
        n_liq = float(_cp[int(_cross_i)]) # Sustained-crossing cycle at the original cycle resolution.
        # Keep the target envelope compatible with the confirmed onset rule.
        # Isotonic smoothing can average a short confirmed above-threshold block
        # below threshold. Adjust the retrospective target after confirmed onset;
        # the causal envelope and input prefix remain unchanged.
        ppr_sm_pc[int(_cross_i):] = np.maximum(ppr_sm_pc[int(_cross_i):], LIQ_THRESHOLD)
        ppr_sm_pc = np.maximum.accumulate(ppr_sm_pc)
    else:
        # Without a crossing, right-censor N_liq at the final observed cycle.
        # Observed duration supplies a lower bound on event time;
        # it is distinct from the planned/query horizon.
        liq = 0
        n_liq = float(last_obs)

    # Common early grid: first k nodes = geomspace(1, N0, k),
    # ending at the shared max_cycle_reference horizon for every specimen.
    # The mask marks target availability through last_obs;
    # interpolation beyond last_obs is constant and those values are masked.
    event_in_prefix = 0 # Causal flag: an event is already observable by N0.
    if landmark_n0 is not None:
        grid = forecasting_cycles(
            grid_horizon, seq_len, float(landmark_n0), int(landmark_k or 12), query_prefix_horizon
        ).astype(np.float32)
        if len(_cp):
            r = np.interp(grid, _cp, ppr_sm_pc).astype(np.float32)
            r_causal = np.interp(grid, _cp, ppr_causal_pc).astype(np.float32)
        else:
            r = np.zeros(seq_len, np.float32)
            r_causal = np.zeros(seq_len, np.float32)
        mask = (grid <= last_obs + 1e-6).astype(np.float32)

        # Smooth the prefix using only observations through N0.
        # Retain the measured prefix without outcome-conditioned clipping.
        # Neither future event time nor the retrospective label is used
        # to alter its values. An observed crossing at the landmark
        # is recorded in event_in_prefix and handled by risk-set eligibility
        # during cohort assembly.
        early = np.asarray(_cp) <= float(landmark_n0) + 1e-9
        if int(early.sum()) > 0:
            cp_early = np.asarray(_cp)[early]
            peaks_early = np.asarray(ppr_peaks)[early]
            ppr_early = monotone_smooth(peaks_early)
            prefix_r = np.interp(grid, cp_early, ppr_early).astype(np.float32)
            # Compute event_in_prefix from sustained crossings on raw peaks
            # within the observed window through N0, independently of the label.
            # A specimen with an already observed event is excluded from forecasting.
            event_in_prefix = int(_sustained_first_crossing(peaks_early, cp_early) >= 0)
        else:
            prefix_r = np.zeros(seq_len, np.float32)
            event_in_prefix = 0
    else:
        grid = coarse_grid.astype(np.float32)
        r = coarse_r.astype(np.float32)
        mask = coarse_mask.astype(np.float32)
        r_causal = (np.interp(grid, _cp, ppr_causal_pc).astype(np.float32)
                    if len(_cp) else np.zeros(seq_len, np.float32))
        # fixed_k/preonset also use a one-sided causal input envelope.
        # Keep it distinct from the retrospective target envelope, whose smoothing
        # can depend on later peaks.
        prefix_r = r_causal.copy()

    # Interpolate q(N) and strain directly from the per-cycle peaks,
    # preserving early resolution without a second resampling step.
    dev = gv(data_obj, "deviator"); eps_sig = gv(data_obj, "strain")
    q_grid = _peak_on_grid(cyc, dev, grid, pic) if dev is not None else np.zeros(seq_len, np.float32)
    eps_grid = _peak_on_grid(cyc, eps_sig, grid, pic) if eps_sig is not None else np.zeros(seq_len, np.float32)

    # Only the prescribed amplitude is available for forecasting. The measured deviator channel
    # contains the test stop and response; it must not reconstruct the future control program.
    csr_series = np.full(seq_len, csr, np.float32)
    meas_nonstat = 0.05
    if dev is not None:
        _ca, _amp = extract_cycle_amplitude(cyc, dev, pic)
        _amp = np.asarray(_amp, float)
        if _amp.size >= 3 and np.nanmax(_amp) > 0:
            meas_nonstat = float(np.clip(np.nanstd(_amp) / max(abs(np.nanmean(_amp)), 1e-6), 0.0, 1.0))

    # Soil properties.
    tg = int(gv(phys, "type_ground", 7) or 7)
    Ip = gv(phys, "Ip", None); e = gv(phys, "e", None)
    rho = gv(phys, "r", 2.0) or 2.0
    # DigitRock convention: Vs = sqrt(G/rho), G = E/(2*(1+nu)). Priority:
    # 1. Measured shear-wave velocity, converted by a factor of 31.6228;
    # 2. Dynamic shear modulus result.G in MPa;
    # 3. Static E0/E from _test_params, converted from kPa to MPa,
    #    then G = E/(2*(1+nu)).
    # Use G=25 MPa only if all three sources are missing.
    def _modulus_mpa(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if v <= 0:
            return None
        return v / 1000.0 if v > 200.0 else v # Values >200 are treated as kPa and converted to MPa.
    nu = gv(tp, "unload_poisons_ratio", 0.2) or 0.2
    tw = gv(tr, "transverse_waves_velocity", None)
    _G_dyn = gv(tr, "G", None)
    _E = _modulus_mpa(gv(tr, "E0", None)) or _modulus_mpa(gv(tp, "E0", None)) or _modulus_mpa(gv(tp, "E", None))
    # Vs has a numeric value after imputation, but its measurement source
    # is retained through miss_vs, including static-E and constant-G estimates.
    _vs_direct = 0.0
    if tw and float(tw) > 0:
        Vs = float(tw) * 31.6228
        _vs_direct = 1.0
    elif _G_dyn and float(_G_dyn) > 0:
        Vs = float((float(_G_dyn) * 1e6 / (float(rho) * 1e3)) ** 0.5)
        _vs_direct = 1.0
    elif _E is not None:                                  # Static E0/E -> G=E/(2*(1+nu)) -> Vs.
        G_mpa = float(_E) / (2 * (1 + float(nu)))
        Vs = float((G_mpa * 1e6 / (float(rho) * 1e3)) ** 0.5)
    else:
        Vs = float((25.0 * 1e6 / (float(rho) * 1e3)) ** 0.5)
    Vs = float(np.clip(Vs, 40, 600))
    sigma_eff = float(sigma_1)
    fines, clay, Cu = fines_clay_cu(phys, tg, Ip)
    # K0 priority: measured value, Jaky relation 1-sin(phi), then 0.5.
    # Retain the source flag to distinguish an estimate from a measurement.
    _k0 = gv(tp, "K0", None); _fi = gv(tp, "fi", None)
    _k0_direct = 0.0
    if _k0 not in (None, "") and float(_k0) > 0:
        K0_val = float(_k0)
        _k0_direct = 1.0
    elif _fi not in (None, "") and float(_fi) > 0:
        K0_val = float(round(1.0 - np.sin(np.pi * float(_fi) / 180.0), 3))
    else:
        K0_val = 0.5
    K0_val = float(np.clip(K0_val, 0.2, 1.0))

    soil = dict(
        laboratory_number=gv(phys, "laboratory_number", ""), borehole=gv(phys, "borehole", ""),
        depth=gv(phys, "depth", None), soil_name=gv(phys, "soil_name", ""),
        type_ground=tg, class_id=tg - 1,
        e=float(e) if e is not None else 0.7, D_r=dr_proxy(e),
        I_p=float(Ip) if Ip is not None else 0.0,
        Il=float(gv(phys, "Il", 0.0) or 0.0), Ir=float(gv(phys, "Ir", 0.0) or 0.0),
        V_s=Vs, Vs1=float(np.clip(Vs * (100.0 / max(sigma_eff, 1.0)) ** 0.25, 60, 450)),
        xi=float((gv(tp, "damping_ratio", 1.5) or 1.5) / 100.0),
        sigma_eff=sigma_eff, permeability=PERM_BY_TYPE.get(tg, 1e-7),
        fines_content=fines, clay_fraction=clay, Cu=Cu,
        K0=K0_val, static_shear_ratio=0.0,
        ige=str(gv(phys, "ige", "") or ""), # Engineering-geological unit used to group CRR series.
    )

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    soil.update(dict(
        rs=_f(gv(phys, "rs", None)), rd=_f(gv(phys, "rd", None)), r=_f(rho),
        n_porosity=_f(gv(phys, "n", None)), W=_f(gv(phys, "W", None)),
        Wl=_f(gv(phys, "Wl", None)), Wp=_f(gv(phys, "Wp", None)),
        saturation=_f(gv(phys, "Sr", None)), ground_water_depth=_f(gv(phys, "ground_water_depth", None)),
        B_value=_f(gv(phys, "skempton_initial", None)), cohesion=_f(gv(tp, "c", None)),
        phi=_f(gv(tp, "fi", None)), E_modulus=_f(gv(tp, "E", None)),
        damping_ratio=_f(gv(tp, "damping_ratio", None)),
        calcite=_f(gv(phys, "calcite", None)), dolomite=_f(gv(phys, "dolomite", None)),
    ))
    for col, key in zip(GRAN_COLS, GRAN):
        g = gv(phys, f"granulometric_{key}", None)
        soil[col] = 0.0 if g in (None, "") else float(g)

    # A value of 1 means no direct measurement: formula or constant imputation.
    # Preserve source flags alongside the completed numeric properties.
    _gran_present = any(gv(phys, f"granulometric_{k}", None) not in (None, "") for k in GRAN)
    soil.update(dict(
        miss_e=0.0 if e is not None else 1.0,
        miss_Ip=0.0 if Ip is not None else 1.0,
        miss_K0=float(1.0 - _k0_direct),
        miss_vs=float(1.0 - _vs_direct),
        miss_gran=0.0 if _gran_present else 1.0,
    ))

    # event_in_prefix=1 indicates a sustained raw-peak crossing
    # already observed within the causal window through N0.
    # Preserve the flag in metadata, independently of the retrospective label.
    # Cohort assembly also checks retrospective onset<=N0;
    # discrepancies are counted without rewriting input observations.
    load = dict(
        CSR_base=csr, frequency=float(gv(tp, "frequency", 0.5) or 0.5),
        amp_scale=1.0, N_max=n_max, N_max_is_planned=(n_max_planned is not None),
        nonstationarity=meas_nonstat, # Post-hoc diagnostic; excluded from model inputs.
        load_mode=TYPE_TO_MODE.get(test_type, "seismic"),
        event_in_prefix=event_in_prefix, # Onset within the N0 window; exclude from the risk set.
        onset_terminal_ambiguous=onset_terminal_ambiguous,
    )
    arrays = dict(
        cycles=grid.astype(np.float32), csr=csr_series, # Prescribed CSR(N) loading program.
        r=r.astype(np.float32), mask=mask.astype(np.float32),
        prefix_r=prefix_r.astype(np.float32),
        r_causal=r_causal.astype(np.float32),
        q=q_grid, eps=eps_grid, # Deviator and strain on the cycle grid for optional diagnostics.
    )
    return soil, load, arrays, int(liq), float(n_liq)


# Discover object directories.

def find_object_pickles(obj_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Find data/handler CyclicModel pickles inside an object directory."""
    dp = glob.glob(os.path.join(obj_dir, "**", "data CyclicModel*"), recursive=True)
    hp = glob.glob(os.path.join(obj_dir, "**", "handler CyclicModel*"), recursive=True)
    return (dp[0], hp[0]) if dp and hp else (None, None)


def load_object(obj_dir: str, test_type: str, seq_len: int,
                landmark_n0: Optional[float] = None, landmark_k: Optional[int] = None,
                horizon_default: Optional[float] = None,
                onset_sustain_cycles: int = 3,
                query_prefix_horizon: Optional[float] = None) -> List[Tuple[str, tuple]]:
    """Extract specimens from one object as (key, extract_test_record) pairs."""
    dpath, hpath = find_object_pickles(obj_dir)
    if not dpath:
        return []
    D = load_pickle(dpath)["data"]
    H = load_pickle(hpath)["data"]
    out = []
    for key in D:
        if key not in H:
            continue
        rec = extract_test(D[key], H[key], test_type, seq_len, landmark_n0=landmark_n0,
                           landmark_k=landmark_k, horizon_default=horizon_default,
                           onset_sustain_cycles=onset_sustain_cycles,
                           query_prefix_horizon=query_prefix_horizon)
        if rec is not None:
            out.append((key, rec))
    return out


def discover_objects(type_dir: str) -> List[Tuple[str, str]]:
    """List (name, path) object directories containing pickles within a loading-type folder."""
    if not os.path.isdir(type_dir):
        return []
    res = []
    for name in sorted(os.listdir(type_dir)):
        p = os.path.join(type_dir, name)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "**", "data CyclicModel*"), recursive=True):
            res.append((name, p))
    return res


def find_cloud_root(candidates: Sequence[os.PathLike | str]) -> Optional[Path]:
    """Return the first candidate containing raw test pickles, or None.

    Require at least one data CyclicModel* file; skip empty archive directories.
    """
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() and glob.glob(os.path.join(str(p), "**", "data CyclicModel*"), recursive=True):
            return p
    return None


# Property statements (.xls).

# Statement column positions, following DigitRock PhysicalPropertyPosition.
STATEMENT_COLS = {
    "laboratory_number": 0, "borehole": 1, "depth": 2, "soil_name": 3,
    "granulometric_10": 4, "granulometric_5": 5, "granulometric_2": 6, "granulometric_1": 7,
    "granulometric_05": 8, "granulometric_025": 9, "granulometric_01": 10, "granulometric_005": 11,
    "granulometric_001": 12, "granulometric_0002": 13, "granulometric_0000": 14,
    "rs": 15, "r": 16, "rd": 17, "n": 18, "e": 19, "W": 20, "Sr": 21, "Wl": 22, "Wp": 23,
    "Ip": 24, "Il": 25, "Ir": 30, "ground_water_depth": 35,
}


def read_statement(xls_path: str) -> pd.DataFrame:
    """Read an .xls property statement indexed by laboratory specimen number.

    Requires xlrd. Detect the first data row from a nonempty specimen number in
    column A, such as '1-1' or '12'.
    """
    raw = pd.read_excel(xls_path, header=None, engine="xlrd")
    start = None
    for i in range(min(40, len(raw))):
        a = str(raw.iat[i, 0]).strip()
        if re.match(r"^\d+(-\d+)?$", a):
            start = i; break
    if start is None:
        return pd.DataFrame()
    rows = []
    for i in range(start, len(raw)):
        a = str(raw.iat[i, 0]).strip()
        if not re.match(r"^\d+(-\d+)?$", a):
            continue
        rows.append({name: raw.iat[i, col] for name, col in STATEMENT_COLS.items()})
    return pd.DataFrame(rows).set_index("laboratory_number")


# Measured CRR(N) grouped by engineering-geological unit.

def _hyp(N, alpha, betta):
    """CSR(N) = beta/N^(1-alpha), following DigitRock define_cyclic_stress_ratio_hyp."""
    return betta / (N ** (1.0 - alpha))


def fit_alpha_betta(cycles, csr) -> Optional[Tuple[float, float]]:
    """Fit (alpha, beta) to (N_fail, CSR) pairs using the DigitRock fitting convention."""
    from scipy.optimize import curve_fit, differential_evolution

    cycles = np.asarray(cycles, float); csr = np.asarray(csr, float)
    m = (cycles > 0) & np.isfinite(cycles) & np.isfinite(csr) & (csr > 0)
    cycles, csr = cycles[m], csr[m]
    if len(cycles) < 3:
        return None
    try:
        de = differential_evolution(lambda p: float(np.sum((csr - _hyp(cycles, *p)) ** 2)),
                                    [(0, 1), (0, 1.5)], seed=3, maxiter=60, tol=1e-3, polish=False)
        popt, _ = curve_fit(_hyp, cycles, csr, p0=de.x, bounds=([0, 0], [0.999, 3.0]), maxfev=5000)
        a, b = float(popt[0]), float(popt[1])
        return (a, b) if (np.isfinite(a) and np.isfinite(b)) else None
    except Exception:
        return None


def build_crr_obs(soil_df: pd.DataFrame, load_df: pd.DataFrame, cycles: np.ndarray,
                  n_liq: np.ndarray, liq_label: np.ndarray, tags: Sequence[str],
                  seq_len: int, min_group: int = 3):
    """Fit group-level CRR(N) = beta/N^(1-alpha) to failed specimens in resistance-test series."""
    n = len(soil_df)
    crr = np.zeros((n, seq_len), np.float32); mask = np.zeros(n, np.float32)
    is_pot = np.array(["Потенциал" in str(t) for t in tags])
    csr_base = load_df["CSR_base"].to_numpy()
    key = (soil_df["object"].astype(str) + "|" + soil_df["ige"].astype(str)).to_numpy()
    df = pd.DataFrame({"i": np.arange(n), "key": key})
    n_groups = 0
    for _, grp in df.groupby("key"):
        idx = grp["i"].to_numpy(); pidx = idx[is_pot[idx]]
        fl = pidx[liq_label[pidx] == 1]
        if len(fl) < min_group:
            continue
        res = fit_alpha_betta(n_liq[fl], csr_base[fl])
        if res is None:
            continue
        a, b = res; n_groups += 1
        for i in pidx:
            crr[i] = np.clip(b / np.maximum(cycles[i], 1.0) ** (1.0 - a), 0.02, 0.9).astype(np.float32)
            mask[i] = 1.0
    return crr, mask, n_groups


def canonical_site_id(object_name: str) -> str:
    """Derive a canonical geological-site key from the object folder name.

    Related projects at the same address share a fold. Normalize names by removing
    the leading project code NNN-NN, parenthesized annotations, technical suffixes
    and punctuation, and by normalizing Russian spelling and whitespace. Matching
    addresses receive the same site_id.

    :param object_name: object folder name
    :return: normalized address key
    """
    s = str(object_name).replace("ё", "е").replace("Ё", "Е").lower()
    s = re.sub(r"\([^)]*\)", " ", s) # Parenthesized project annotations.
    s = re.sub(r"^\s*\d+\s*[-_]\s*\d+\s*", " ", s) # Leading project code NNN-NN.
    s = re.sub(r"\b(plaxis|мех|вибро|сейсмо|сейсмо|g0|gо|этап)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s) # Replace punctuation with whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s or str(object_name).strip().lower()


# Assemble the dataset.

def build_real_objects_population(
    source_specs: Sequence[Tuple[os.PathLike | str, Sequence[str]]],
    config: ExperimentConfig,
    seq_len: Optional[int] = None,
    max_objects: Optional[int] = None,
) -> Dict[str, object]:
    """Read laboratory objects and assemble a prepared dataset.

    Discover objects, extract specimens, combine property/loading tables and PPR
    arrays, fit group CRR targets, then call build_population_from_experiments.

    :param source_specs: (archive root, loading-type folder names) pairs
    :param config: lengths, normalization and split settings
    :param seq_len: PPR grid length; defaults to config.seq_len
    :param max_objects: optional per-type object limit for short runs
    :return: prepared dataset dictionary
    :raises ValueError: if no specimens are found
    """
    seq_len = int(seq_len or config.seq_len)
    soil_rows: List[dict] = []; load_rows: List[dict] = []
    CY, CS, RM, RC, PR, VM, LB, NL, TAG, QM, EM = [], [], [], [], [], [], [], [], [], [], []
    SITE: List[str] = [] # Geological site_id.
    multi = len(source_specs) > 1

    for root, types in source_specs:
        root = Path(root)
        rtag = root.name
        for test_type in types:
            objects = discover_objects(str(root / test_type))
            if max_objects:
                objects = objects[:max_objects]
            _lm = getattr(config, "prefix_mode", "preonset") == "landmark"
            _lm_n0 = float(getattr(config, "prefix_landmark_cycles", 20.0)) if _lm else None
            _lm_k = int(getattr(config, "prefix_len", 12)) if _lm else None
            # Query-grid endpoint uses the prescribed horizon or, if absent,
            # the common max_cycle_reference; it does not use last_obs.
            _hz = float(getattr(config, "max_cycle_reference", 3000.0))
            _sustain = int(getattr(config, "onset_sustain_cycles", 3))
            for oname, opath in objects:
                recs = load_object(opath, test_type, seq_len, landmark_n0=_lm_n0, landmark_k=_lm_k,
                                   horizon_default=_hz, onset_sustain_cycles=_sustain,
                                   query_prefix_horizon=config.query_prefix_horizon)
                otag = f"{rtag} · {test_type}/{oname}" if multi else f"{test_type}/{oname}"
                _site = canonical_site_id(oname) # Geological-site address.
                for _key, (soil, load, arr, liq, nl) in recs:
                    soil["specimen_id"] = f"{otag}::{_key}"
                    soil_rows.append(soil); load_rows.append(load)
                    CY.append(arr["cycles"]); CS.append(arr["csr"]); RM.append(arr["r"])
                    RC.append(arr.get("r_causal", arr["r"]))
                    PR.append(arr.get("prefix_r", arr["r"])); VM.append(arr["mask"])
                    QM.append(arr["q"]); EM.append(arr["eps"])
                    LB.append(liq); NL.append(nl); TAG.append(otag); SITE.append(_site)

    if not soil_rows:
        raise ValueError("No objects found; check source_specs and the laboratory archive path.")

    # Impute absent N_max from other planned durations.
    # Use the same-object median when available;
    # otherwise use the global median of planned values.
    from collections import defaultdict as _dd
    _planned_all = [float(r["N_max"]) for r in load_rows if r.get("N_max_is_planned")]
    _glob = float(np.median(_planned_all)) if _planned_all else float(config.max_cycle_reference)
    _obj_planned = _dd(list)
    for r, tag in zip(load_rows, TAG):
        if r.get("N_max_is_planned"):
            _obj_planned[tag].append(float(r["N_max"]))
    for i, (r, tag) in enumerate(zip(load_rows, TAG)):
        if not r.get("N_max_is_planned"):
            # The imputed planned duration is retained as N_max metadata.
            # Do not increase it using N_liq, which would introduce outcome information.
            # N_liq censoring remains tied to the actual last_obs;
            # planned horizon and censoring time describe different quantities.
            r["N_max"] = float(np.median(_obj_planned[tag])) if _obj_planned.get(tag) else _glob
    _imp_vals = [float(r["N_max"]) for r in load_rows if not r.get("N_max_is_planned")]
    _n_imputed = len(_imp_vals)
    if _n_imputed:
        _iv = np.array(_imp_vals)
        # Report the count and distribution of imputed N_max values.
        print(f"[N_max] planned cycles_count is missing for {_n_imputed}/{len(load_rows)} "
              f"({100*_n_imputed/len(load_rows):.1f}%); imputed with the object median, without N_liq. "
              f"Imputed N_max: min/median/max = {_iv.min():.0f}/{np.median(_iv):.0f}/{_iv.max():.0f}.")
    # Remove event_in_prefix from the loading-feature table,
    # retaining a separate array for eligibility and diagnostics.
    EIP = [int(r.get("event_in_prefix", 0)) for r in load_rows]
    OTA = [int(r.get("onset_terminal_ambiguous", 0)) for r in load_rows]
    for r in load_rows:
        r.pop("N_max_is_planned", None) # Internal flag; exclude from features.
        r.pop("event_in_prefix", None) # Causal eligibility flag; retain only in metadata.
        r.pop("onset_terminal_ambiguous", None) # Label diagnostic; exclude from features.

    # Event labels in metadata use the same horizon H
    # as splitting, training and evaluation.
    _H = float(config.max_cycle_reference)
    for i in range(len(LB)):
        if LB[i] > 0.5 and NL[i] > _H:
            LB[i] = 0.0 # An event after H is a non-event within the scoring window.
            NL[i] = _H # Right-censor at H.

    cohort_filter_counts = {
        "raw_specimens": int(len(LB)),
        "raw_objects": int(len(set(TAG))),
        "excluded_event_in_prefix_causal": 0, # Event observed within the causal window through N0.
        "excluded_event_before_N0": 0, # Retrospective onset<=N0: target already occurred.
        "excluded_censored_before_N0": 0, # Censoring before N0: no eligible risk period.
        "audit_prefix_crossed_but_target_after_N0": 0, # Causal crossing by N0, retrospective onset after N0.
        "audit_target_by_N0_but_prefix_not_crossed": 0, # Retrospective onset by N0 without a causal-prefix crossing.
        "terminal_onset_ambiguous": int(sum(OTA)), # Total terminal-ambiguous records.
        "excluded_terminal_ambiguous": 0, # Records excluded when the ambiguity flag is enabled.
    }
    _excl_ambig = bool(getattr(config, "exclude_terminal_ambiguous", True))
    # Exclude observed crossings by N0, retrospective onset<=N0
    # and censoring<=N0. Count discrepancies between the
    # causal-prefix crossing and the retrospective target
    # without changing measured inputs.
    if getattr(config, "prefix_mode", "preonset") == "landmark":
        _N0 = float(getattr(config, "prefix_landmark_cycles", 20.0))
        cohort_filter_counts["excluded_event_in_prefix_causal"] = int(sum(EIP))
        cohort_filter_counts["excluded_event_before_N0"] = int(sum(
            LB[i] > 0.5 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["excluded_censored_before_N0"] = int(sum(
            LB[i] < 0.5 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["audit_prefix_crossed_but_target_after_N0"] = int(sum(
            EIP[i] == 1 and NL[i] > _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["audit_target_by_N0_but_prefix_not_crossed"] = int(sum(
            EIP[i] == 0 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["excluded_terminal_ambiguous"] = int(sum(
            _excl_ambig and OTA[i] and EIP[i] == 0 and NL[i] > _N0 for i in range(len(LB))
        ))
        # Keep eligible post-N0 records, optionally excluding terminal ambiguity.
        _keep = [i for i in range(len(LB))
                 if (EIP[i] == 0 and NL[i] > _N0 and not (_excl_ambig and OTA[i]))]
        if len(_keep) < len(LB):
            soil_rows = [soil_rows[i] for i in _keep]; load_rows = [load_rows[i] for i in _keep]
            CY = [CY[i] for i in _keep]; CS = [CS[i] for i in _keep]; RM = [RM[i] for i in _keep]
            RC = [RC[i] for i in _keep]
            PR = [PR[i] for i in _keep]; VM = [VM[i] for i in _keep]
            QM = [QM[i] for i in _keep]; EM = [EM[i] for i in _keep]
            LB = [LB[i] for i in _keep]; NL = [NL[i] for i in _keep]; TAG = [TAG[i] for i in _keep]
            SITE = [SITE[i] for i in _keep]
            OTA = [OTA[i] for i in _keep]

    soil_df = pd.DataFrame(soil_rows); load_df = pd.DataFrame(load_rows)
    cycles = np.array(CY); csr = np.array(CS)
    r_measured = np.array(RM); valid_mask = np.array(VM)
    liq_label = np.array(LB); n_liq = np.array(NL)
    soil_df["object"] = TAG
    soil_df["site_id"] = SITE # Geological site for grouped cross-validation.
    soil_df["onset_terminal_ambiguous"] = np.asarray(OTA, dtype=np.int8) # Audit column kept only in metadata.

    crr_obs, crr_obs_mask, _n_groups = build_crr_obs(
        soil_df, load_df, cycles, n_liq, liq_label, soil_df["object"].tolist(), seq_len)
    if crr_obs_mask.sum() == 0:
        crr_obs = crr_obs_mask = None

    population = build_population_from_experiments(
        soil_df=soil_df, load_df=load_df, cycles=cycles, csr=csr, r_measured=r_measured,
        valid_mask=valid_mask, liq_label=liq_label, n_liq=n_liq, config=config,
        prefix_source=np.array(PR, np.float32),
        crr_obs=crr_obs, crr_obs_mask=crr_obs_mask)
    population["cohort_filter_counts"] = cohort_filter_counts
    population["r_causal"] = np.array(RC, np.float32)
    # Retain deviator q(N) and strain epsilon(N) in the artifact;
    # the serializer writes these additional ndarray fields for diagnostics.
    population["q_obs"] = np.array(QM, np.float32)
    population["eps_obs"] = np.array(EM, np.float32)
    return population
