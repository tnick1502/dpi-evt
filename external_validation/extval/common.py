"""Convert external element tests to the dataset format used for the internal data.

Each external test is described by one record with the fields
    soil  : physical properties computed with the conventions of
            dpi_evt.data.raw_loader.extract_test (D_r from dr_proxy(e), Vs from static E,
            K0 = 1 - sin(phi), fines as fractions below 0.1 mm with the internal labels)
    load  : dict(CSR_base, frequency, amp_scale, N_max, nonstationarity, load_mode)
    N     : physical cycle numbers of the per-cycle peaks, strictly increasing
    ru    : raw per-cycle peak pore-pressure ratio Δu_max / σ'_c
    event : dict(liq, n_liq, rule), the retrospective target defined by each adapter

`build_external_population` repeats the landmark branch of `extract_test` (monotone_smooth
envelope, landmark grid, causal prefix up to N0, event_in_prefix flag and cohort filter) and
then calls `build_population_from_experiments`. The features are therefore built by the same
code as for the internal data. `make_external_split` applies the scalers fitted on the internal
training split. They are not refitted on external data, which would pass statistics of the
target domain into a zero-shot evaluation."""
from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import paths  # noqa: F401  (puts <repo>/src on sys.path)
from dpi_evt.data.ppr_envelope import forecasting_cycles, monotone_smooth  # noqa: E402
from dpi_evt.data.raw_loader import sustained_first_crossing  # noqa: E402
from dpi_evt.data.real_adapter import build_population_from_experiments  # noqa: E402
from dpi_evt.data.raw_loader import LIQ_THRESHOLD  # noqa: E402

GRAN = ["10", "5", "2", "1", "05", "025", "01", "005", "001", "0002", "0000"]
GRAN_SIZE = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.002, 0.001]
PERM_BY_TYPE = {1: 1e-4, 2: 5e-5, 3: 1e-5, 4: 5e-6, 5: 1e-6, 6: 1e-7, 7: 1e-8, 8: 1e-9, 9: 1e-7}


def dr_proxy(e: float) -> float:
    """Same as raw_loader.dr_proxy. The training feature 'D_r' is a void-ratio proxy."""
    return float(np.clip((1.10 - float(e)) / (1.10 - 0.40), 0.0, 1.0))


def fines_clay_cu(frac: Dict[str, float]):
    """Same as raw_loader.fines_clay_cu for a measured grain-size distribution in percent."""
    v = {k: float(frac.get(k, 0.0)) for k in GRAN}
    fines = v["005"] + v["001"] + v["0002"] + v["0000"]
    clay = v["0000"]
    frac_pass = np.clip(np.cumsum([v[k] for k in GRAN][::-1])[::-1], 0, 100)
    d = np.array(GRAN_SIZE, float)
    order = np.argsort(frac_pass)
    D10 = np.interp(10, frac_pass[order], d[order])
    D60 = np.interp(60, frac_pass[order], d[order])
    Cu = float(np.clip(D60 / max(D10, 1e-4), 1.0, 200.0))
    return float(fines), float(clay), Cu


def vs_from_static_E(E_mpa: float, rho_gcm3: float, nu: float = 0.2) -> float:
    """Third raw_loader option: G = E/(2(1+nu)), Vs = sqrt(G/rho), clipped to [40, 600]."""
    G = float(E_mpa) / (2 * (1 + nu))
    return float(np.clip((G * 1e6 / (rho_gcm3 * 1e3)) ** 0.5, 40, 600))


def soil_row(*, e, Ip, rho, E_mpa, phi_deg, sigma_1, frac, type_ground, xi=0.024, nu=0.2,
             name="", obj="external"):
    """Build one soil dictionary with the raw_loader conventions described above."""
    fines, clay, Cu = fines_clay_cu(frac)
    Vs = vs_from_static_E(E_mpa, rho, nu)
    K0 = float(np.clip(round(1.0 - np.sin(np.pi * phi_deg / 180.0), 3), 0.2, 1.0))
    row = dict(
        laboratory_number=name, borehole=obj, depth=np.nan, soil_name=name,
        type_ground=int(type_ground), class_id=int(type_ground) - 1,
        e=float(e), D_r=dr_proxy(e), I_p=float(Ip), Il=0.0, Ir=0.0,
        V_s=Vs, Vs1=float(np.clip(Vs * (100.0 / max(sigma_1, 1.0)) ** 0.25, 60, 450)),
        xi=float(xi), sigma_eff=float(sigma_1), permeability=PERM_BY_TYPE[int(type_ground)],
        fines_content=fines, clay_fraction=clay, Cu=Cu, K0=K0, static_shear_ratio=0.0,
        ige=obj, rs=2.65, rd=np.nan, r=rho, n_porosity=100 * e / (1 + e), W=np.nan, Wl=np.nan,
        Wp=np.nan, saturation=1.0, ground_water_depth=np.nan, B_value=0.95, cohesion=0.0,
        phi=phi_deg, E_modulus=E_mpa, damping_ratio=xi * 100, calcite=np.nan, dolomite=np.nan,
        miss_e=0.0, miss_Ip=1.0, miss_K0=1.0, miss_vs=1.0, miss_gran=0.0,
    )
    for k in GRAN:
        row[f"gran_{k}"] = float(frac.get(k, 0.0))
    return row


def trajectory_arrays(N, ru, n_liq, liq, cfg):
    """Repeat the landmark branch of extract_test for per-cycle peaks (N, ru)."""
    N = np.asarray(N, float)
    ru = np.clip(np.asarray(ru, float), 0.0, 1.0)
    seq_len = int(cfg.seq_len)
    n0 = float(cfg.prefix_landmark_cycles)
    grid = forecasting_cycles(float(cfg.max_cycle_reference), seq_len, n0,
                              int(cfg.prefix_len), cfg.query_prefix_horizon)
    env = monotone_smooth(ru)
    if liq:  # Same post-onset adjustment as extract_test, so the target agrees with the label.
        i = int(np.searchsorted(N, n_liq))
        env[i:] = np.maximum(env[i:], LIQ_THRESHOLD)
        env = np.maximum.accumulate(env)
    r = np.interp(grid, N, env).astype(np.float32)
    last_obs = float(N[-1])
    mask = (grid <= last_obs + 1e-6).astype(np.float32)
    early = N <= n0 + 1e-9
    if early.sum() > 0:
        pe = monotone_smooth(ru[early])
        prefix_r = np.interp(grid, N[early], pe).astype(np.float32)
        eip = int(sustained_first_crossing(ru[early], thr=LIQ_THRESHOLD,
                                           sustain=int(cfg.onset_sustain_cycles), cycles=N[early]) >= 0)
    else:
        prefix_r = np.zeros(seq_len, np.float32)
        eip = 0
    return dict(cycles=grid, r=r, mask=mask, prefix_r=prefix_r, event_in_prefix=eip, last_obs=last_obs)


def build_external_population(records: List[dict], cfg, apply_cohort_filter=True):
    """Convert records to a dataset dictionary and an audit DataFrame."""
    SO, LO, CY, CS, RM, VM, PR, LB, NL, TAG, AUD = [], [], [], [], [], [], [], [], [], [], []
    H = float(cfg.max_cycle_reference)
    n0 = float(cfg.prefix_landmark_cycles)
    for rec in records:
        ev = rec["event"]
        liq, nl = int(ev["liq"]), float(ev["n_liq"])
        ta = trajectory_arrays(rec["N"], rec["ru"], nl, liq, cfg)
        if not liq:
            nl = ta["last_obs"]
        if liq and nl > H:
            liq, nl = 0, H
        keep = (ta["event_in_prefix"] == 0) and (nl > n0)
        AUD.append(dict(test=rec["name"], liq=liq, n_liq=nl, event_rule=ev.get("rule", ""),
                        event_in_prefix=ta["event_in_prefix"], last_obs=ta["last_obs"],
                        kept=bool(keep or not apply_cohort_filter), note=rec.get("note", "")))
        if apply_cohort_filter and not keep:
            continue
        so = dict(rec["soil"])
        so["specimen_id"] = f"{rec.get('object', 'external')}::{rec['name']}"
        SO.append(so); LO.append(rec["load"])
        CY.append(ta["cycles"]); CS.append(np.full(cfg.seq_len, rec["load"]["CSR_base"], np.float32))
        RM.append(ta["r"]); VM.append(ta["mask"]); PR.append(ta["prefix_r"])
        LB.append(liq); NL.append(nl); TAG.append(rec.get("object", "external"))
    soil_df = pd.DataFrame(SO)
    load_df = pd.DataFrame(LO)
    soil_df["object"] = TAG
    soil_df["site_id"] = TAG
    soil_df["onset_terminal_ambiguous"] = 0
    pop = build_population_from_experiments(
        soil_df=soil_df, load_df=load_df, cycles=np.array(CY), csr=np.array(CS),
        r_measured=np.array(RM), valid_mask=np.array(VM), liq_label=np.array(LB, float),
        n_liq=np.array(NL, float), config=cfg, prefix_source=np.array(PR, np.float32))
    pop["meta"]["test"] = [r for r, a in zip([a["test"] for a in AUD], AUD) if a["kept"]]
    return pop, pd.DataFrame(AUD)


def make_external_split(pop: Dict, scalers: Dict, cfg, device=torch.device("cpu")):
    """Counterpart of splits.prepare_benchmark_dataset for one external split and fitted training scalers."""
    n = len(pop["meta"])
    static_raw = pop["static_features"]
    prefix_raw = pop["prefix_summary"]
    seq_raw = pop["seq_inputs"]
    static_scaled = scalers["static"].transform(static_raw).astype(np.float32)
    prefix_scaled = scalers["prefix"].transform(prefix_raw).astype(np.float32)
    seq_scaled = ((seq_raw - scalers["seq_mean"][None, None, :]) / scalers["seq_std"][None, None, :]).astype(np.float32)
    H = float(cfg.max_cycle_reference)
    N0 = float(cfg.prefix_landmark_cycles)
    liq_label = pop["liq_label"].astype(np.float32).copy()
    n_liq_true = pop["n_liq_true"].astype(np.float32)
    liq_label[(n_liq_true > H) & (liq_label > 0.5)] = 0.0
    n_liq_target = np.minimum(n_liq_true, H).astype(np.float32)
    n_liq_norm = (np.log1p(n_liq_target) / np.log1p(H)).astype(np.float32)
    is_liq = liq_label > 0.5
    reached = n_liq_true >= H - 1e-6
    risk_obs = (is_liq | (~is_liq & reached)).astype(np.float32)
    nliq_valid = (n_liq_true > N0 + 1e-6).astype(np.float32)
    valid_mask = pop["valid_mask"].astype(np.float32)
    prefix_mask = pop["prefix_mask"].astype(np.float32)
    arrays = {
        "static": static_scaled, "static_raw": static_raw.astype(np.float32),
        "prefix_summary": prefix_scaled, "prefix_summary_raw": prefix_raw.astype(np.float32),
        "seq_in": seq_scaled, "seq_in_raw": seq_raw.astype(np.float32),
        "cycles": pop["cycles"].astype(np.float32), "delta_cycles": pop["delta_cycles"].astype(np.float32),
        "csr": pop["csr"].astype(np.float32), "r_obs": pop["r_obs"].astype(np.float32),
        "mask": valid_mask, "prefix_mask": prefix_mask, "prefix_obs": pop["prefix_obs"].astype(np.float32),
        "label": liq_label, "n_liq_true": n_liq_target, "n_liq_raw": n_liq_true,
        "n_liq_norm": n_liq_norm, "n_liq_observed": nliq_valid.copy(), "trajectory_mask": valid_mask,
        "continuation_mask": valid_mask * (1.0 - np.minimum(prefix_mask, 1.0)),
        "nliq_censor_valid": nliq_valid, "risk_label_observed": risk_obs,
        "regime_liq": is_liq.astype(np.float32),
        "regime_stable": (~is_liq & reached).astype(np.float32),
        "regime_unfinished": (~is_liq & ~reached).astype(np.float32),
        "g_obs": pop["g_obs"].astype(np.float32), "risk_proxy": pop["risk_proxy"].astype(np.float32),
    }
    split = {k: torch.from_numpy(v) for k, v in arrays.items()}
    split["meta"] = pop["meta"].reset_index(drop=True)
    split["indices"] = np.arange(n)
    return split
