"""Adapazari and Christchurch laboratory collections from NGL.

* Adapazari: Sancio (2003), Bray & Sancio (2006); DesignSafe PRJ-4583, doi:10.17603/ds2-24hv-kf48 (ODC-BY).
  103 stress-controlled cyclic triaxial tests on undisturbed silts and clayey silts (ZETAS Istanbul, UC Berkeley).
* Christchurch: Beyzaei (2014) programme in the NGL database, DesignSafe PRJ-6435 (NGL database tables exported on
  2026-08-21), doi:10.17603/ds2-cpfr-c782 (ODC-BY). 42 cyclic triaxial tests on gel-push samples (University of
  Canterbury), using the first undrained dynamic stage of each test.

`data/external/ngl/ngl_cyclic_payload.txt` holds per-cycle series extracted from the full raw records (30 Hz, 50 points
per cycle) with the cycle definition used for LEAP and KFS (upward crossings of q − q_static with ±0.3·q_ampl hysteresis).
Each row contains N, ru*×1000 and ε_DA×10, where the corrected pressure ratio is
    ru*(N) = max_cycle [Δu − (Δq/3 + Δσ_cell)] / σ′3c.
Cycles 1–12 are kept at full resolution, and later cycles on a quadratically spaced subsample of at most 40 points plus
the last cycle. Event cycles (first ru* ≥ 0.95, the internal three-cycle rule, the same for the raw peak ru, and
ε_DA ≥ 2.5 or 5 %) were computed at full resolution and are stored in the header of each record together with a
checksum."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import soil_row, GRAN
from .paths import RAW as _RAW
from dpi_evt.data.grainsize import define_type_ground

RAW = _RAW / "ngl" / "ngl_cyclic_payload.txt"
SIZES = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.002, 0.001]
DATASETS = ("Adapazari", "Christchurch")


def _num(x):
    try:
        v = float(str(x).strip())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def read_payload():
    """Return {'ADA': metadata DataFrame with an N×3 integer 'rows' column, 'CHC': ...} after checksum checks."""
    lines = RAW.read_text().rstrip("\n").split("\n")
    out, cur, hdr, i = {}, None, None, 0
    while i < len(lines):
        L = lines[i]
        if L.startswith("#"):
            cur = L[1:]; out[cur] = []; hdr = lines[i + 1].split("|"); i += 2; continue
        vals = L.split("|"); rec = dict(zip(hdr, vals))
        rows = np.array([[int(v) for v in r.split()] for r in lines[i + 1].split(";")], int)
        assert len(rows) == int(rec["nr"]) and rows.sum() == int(rec["cs"]), f"checksum failed: {vals[0]}"
        rec["rows"] = rows
        out[cur].append(rec); i += 2
    return {k: pd.DataFrame(v) for k, v in out.items()}


def fractions_from_passing(p):
    """Convert percent passing at SIZES to the 11 internal fractions (lower bound in mm; '0000' is < 0.002 mm)."""
    P = np.minimum.accumulate(np.clip(np.asarray(p, float), 0, 100))
    edges = [100.0] + list(P[:10])            # 100 percent passes above 10 mm.
    frac = {}
    for k, (a, b) in zip(GRAN[:-1], zip(edges[:-1], edges[1:])):
        frac[k] = max(a - b, 0.0)
    frac[GRAN[-1]] = float(P[9])              # < 0.002 mm
    return frac


def training_imputers(training_meta=None):
    """Fit E (MPa), phi (deg) and c (kPa) as functions of soil type, e and sigma'1 on the training fold."""
    if training_meta is None:
        raise ValueError("NGL mechanical-property imputation requires the authorized internal training-fold metadata.")
    else:
        m = training_meta
    X = lambda d: np.c_[np.ones(len(d)), d.e.to_numpy(float), np.log(d.sigma_eff.to_numpy(float) / 100.0)]  # noqa: E731
    coefE, coefP, cmed = {}, {}, {}
    for t, d in m.groupby("type_ground"):
        if len(d) < 8:
            d = m[m.type_ground.between(t - 1, t + 1)]
        coefE[t] = np.linalg.lstsq(X(d), np.log(d.E_modulus.to_numpy(float) / 1000.0), rcond=None)[0]
        coefP[t] = np.linalg.lstsq(X(d)[:, :2], d.phi.to_numpy(float), rcond=None)[0]
        cmed[t] = float(np.median(d.cohesion)) * 1000.0
    lo, hi = float(m.E_modulus.min() / 1000), float(m.E_modulus.max() / 1000)

    def f(t, e, s1):
        t = int(t) if int(t) in coefE else min(coefE, key=lambda k: abs(k - int(t)))
        x = np.array([1.0, e, np.log(s1 / 100.0)])
        Em = float(np.clip(np.exp(x @ coefE[t]), lo, hi))
        ph = float(np.clip(x[:2] @ coefP[t], 10.0, 43.0))
        return Em, ph, cmed[t]
    return f


def _fill_passing(df):
    """Replace a missing grain-size curve by the median curve of the same dataset."""
    P = df["pass"].map(lambda s: [float(v) for v in s.split(",")] if isinstance(s, str) and s else None)
    med = np.median(np.array([p for p in P if p is not None]), axis=0)
    miss = P.isna()
    return P.map(lambda p: p if p is not None else list(med)), miss


def table(dataset):
    """Per-test table with the quality-control flags used for inclusion."""
    pl = read_payload()
    if dataset == "Adapazari":
        d = pl["ADA"].copy()
        d["test"] = d["name"]; d["object"] = "ADA-" + d["site"].str.replace("Site_", "")
        d["PI_num"] = d["PI"].map(_num); d["Ip"] = d["PI_num"].fillna(0.0)
        d["Ip_measured"] = d["PI"].map(lambda s: str(s).strip() not in ("", "-"))
        d["rho"] = d["gwet"].map(_num) / 9.81
        d["sh_nom"] = d["sh"].map(_num)
        d["B_num"] = d["B"].map(_num)
    else:
        d = pl["CHC"].copy()
        d["test"] = d["name"]; d["object"] = "CHC-" + d["site"]
        LL, PL = d["LL"].map(_num), d["PL"].map(_num)
        d["PI_num"] = np.where(LL > 0, LL - PL, 0.0); d["Ip"] = np.clip(d["PI_num"], 0, None)
        d["Ip_measured"] = LL.notna()
        Gs = d["Gs"].map(_num).fillna(2.65); w = d["w0"].map(_num) / 100.0; e = d["e0"].map(_num)
        d["rho"] = Gs * (1 + w) / (1 + e)
        d["sh_nom"] = np.nan
        d["B_num"] = d["B"].map(_num)
    for c in ("e0", "qs", "qa", "qa_cv", "s3c", "ncyc", "first95", "sus3", "first95_raw", "sus3_raw", "da5", "da25"):
        d[c] = d[c].map(_num)
    d["pass_list"], d["pass_imputed"] = _fill_passing(d)
    # Quality-control and inclusion rules.
    d["qc_stress_controlled"] = d.qa_cv.fillna(1.0) <= 0.10
    d["qc_B"] = d.B_num.isna() | (d.B_num >= 0.90)
    d["qc_meta"] = d.sh_nom.isna() | ((d.s3c - d.sh_nom).abs() / d.sh_nom <= 0.20)          # consistent metadata
    sig = d.rows.map(lambda r: hash(r.tobytes()))
    dup = sig.duplicated(keep=False)
    d["qc_unique"] = ~dup | d.qc_meta                                                          # duplicate records
    grp = d[dup & d.qc_meta].groupby(sig[dup & d.qc_meta]).size()
    if (grp > 1).any():                                                                        # both copies have consistent metadata
        bad = set(grp[grp > 1].index); d.loc[sig.isin(bad), "qc_unique"] = False
    d["qc_length"] = (d.ncyc >= 11) | (d.sus3 > 0)
    d["qc_pass"] = d.qc_stress_controlled & d.qc_B & d.qc_meta & d.qc_unique & d.qc_length
    return d


def records(dataset, target="corrected", event_rule="repo_sustained3", E_scale=1.0, frequency=0.5, mode="seismic",
            xi=0.024, include_failed_qc=False, training_meta=None):
    """Build records with target 'corrected' (primary ru*) or 'raw' (uncorrected peak ru, events only)."""
    d = table(dataset)
    imp = training_imputers(training_meta)
    recs = []
    for _, r in d.iterrows():
        if not (r.qc_pass or include_failed_qc):
            continue
        rows = r.rows
        N = rows[:, 0].astype(float); ru = rows[:, 1] / 1000.0; da = rows[:, 2] / 10.0
        s3 = float(r.s3c); qs = max(float(r.qs), 0.0); s1 = s3 + qs
        e = float(r.e0)
        frac = fractions_from_passing(r.pass_list)
        tg = int(define_type_ground(np.array([[frac[k] for k in GRAN]]), np.array([float(r.Ip)]), np.array([0.0]))[0])
        Em, phi, c = imp(tg, e, s1)
        soil = soil_row(e=e, Ip=float(r.Ip), rho=float(r.rho), E_mpa=Em * E_scale, phi_deg=phi, sigma_1=s1,
                        frac=frac, type_ground=tg, xi=xi, name=r.test, obj=r.object)
        soil["K0"] = float(np.clip(s3 / s1, 0.2, 1.0)); soil["miss_K0"] = 0.0
        soil["miss_Ip"] = 0.0 if (r.Ip_measured and r.Ip > 0) else 1.0
        soil["cohesion"] = c; soil["static_shear_ratio"] = 0.0
        csr = float(r.qa) / (2.0 * s1)
        if target == "corrected":
            n_ev = r.sus3 if event_rule == "repo_sustained3" else r.first95
        else:
            n_ev = r.sus3_raw if event_rule == "repo_sustained3" else r.first95_raw
        liq = int(n_ev > 0); nl = float(n_ev) if liq else float(N[-1])
        load = dict(CSR_base=csr, frequency=frequency, amp_scale=1.0, N_max=float(N[-1]), nonstationarity=0.0,
                    load_mode=mode)
        recs.append(dict(name=r.test, object=r.object, soil=soil, load=load, N=N, ru=np.clip(ru, 0, 1),
                         event=dict(liq=liq, n_liq=nl, rule=f"{target}/{event_rule}"),
                         info=dict(dataset=dataset, site=r.site, e=e, Ip=float(r.Ip), type_ground=tg, s3c=s3, s1=s1,
                                   K0=s3 / s1, CSR=csr, qs=qs, qa=float(r.qa), reversal=bool(r.qa > qs),
                                   N_DA5=r.da5, N_DA25=r.da25, N_first95=r.first95, N_first95_raw=r.first95_raw,
                                   N_sus3_raw=r.sus3_raw, E_imputed_MPa=Em, phi_imputed=phi,
                                   pass_imputed=bool(r.pass_imputed), fines_lt_0p1=100 - frac["025"] - frac["05"] - frac["1"] - frac["2"] - frac["5"] - frac["10"] - frac["01"],
                                   epsDA_end=float(da[-1]), ncyc=int(r.ncyc))))
    return recs
