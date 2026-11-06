"""LEAP-2017 GWU stress-controlled cyclic triaxial tests on Ottawa F65 sand.

Source: El Ghoraiby, Park & Manzari (2018) "LEAP-2017 GWU Laboratory Tests", DesignSafe-CI
PRJ-1783, doi:10.17603/DS2210X, folder 'Cyclic Stress-Controlled Triaxial Tests' (23 tests,
e0 = 0.515 / 0.542 / 0.585, isotropic p'0 = 100 kPa, CSR = q_ampl/(2 p'0) = 0.10…0.60).
Raw columns: time_min, eps_a[%], eps_v, sigmaP_3_kPa, sigma_3_kPa, sigma_d_kPa.
Per-cycle peaks were extracted from the full time histories (cycles = upward zero crossings of
sigma_d with ±0.3 q_ampl hysteresis):
    ru_max(N) = max_{cycle N} [ (σ'3,0 − σ'3) + (σ3 − σ3,0) ] / σ'3,0      (= Δu/σ'3,0)
Loading period T = 1 min (e0 = 0.515, 0.542 and the CSR=0.10 test) or 3 min (other e0=0.585).
The 0.1 min sampling gives about 10 samples per 1-min cycle, so per-cycle ru peaks are slightly
under-sampled, with noise of a few hundredths near ru≈1.

Ottawa F65 (LEAP-2017 characterisation, El Ghoraiby et al. 2019): Gs = 2.665, e_min = 0.492,
e_max = 0.739, D50 ≈ 0.20 mm, Cu ≈ 1.6, fines (<0.075 mm) < 1 %.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import soil_row
from .kfs import fit_tmd

from .paths import RAW as _RAW
RAW = str(_RAW / "leap2017_gwu")
GS, EMIN, EMAX = 2.665, 0.492, 0.739
# Grain-size fractions with internal labels (lower bound, mm).
F65_FRAC = {"025": 30.0, "01": 65.0, "005": 4.7, "001": 0.3}


def read_cycles():
    L = open(f"{RAW}/leap2017_ctx_cycles.txt").read().strip().split("\n")
    out = {}
    for i in range(0, len(L), 2):
        h = L[i].split()
        name, n, s = h[0][1:], int(h[1]), int(h[2])
        kv = dict(x.split("=") for x in h[3:])
        rows = np.array([list(map(int, r.split())) for r in L[i + 1].split(";")])
        assert len(rows) == n and rows.sum() == s, f"checksum failed for {name}"
        d = pd.DataFrame({"N": rows[:, 0], "ru_max": rows[:, 1] / 1000, "eps_max": rows[:, 2] / 1000,
                          "eps_min": rows[:, 3] / 1000})
        out[name] = (d, {k: float(v) for k, v in kv.items()})
    return out


def records(mode="seismic", frequency=0.5, xi=0.024, E_scale=1.0, event_rule="repo_sustained3",
            drop_last_partial=True, control_amplitude="measured"):
    """Build LEAP records with event_rule 'repo_sustained3' (primary) or 'first_crossing'.

    The primary rule can be used here because the full per-cycle resolution is available."""
    from dpi_evt.data.raw_loader import sustained_first_crossing
    pub = pd.read_csv(f"{RAW}/leap2017_nocyc_published.csv")
    cyc = read_cycles()
    E50, *_ = fit_tmd()
    recs = []
    for name, (d, kv) in cyc.items():
        e0_nom = float("0." + name.split("_")[2])
        csr_nom = float("0." + name.split("csr_0_")[1])
        m = pub[(np.isclose(pub.e0_nominal, e0_nom)) & (np.isclose(pub.CSR, csr_nom))]
        e = float(m.e0.iloc[0]) if len(m) else e0_nom
        if drop_last_partial and len(d) > 3:
            d = d.iloc[:-1] if (d.eps_min.iloc[-1] > d.eps_min.iloc[-2] + 0.5) else d
        p0 = kv["s30"]
        if control_amplitude not in ("measured", "nominal"):
            raise ValueError("control_amplitude must be measured or nominal")
        csr = csr_nom if control_amplitude == "nominal" else kv["qa"] / (2.0 * p0)
        rho = (GS + e) / (1 + e)
        soil = soil_row(e=e, Ip=0.0, rho=rho, E_mpa=E50(p0, e) * E_scale, phi_deg=33.0,
                        sigma_1=p0, frac=F65_FRAC, type_ground=4, xi=xi, name=name, obj="LEAP2017-GWU")
        soil["K0"] = 1.0; soil["miss_K0"] = 0.0        # Isotropic consolidation; see the K0 note in kfs.py.
        ru = np.clip(d.ru_max.to_numpy(), 0, 1.0)
        N = d.N.to_numpy(float)
        if event_rule == "repo_sustained3":
            i = sustained_first_crossing(ru, thr=0.95, sustain=3, cycles=N)
        else:
            hit = np.where(ru >= 0.95)[0]; i = int(hit[0]) if len(hit) else -1
        liq = int(i >= 0); nl = float(N[i]) if liq else float(N[-1])
        T = kv.get("T", 1.0)
        load = dict(CSR_base=csr, frequency=frequency, amp_scale=1.0, N_max=float(N[-1]),
                    nonstationarity=0.0, load_mode=mode)
        recs.append(dict(name=name, object="LEAP2017-GWU", soil=soil, load=load, N=N, ru=ru,
                         event=dict(liq=liq, n_liq=nl, rule=event_rule),
                         info=dict(e=e, Dr=(EMAX - e) / (EMAX - EMIN), p0=p0, CSR=csr, CSR_nominal=csr_nom,
                                   period_min=T, N_pub_2p5pctSA=(int(m.N_cyc_published.iloc[0]) if len(m) else None),
                                   epsDA_max=float((d.eps_max - d.eps_min).max()))))
    return recs
