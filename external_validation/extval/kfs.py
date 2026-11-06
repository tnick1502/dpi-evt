"""Karlsruhe fine sand (Wichtmann & Triantafyllidis 2016, Acta Geotechnica 11:739-761).

Raw data: https://www.torsten-wichtmann.de/KFSDB-TCUI.htm (and -TCUA, -TMD).  Per-cycle peaks were
extracted from the '*-test-data-vs-time.dat' files of each test (u, p, q, eps1 vs time) by
upward zero-crossings of (q - q_static) with ±0.3·q_ampl hysteresis:
    ru_max(N) = (max_{cycle N} u - u0) / p0'   (u0, p0' = values at the start of undrained cycling)
Cycles 1..30 are kept at full resolution, and later cycles on a quadratically spaced subsample
of at most 50 points plus the last cycle. This is sufficient for the monotone envelope and the
landmark grid. Integers in kfs_cycles_part*.txt are multiplied by 1000 (ru, and eps1 in percent),
and each block carries a checksum.

Material constants (Wichtmann & Triantafyllidis 2016, Part I): d50 = 0.14 mm, Cu = 1.5,
e_min = 0.677, e_max = 1.054, Gs = 2.65, subangular quartz fine sand, no plastic fines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import soil_row

from .paths import RAW as _RAW
RAW = str(_RAW / "kfs")
EMIN, EMAX, GS = 0.677, 1.054, 2.65
# Grain-size fractions with internal labels (lower bound, mm) for d50=0.14, Cu=1.5 (d10≈0.10, d60≈0.15).
KFS_FRAC = {"025": 2.0, "01": 88.0, "005": 9.5, "001": 0.5}


def read_cycles():
    out = {}
    for f in ["kfs_cycles_part1.txt", "kfs_cycles_part2.txt"]:
        L = open(f"{RAW}/{f}").read().strip().split("\n")
        for i in range(0, len(L), 2):
            h = L[i].split(); name, n, s = h[0][1:], int(h[1]), int(h[2])
            rows = np.array([list(map(int, r.split())) for r in L[i + 1].split(";")])
            assert len(rows) == n and rows.sum() == s, f"checksum failed for {name}"
            out[name] = pd.DataFrame({"N": rows[:, 0], "ru_max": rows[:, 1] / 1000, "ru_end": rows[:, 2] / 1000,
                                      "eps_max": rows[:, 3] / 1000, "eps_min": rows[:, 4] / 1000})
    return out


def fit_tmd():
    """E50 and phi_peak of KFS from the 25 drained monotonic TMD tests of the same database."""
    t = pd.read_csv(f"{RAW}/kfs_tmd_summary.csv")
    X = np.c_[np.ones(len(t)), np.log(t.p0 / 100), t.e0]
    bE = np.linalg.lstsq(X, np.log(t.E50_MPa), rcond=None)[0]
    X2 = np.c_[np.ones(len(t)), t.e0]
    bP = np.linalg.lstsq(X2, t.phi_peak, rcond=None)[0]
    E50 = lambda p, e: float(np.exp(bE[0] + bE[1] * np.log(p / 100) + bE[2] * e))  # noqa: E731
    phi = lambda e: float(bP[0] + bP[1] * e)  # noqa: E731
    return E50, phi, bE, bP


def records(series=("TCUI", "TCUA"), mode="seismic", frequency=0.5, xi=0.024, E_scale=1.0,
            event_rule="first_crossing_DA_confirmed", drop_duplicate=True, include_extension=False,
            k0_feature="test", control_amplitude="measured"):
    """Build external-validation records.

    event_rule:
      'first_crossing_DA_confirmed' (primary): N_liq is the first cycle with ru_max ≥ 0.95, and the
          test counts as liquefied when ε_DA ≥ 5 % is also reached (checked at full resolution in
          kfs_events.csv). KFS tests stop at liquefaction, so the internal three-cycle window does
          not fit in six tests, which the internal rule would mark as terminal-ambiguous and drop.
      'repo_sustained3': the internal three-cycle rule; terminal-ambiguous tests become censored.
    """
    meta = pd.read_csv(f"{RAW}/kfs_meta.csv")
    ev = pd.read_csv(f"{RAW}/kfs_events.csv").set_index("test")
    cyc = read_cycles()
    E50, phi, *_ = fit_tmd()
    recs = []
    for _, m in meta.iterrows():
        name = m.test
        if name[:4] not in series or name not in cyc:
            continue
        if drop_duplicate and name == "TCUA10":   # TCUA10.zip duplicates TCUA13.zip byte for byte.
            continue
        if (m.eta0 < 0) and not include_extension:  # Extension side, sigma'3/sigma'1 > 1, absent from the internal data.
            continue
        d = cyc[name]
        e = EMAX - m.ID0 * (EMAX - EMIN)
        p0 = float(m.p0)
        q0 = float(m.q0) if name.startswith("TCUA") else 0.0
        sigma_1 = p0 + 2.0 * q0 / 3.0            # axial effective stress at consolidation
        sigma_3 = p0 - q0 / 3.0
        rho = (GS + e) / (1 + e)                 # saturated density, g/cm3
        if control_amplitude not in ("measured", "nominal"):
            raise ValueError("control_amplitude must be measured or nominal")
        qa = (float(m.qampl_nom) if control_amplitude == "nominal" else
              float(m.qa_meas) if m.qa_meas > 0.5 * m.qampl_nom else float(m.qampl_nom))
        csr = qa / (2.0 * sigma_1)               # digitrock: CSR = t / sigma_1, t = q_ampl/2
        ru = d.ru_max.to_numpy() * p0 / sigma_3  # normalise by sigma'_3 (= p0 for TCUI)
        soil = soil_row(e=e, Ip=0.0, rho=rho, E_mpa=E50(p0, e) * E_scale, phi_deg=phi(e),
                        sigma_1=sigma_1, frac=KFS_FRAC, type_ground=4, xi=xi,
                        name=name, obj="KFS-" + name[:4])
        # In the internal data K0 is the consolidation stress ratio of the test, sigma'3 = K0*sigma'1
        # (q_static = sigma'1*(1-K0) with correlation 1.000 over the 812 specimens), and not a Jaky
        # in-situ estimate. The TCUI tests therefore map to K0=1 and TCUA with eta0>0 to K0 in [0.4, 0.79].
        if k0_feature == "test":
            soil["K0"] = float(np.clip(sigma_3 / sigma_1, 0.2, 1.0)); soil["miss_K0"] = 0.0
        soil["static_shear_ratio"] = 0.0         # Constant 0 in the internal data; static shear enters through K0.
        load = dict(CSR_base=csr, frequency=frequency, amp_scale=1.0, N_max=float(d.N.iloc[-1]),
                    nonstationarity=0.0, load_mode=mode)
        epsDA = (d.eps_max - d.eps_min).to_numpy()
        e_ = ev.loc[name]
        if event_rule == "repo_sustained3":
            liq = int(e_.N_sustained3 > 0); nl = float(e_.N_sustained3) if liq else float(d.N.iloc[-1])
        else:
            if name.startswith("TCUI"):          # Full-resolution event table, sigma'3 = p0.
                nf, nda = int(e_.N_first095), int(e_.N_DA5)
            else:                                # Recompute with the sigma'3 normalization.
                hit = np.where(ru >= 0.95)[0]
                nf = int(d.N.iloc[hit[0]]) if len(hit) else -1
                nda = int(e_.N_DA5)
            ok = (nf > 0) and (nda > 0)          # Both ru_max >= 0.95 and eps_DA >= 5 % are reached.
            liq = int(ok); nl = float(nf) if liq else float(d.N.iloc[-1])
        recs.append(dict(name=name, object="KFS-" + name[:4], soil=soil, load=load,
                         N=d.N.to_numpy(float), ru=ru,
                         event=dict(liq=liq, n_liq=nl, rule=event_rule),
                         info=dict(ID0=m.ID0, e=e, p0=p0, eta0=m.eta0, q0=q0, qampl=qa, CSR=csr, K0_test=sigma_3 / sigma_1,
                                   reversal=bool(qa > q0),
                                   epsDA_end=float(epsDA[-1]), ncyc=int(m.ncyc))))
    return recs
