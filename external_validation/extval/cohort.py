"""Physically admissible cohorts and the Mohr–Coulomb pore-pressure cap.

Mechanics.  In a stress-controlled triaxial test consolidated to σ'1c, σ'3c = K0·σ'1c, with static
deviator q_s = σ'1c(1−K0) and cyclic amplitude q_a = 2·CSR·σ'1c, the minimum deviator of a cycle is
q_min = q_s − q_a.  If q_min > 0 (no stress reversal) the effective stress state at q_min must stay
inside the Mohr–Coulomb envelope  σ'1 ≤ K_p σ'3 + 2c√K_p  (K_p = (1+sinφ)/(1−sinφ)), hence

        σ'3 ≥ σ'3,f = max(0, (q_min − 2c√K_p)/(K_p − 1))
        ru = Δu/σ'3c ≤ ru_MC = 1 − σ'3,f/σ'3c.

With stress reversal (q_min ≤ 0) the bound is 1 (σ'3 → 0 is reachable at q ≈ 0).

Functions
* `ru_cap(meta)` returns the admissible ceiling of each specimen, plus a tolerance, from its own
  φ, c, K0, CSR and σ'1. In the internal dataset φ and c come from the test statement, with c in MPa.
* `training_flags(pop)` flags internal specimens whose recorded ru_max exceeds the ceiling by more
  than the tolerance.
* `subset_population(pop, keep)` returns a row-filtered copy of a dataset dictionary, including all
  arrays and metadata.
"""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd

from . import paths  # noqa: F401

TOL = 0.02


def ru_cap(meta: pd.DataFrame, tol: float = TOL, c_to_kpa: float = 1000.0,
           phi_default: float = 35.0) -> np.ndarray:
    s1 = meta["sigma_eff"].to_numpy(float)
    K0 = meta["K0"].to_numpy(float)
    csr = meta["CSR_base"].to_numpy(float)
    phi = meta["phi"].fillna(phi_default).to_numpy(float) if "phi" in meta else np.full(len(meta), phi_default)
    c = meta["cohesion"].fillna(0.0).to_numpy(float) * c_to_kpa if "cohesion" in meta else np.zeros(len(meta))
    qmin = s1 * (1 - K0) - 2 * csr * s1
    s3c = np.maximum(K0 * s1, 1e-6)
    kp = (1 + np.sin(np.radians(phi))) / (1 - np.sin(np.radians(phi)))
    s3f = np.maximum((qmin - 2 * c * np.sqrt(kp)) / (kp - 1), 0.0)
    bound = np.where(qmin > 0, 1.0 - s3f / s3c, 1.0)
    return np.clip(bound + tol, 0.0, 1.0).astype(np.float32)


def training_flags(pop: Dict, tol: float = TOL) -> pd.DataFrame:
    m = pop["meta"].reset_index(drop=True)
    cap = ru_cap(m, tol=0.0)
    rmax = np.asarray(pop["r_obs"]).max(1)
    s1 = m.sigma_eff.to_numpy(float)
    qmin = s1 * (1 - m.K0.to_numpy(float)) - 2 * m.CSR_base.to_numpy(float) * s1
    df = pd.DataFrame(dict(object=m.object.str.split("/").str[-1], soil=m.soil_type, K0=m.K0,
                           CSR=m.CSR_base, phi=m.phi, cohesion_kPa=m.cohesion * 1000,
                           stress_reversal=qmin <= 0, ru_max=rmax, ru_MC=cap, liq=m.liq_label))
    df["mc_violation"] = df.ru_max > df.ru_MC + tol
    df["admissible"] = ~df.mc_violation
    return df


def subset_population(pop: Dict, keep: np.ndarray) -> Dict:
    keep = np.asarray(keep, bool)
    n = len(pop["meta"])
    out = {}
    for k, v in pop.items():
        if k == "meta":
            out[k] = v.loc[keep].reset_index(drop=True)
        elif isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            out[k] = v[keep]
        elif k == "benchmark":
            continue
        else:
            out[k] = copy.deepcopy(v)
    return out


EXTERNAL_EXCLUDE = {
    "TCUA1": "cyclic amplitude could not be applied (flow failure in the 1st cycle: measured q_ampl ≈ 20 of 120 kPa)",
    "TCUA10": "byte-identical duplicate of TCUA13 in the KFS database",
}


def external_flags(recs) -> pd.DataFrame:
    """Quality exclusions of external element tests, applied before evaluation."""
    return pd.DataFrame([dict(test=r["name"], exclude=r["name"] in EXTERNAL_EXCLUDE,
                              reason=EXTERNAL_EXCLUDE.get(r["name"], "")) for r in recs])
