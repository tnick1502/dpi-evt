"""Small-strain G0 and shear-wave velocity using DigitRock property relations.

Combine G0 = 0.6*G0_plaxis + 0.4*G0_empirical and the midpoint of the
DigitRock soil-type factor range. Random perturbations are omitted.
Vs [m/s] = sqrt(G0 [MPa] * 1000 / density [g/cm^3]).
G0 uses static p_ref, e, c, phi, K0, soil type and Ip.
"""

from __future__ import annotations

import numpy as np

__all__ = ["g0_mpa", "vs_from_g0"]

ATM = 0.1 * 1000.0 # Atmospheric pressure in kPa, following DigitRock.

# K_ground_type uses the midpoint of DigitRock dependence_Eur ranges.
_K_GROUND = {1: 0.75, 2: 0.75, 3: 0.80, 4: 0.825, 5: 0.90, 6: 0.975, 7: 1.15, 8: 1.30, 9: 0.75}


def _e50(e50ref: float, c: float, fi_deg: float, sigma_3: float, p_ref: float, m: float) -> float:
    """DigitRock general_functions.define_E50 without random perturbations."""
    fi = np.deg2rad(fi_deg)
    up = c * np.cos(fi) + sigma_3 * np.sin(fi)
    down = c * np.cos(fi) + p_ref * np.sin(fi)
    if down == 0:
        return e50ref
    return e50ref * (up / down) ** m


# Empirical G0 relations in MPa, with p_ref in MPa.
def _delia_sandy_silt(p, e): return 358 * e ** -1.21 * (p * 1000) ** 0.57 * ATM ** 0.43 / 1000
def _delia_clayey_silt(p, e): return 358 * e ** -1.21 * (p * 1000) ** 0.57 * ATM ** 0.43 / 1000
def _kallioglou(p, PI, e): return (6290 - 80 * PI) * e ** -0.63 * (p * 1000) ** 0.5 / 1000
def _sas(p): return (3.02 * (p * 1000) ** 0.68 + 0.82 * (p * 1000) ** 0.96) / 2
def _sands(p, e): return ((220 * (2.17 - e) ** 2 * (p * 1000) ** 0.623) / (1 + e)) * 0.5 / 1000
def _hardin_black(p, e): return 3231 * (2.97 - e) ** 2 / (1 + e) * (p * 1000) ** 0.5 / 1000
def _marcuson_wahls(p, e): return 445 * (4.4 - e) ** 2 / (1 + e) * (p * 1000) ** 0.5 / 1000
def _kim_novac(p, e): return 1576 * (2.97 + e) ** 2 / (1 + e) * (p * 1000) ** 0.5 / 1000
def _kokusho_alluvial(p, e): return 141 * (7.32 + e) ** 2 / (1 + e) * (p * 1000) ** 0.6 / 1000
def _jamiolkowski(p, e): return 600 * e ** -1.3 * (p * 1000) ** 0.5 * ATM ** 0.5 / 1000
def _shibuya_tanaka(p, e): return 5000 * e ** -1.3 * (p * 1000) ** 0.5 / 1000
def _vrettos_savidis(p, e): return 9600 * (1 / (1 + 1.2 * e ** 2)) * (p * 1000) ** 0.5 / 1000
def _clays(p, Ip, e):
    Ip = Ip if Ip and Ip > 0 else 1.0
    return ((((330 * (2.17 - e) ** 2 * p ** 0.5) * 1.4 / (1 + e)) + ((4000 * p) / (Ip ** 0.7))) / 2) * 0.5 / 1000


def _g0_plaxis(p, e, c, fi, tg):
    g0_ref = ((2.97 - e) ** 2 / (1 + e)) * 33
    g0 = _e50(g0_ref, c, fi, sigma_3=p, p_ref=0.1, m=0.5)
    if tg == 9:
        g0 *= 0.3
    elif tg in (1, 2, 3, 4, 5):
        g0 *= 0.7 - tg * 0.05
    elif tg in (6, 7, 8):
        g0 *= 1.1 - 0.1 * tg
    else:
        g0 *= 0.8
    return g0


def _g0_scalar(p_ref, e, c, fi, K0, type_ground, Ip):
    """Deterministic G0 in MPa using DigitRock define_G0_threshold_shear_strain."""
    p = max(float(p_ref), 0.01)
    e = float(e) if (e and e > 0.1) else 0.65
    c = float(c) if (c is not None and c > 0) else 0.001
    fi = float(fi) if (fi is not None and fi > 0) else 20.0
    tg = int(type_ground) if type_ground else 7
    PI = float(Ip) if (Ip and Ip > 0) else 0.0

    g0_plaxis = _g0_plaxis(p, e, c, fi, tg)

    if tg == 9:
        g0 = (_delia_sandy_silt(p, e) + _delia_clayey_silt(p, e) + _kallioglou(p, PI, e) + _sas(p)) / 4
    elif tg in (1, 2, 3, 4, 5):
        g0 = (_delia_sandy_silt(p, e) + _kallioglou(p, 0, e) + _sas(p) + _sands(p, e)) / 4
        g0 *= (1 + (0.75 - tg * 0.15))
    else: # Clay soil codes 6, 7 and 8.
        g0 = ((_hardin_black(p, e) + _marcuson_wahls(p, e) + _kim_novac(p, e) + _kokusho_alluvial(p, e)
               + _jamiolkowski(p, e) + _shibuya_tanaka(p, e) + _vrettos_savidis(p, e)
               + _kallioglou(p, PI, e) + _sas(p)) / 9) * 0.8 * 0.6 + _clays(p, PI, e) * 0.4

    g0 = g0_plaxis * 0.6 + g0 * 0.4
    g0 *= _K_GROUND.get(tg, 0.8)
    return float(max(g0, 1.0))


_g0_vec = np.vectorize(_g0_scalar, otypes=[float])


def g0_mpa(p_ref, e, c, fi, K0, type_ground, Ip) -> np.ndarray:
    """Vectorized G0 in MPa from static soil properties using DigitRock relations."""
    return _g0_vec(p_ref, e, c, fi, K0, type_ground, Ip)


def vs_from_g0(g0_mpa_arr, r_gcm3) -> np.ndarray:
    """Vs in m/s from G0 in MPa and density in g/cm^3: sqrt(G0*1000/density)."""
    g0 = np.asarray(g0_mpa_arr, dtype=float)
    r = np.asarray(r_gcm3, dtype=float)
    r = np.where(r > 0.1, r, 2.0)
    return np.sqrt(g0 * 1000.0 / r)
