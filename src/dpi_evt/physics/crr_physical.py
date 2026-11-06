"""Property-based CRR(N) model in hyperbolic-power form.

CRR(N) = beta / N^(1-alpha), with
CRR_ref = CRR15_density(Dr) * product(factor_i), s=1-alpha,
and beta = CRR_ref * N_ref^s, usually at N_ref=15.

Factors represent density, fines/plasticity, liquidity, organics, effective
overburden, OCR, Vs1, cementation/age, static shear, loading mode, saturation,
damping, frequency and K0 anisotropy. The numpy implementation follows the
property-model conventions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "RESPONSE_TYPES",
    "compute_crr_components",
    "crr_curve",
]

RESPONSE_TYPES = ("contractive", "transitional", "dilative", "plastic")
"""Cyclic-response categories."""


def _sigmoid(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Evaluate a smooth logistic transition.

    :param x: input array
    :param center: transition center
    :param scale: positive transition width
    :return: logistic values in (0, 1)
    """
    scale = max(abs(scale), 1e-9)
    z = np.clip((x - center) / scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _base_crr15_from_density(dr: np.ndarray) -> np.ndarray:
    """Reference CRR at 15 cycles for clean uncemented sand.

    CRR15 = clip(0.055 + 0.53*Dr^1.65, 0.06, 0.62).

    :param dr: relative density, fraction
    :return: CRR15 before property correction factors
    """
    return np.clip(0.055 + 0.53 * np.power(np.clip(dr, 0.0, 1.0), 1.65), 0.06, 0.62)


def _fines_plasticity_factor(fc: np.ndarray, ip: np.ndarray, clay: np.ndarray) -> np.ndarray:
    """Fines/plasticity correction to CRR_ref.

    :param fc: fines smaller than 0.075 mm, percent
    :param ip: plasticity index, percent
    :param clay: clay smaller than 0.002 mm, percent
    :return: CRR_ref multiplier
    """
    fc = np.clip(fc, 0.0, 100.0)
    ip = np.maximum(ip, 0.0)
    clay = np.clip(clay, 0.0, 100.0)

    nonplastic_valley = np.exp(-((fc - 22.0) / 18.0) ** 2)
    val_lowplastic = 1.0 - 0.22 * nonplastic_valley + 0.04 * (fc / 100.0)
    val_midplastic = 0.98 + 0.006 * (ip - 7.0) * np.minimum(fc, 45.0) / 35.0
    val_highplastic = 1.05 + 0.010 * (ip - 15.0) * np.minimum(fc, 60.0) / 60.0

    factor = np.select(
        [fc <= 5.0, ip < 7.0, ip < 15.0],
        [np.ones_like(fc), val_lowplastic, val_midplastic],
        default=val_highplastic,
    )
    factor = np.where((clay > 20.0) & (ip < 7.0), factor * 0.92, factor)
    factor = np.where(
        (clay > 20.0) & (ip >= 12.0),
        factor * (1.0 + 0.08 * np.minimum((clay - 20.0) / 40.0, 1.0)),
        factor,
    )
    return np.clip(factor, 0.70, 1.38)


def _liquidity_factor(il: np.ndarray) -> np.ndarray:
    """Liquidity-index correction.

    :param il: liquidity index
    :return: CRR_ref multiplier
    """
    softening = np.exp(-0.36 * np.clip(il, 0.0, 1.8))
    hardening = 1.0 + 0.12 * np.clip(-il, 0.0, 1.2)
    return np.clip(softening * hardening, 0.52, 1.16)


def _organic_factor(ir: np.ndarray) -> np.ndarray:
    """Organic-content correction.

    :param ir: organic content, percent
    :return: CRR_ref multiplier
    """
    return np.clip(np.exp(-0.0075 * np.maximum(ir, 0.0)), 0.55, 1.0)


def _overburden_factor(sigma_eff: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """Effective-overburden correction K_sigma.

    :param sigma_eff: effective stress, kPa
    :param dr: relative density, fraction
    :return: CRR_ref multiplier
    """
    sigma = np.maximum(sigma_eff, 5.0)
    pa = 100.0
    n160 = np.clip(2.0 + 46.0 * np.power(dr, 2.0), 2.0, 45.0)
    denom = np.maximum(18.9 - 2.55 * np.sqrt(n160), 3.5)
    c_sigma = np.minimum(1.0 / denom, 0.30)
    k_sigma = 1.0 - c_sigma * np.log(sigma / pa)
    return np.clip(k_sigma, 0.65, 1.10)


def _ocr_factor(ocr: np.ndarray) -> np.ndarray:
    """Overconsolidation correction.

    :param ocr: overconsolidation ratio
    :return: CRR_ref multiplier
    """
    return np.clip(np.power(np.maximum(ocr, 0.4), 0.16), 0.85, 1.35)


def _vs_factor(vs1: np.ndarray) -> np.ndarray:
    """Stress-corrected shear-wave-velocity correction.

    :param vs1: corrected velocity, m/s
    :return: CRR_ref multiplier
    """
    return np.clip(0.74 + 0.58 * _sigmoid(vs1, center=170.0, scale=34.0), 0.76, 1.30)


def _cementation_aging_factor(cementation: np.ndarray, aging_years: np.ndarray) -> np.ndarray:
    """Cementation and aging correction.

    :param cementation: cementation index, 0-3
    :param aging_years: soil age in years
    :return: CRR_ref multiplier
    """
    cementation = np.clip(cementation, 0.0, 3.0)
    aging = np.maximum(aging_years, 0.0)
    return np.clip(1.0 + 0.18 * cementation + 0.055 * np.log10(1.0 + aging), 1.0, 1.55)


def _static_shear_factor(a_static: np.ndarray, response_idx: np.ndarray, dr: np.ndarray) -> np.ndarray:
    """Initial static-shear correction K_alpha.

    :param a_static: initial static-shear ratio
    :param response_idx: response index: 0 contractive, 1 transitional, 2 dilative, 3 plastic
    :param dr: relative density, fraction
    :return: CRR_ref multiplier
    """
    a = np.clip(a_static, 0.0, 0.45)
    contractive = 1.0 - (1.85 - 0.75 * dr) * a
    dilative = 1.0 + 1.15 * a * np.clip((dr - 0.50) / 0.45, 0.0, 1.0)
    plastic = 1.0 - 0.35 * a
    transitional = 1.0 - 0.55 * a + 0.35 * a * dr
    factor = np.select(
        [response_idx == 0, response_idx == 2, response_idx == 3],
        [contractive, dilative, plastic],
        default=transitional,
    )
    factor = np.where(a <= 0.0, 1.0, factor)
    return np.clip(factor, 0.35, 1.35)


def _saturation_factor(sr: np.ndarray, b_value: np.ndarray) -> np.ndarray:
    """Partial-saturation correction.

    :param sr: degree of saturation, fraction
    :param b_value: Skempton B, fraction
    :return: CRR_ref multiplier
    """
    factor = (1.0 + 0.20 * np.clip(0.95 - b_value, 0.0, 0.35) / 0.35) * (
        1.0 + 0.12 * np.clip(0.98 - sr, 0.0, 0.30) / 0.30
    )
    return np.clip(factor, 1.0, 1.18)


def _damping_factor(damping_percent: np.ndarray) -> np.ndarray:
    """Initial material-damping correction.

    :param damping_percent: damping ratio, percent
    :return: CRR_ref multiplier
    """
    low = 1.0 + 0.018 * (damping_percent - 5.0) / 5.0
    high = 1.01 * np.exp(-0.018 * (damping_percent - 8.0))
    return np.clip(np.where(damping_percent <= 8.0, low, high), 0.72, 1.06)


def _frequency_factor(frequency: np.ndarray, ip: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Cyclic-loading frequency correction.

    :param frequency: loading frequency, Hz
    :param ip: plasticity index, percent
    :param fc: fines content, percent
    :return: CRR_ref multiplier
    """
    f = np.maximum(frequency, 0.02)
    weight = _sigmoid(ip, center=8.0, scale=4.0) * _sigmoid(fc, center=25.0, scale=12.0)
    return np.clip(1.0 + 0.030 * np.log10(f / 1.0) * weight, 0.92, 1.12)


def _infer_response(dr, fc, ip, il, ocr) -> tuple:
    """Infer cyclic-response type and a state proxy.

    :param dr: relative density
    :param fc: fines content, percent
    :param ip: plasticity index, percent
    :param il: liquidity index
    :param ocr: overconsolidation ratio
    :return: response indices 0-3 and state proxy
    """
    state_proxy = (
        0.55 - dr + 0.0025 * fc + 0.18 * np.maximum(il, 0.0) - 0.010 * ip - 0.035 * np.log(np.maximum(ocr, 1.0))
    )
    idx = np.select(
        [(ip >= 12.0) & (fc >= 35.0), state_proxy > 0.14, state_proxy < -0.08],
        [np.full_like(state_proxy, 3), np.full_like(state_proxy, 0), np.full_like(state_proxy, 2)],
        default=np.full_like(state_proxy, 1),
    ).astype(int)
    return idx, state_proxy.astype(np.float32)


def _cycle_slope(dr, fc, ip, il, ocr, damping_percent, frequency, a_static, response_idx) -> np.ndarray:
    """Compute the cyclic-degradation slope s=1-alpha.

    :param dr: relative density
    :param fc: fines content, percent
    :param ip: plasticity index, percent
    :param il: liquidity index
    :param ocr: overconsolidation ratio
    :param damping_percent: damping ratio, percent
    :param frequency: loading frequency, Hz
    :param a_static: initial static-shear ratio
    :param response_idx: response indices
    :return: exponent s
    """
    plastic_blend = _sigmoid(ip, 8.0, 4.0) * _sigmoid(fc, 30.0, 12.0)
    slope = 0.305 - 0.075 * dr - 0.105 * plastic_blend
    slope = slope + np.select(
        [response_idx == 0, response_idx == 2, response_idx == 3],
        [0.035, -0.030, -0.060],
        default=0.0,
    )
    slope = slope + 0.018 * np.clip(il, 0.0, 1.5)
    slope = slope - 0.014 * np.log(np.maximum(ocr, 1.0))
    slope = slope + np.where(
        (a_static > 0.0) & (response_idx == 0), 0.045 * np.clip(a_static / 0.25, 0.0, 1.0), 0.0
    )
    slope = slope + 0.0030 * np.maximum(damping_percent - 5.0, 0.0)
    slope = slope - np.where(ip >= 8.0, 0.010 * np.log10(np.maximum(frequency, 0.02)), 0.0)
    return np.clip(slope, 0.08, 0.42)


def compute_crr_components(
    Dr: np.ndarray,
    Ip: np.ndarray,
    Il: np.ndarray,
    Ir: np.ndarray,
    fines_content: np.ndarray,
    clay_fraction: np.ndarray,
    sigma_eff: np.ndarray,
    OCR: np.ndarray,
    K0: np.ndarray,
    Vs1: np.ndarray,
    static_shear_ratio: np.ndarray,
    cementation_index: np.ndarray,
    aging_years: np.ndarray,
    saturation: np.ndarray,
    B: np.ndarray,
    damping_percent: np.ndarray,
    frequency: np.ndarray,
    loading_mode_factor: np.ndarray,
    reference_cycle: float = 15.0,
) -> Dict[str, np.ndarray]:
    """Compute property-model CRR parameters and their factor decomposition.

    Build CRR15 and correction factors, then CRR_ref, slope s and the curve
    CRR(N)=beta/N^(1-alpha). Also return response type and state proxy.

    :param Dr: relative density, fraction
    :param Ip: plasticity index, percent
    :param Il: liquidity index
    :param Ir: organic content, percent
    :param fines_content: fraction smaller than 0.075 mm, percent
    :param clay_fraction: fraction smaller than 0.002 mm, percent
    :param sigma_eff: effective stress, kPa
    :param OCR: overconsolidation ratio
    :param K0: at-rest lateral-earth-pressure coefficient
    :param Vs1: corrected shear-wave velocity, m/s
    :param static_shear_ratio: initial static-shear ratio
    :param cementation_index: cementation index, 0-3
    :param aging_years: soil age in years
    :param saturation: saturation degree, fraction
    :param B: Skempton B, fraction
    :param damping_percent: damping ratio, percent
    :param frequency: loading frequency, Hz
    :param loading_mode_factor: laboratory loading-mode multiplier
    :param reference_cycle: reference cycle N_ref, usually 15
    :return: alpha, betta, crr_ref, cycle_slope, response_idx, state_proxy and f_* factors
    """
    Dr = np.clip(np.asarray(Dr, dtype=np.float64), 0.05, 1.0)
    Ip = np.maximum(np.asarray(Ip, dtype=np.float64), 0.0)
    Il = np.asarray(Il, dtype=np.float64)
    fc = np.clip(np.asarray(fines_content, dtype=np.float64), 0.0, 100.0)
    clay = np.clip(np.asarray(clay_fraction, dtype=np.float64), 0.0, 100.0)
    ocr = np.maximum(np.asarray(OCR, dtype=np.float64), 0.35)
    a_static = np.clip(np.asarray(static_shear_ratio, dtype=np.float64), 0.0, 0.45)

    response_idx, state_proxy = _infer_response(Dr, fc, Ip, Il, ocr)
    anisotropy = np.clip(1.0 - 0.035 * (np.asarray(K0, dtype=np.float64) - 0.5), 0.94, 1.05)

    factors = {
        "f_density_crr15": _base_crr15_from_density(Dr),
        "f_fines_plasticity": _fines_plasticity_factor(fc, Ip, clay),
        "f_liquidity": _liquidity_factor(Il),
        "f_organic": _organic_factor(np.asarray(Ir, dtype=np.float64)),
        "f_overburden": _overburden_factor(np.asarray(sigma_eff, dtype=np.float64), Dr),
        "f_ocr": _ocr_factor(ocr),
        "f_vs1": _vs_factor(np.asarray(Vs1, dtype=np.float64)),
        "f_cementation_aging": _cementation_aging_factor(
            np.asarray(cementation_index, dtype=np.float64), np.asarray(aging_years, dtype=np.float64)
        ),
        "f_static_shear": _static_shear_factor(a_static, response_idx, Dr),
        "f_loading_mode": np.asarray(loading_mode_factor, dtype=np.float64),
        "f_saturation": _saturation_factor(
            np.asarray(saturation, dtype=np.float64), np.asarray(B, dtype=np.float64)
        ),
        "f_damping": _damping_factor(np.asarray(damping_percent, dtype=np.float64)),
        "f_frequency": _frequency_factor(np.asarray(frequency, dtype=np.float64), Ip, fc),
        "f_k0_anisotropy": anisotropy,
    }

    crr_ref = factors["f_density_crr15"].copy()
    for key, value in factors.items():
        if key != "f_density_crr15":
            crr_ref = crr_ref * value
    crr_ref = np.clip(crr_ref, 0.030, 0.900)

    slope = _cycle_slope(
        Dr, fc, Ip, Il, ocr, np.asarray(damping_percent, dtype=np.float64),
        np.asarray(frequency, dtype=np.float64), a_static, response_idx,
    )
    ref_cycle = max(float(reference_cycle), 1.0)
    alpha = np.clip(1.0 - slope, 0.58, 0.94)
    betta = np.clip(crr_ref * np.power(ref_cycle, slope), 0.045, 1.40)

    result = {
        "alpha": alpha,
        "betta": betta,
        "crr_ref": crr_ref,
        "cycle_slope": slope,
        "response_idx": response_idx,
        "state_proxy": state_proxy,
        "reference_cycle": np.full_like(alpha, ref_cycle),
    }
    result.update(factors)
    return result


def crr_curve(cycles: np.ndarray, alpha: np.ndarray, betta: np.ndarray) -> np.ndarray:
    """Evaluate CRR(N)=beta/N^(1-alpha).

    :param cycles: cycle grid, shape (n, seq_len) or (seq_len,)
    :param alpha: shape parameter, shape (n,)
    :param betta: scale parameter, shape (n,)
    :return: CRR values on the cycle grid
    """
    cycles = np.maximum(np.asarray(cycles, dtype=np.float64), 1.0)
    alpha = np.asarray(alpha, dtype=np.float64)
    betta = np.asarray(betta, dtype=np.float64)
    if cycles.ndim == 2:
        exponent = (1.0 - alpha)[:, None]
        return (betta[:, None] / np.power(cycles, exponent)).astype(np.float32)
    return (betta / np.power(cycles, 1.0 - alpha)).astype(np.float32)
