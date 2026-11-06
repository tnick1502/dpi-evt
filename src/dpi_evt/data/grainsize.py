"""Grain-size properties, PLAXIS classes and GOST soil types.

Uses the property-model conventions for 11 size fractions, D10/D50/D60,
Cu = D60/D10, PLAXIS grain-size class and GOST types 1-9. Reference profiles
are also provided for each soil type.

Fraction keys and size intervals (mm): '10' >10, '5' 5-10, '2' 2-5,
'1' 1-2, '05' 0.5-1, '025' 0.25-0.5, '01' 0.1-0.25,
'005' 0.05-0.1, '001' 0.01-0.05, '0002' 0.002-0.01,
and '0000' <0.002. Values are mass percentages.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "FRACTION_KEYS",
    "FRACTION_BOUNDS",
    "TYPE_GROUND_NAMES",
    "TYPE_GROUND_NAMES_EN",
    "TYPE_GROUND_ARCHETYPES",
    "TYPE_GROUND_PROBS",
    "PLAXIS_CLASSES",
    "plaxis_classification",
    "define_type_ground",
    "sample_grain_size",
]

FRACTION_KEYS: List[str] = ["10", "5", "2", "1", "05", "025", "01", "005", "001", "0002", "0000"]
"""Grain-size fraction keys ordered from coarse to fine."""

FRACTION_BOUNDS: Dict[str, float] = {
    "10": 20.0, "5": 10.0, "2": 5.0, "1": 2.0, "05": 1.0, "025": 0.5,
    "01": 0.25, "005": 0.1, "001": 0.05, "0002": 0.01, "0000": 0.002,
}
"""Upper grain-size bounds in mm, expressed as sieve diameters."""

TYPE_GROUND_NAMES = {
    1: "Песок гравелистый",
    2: "Песок крупный",
    3: "Песок средней крупности",
    4: "Песок мелкий",
    5: "Песок пылеватый",
    6: "Супесь",
    7: "Суглинок",
    8: "Глина",
    9: "Торф",
}
"""Russian GOST soil-type names for type_ground."""

TYPE_GROUND_NAMES_EN = {
    1: "Gravelly sand",
    2: "Coarse sand",
    3: "Medium sand",
    4: "Fine sand",
    5: "Silty sand",
    6: "Sandy loam",
    7: "Loam",
    8: "Clay",
    9: "Peat",
}
"""English soil-type names."""

# Mean grain-size profiles by soil type, in mass percent.
TYPE_GROUND_ARCHETYPES: Dict[int, List[float]] = {
    1: [10.0, 12.0, 15.0, 18.0, 18.0, 15.0, 8.0, 2.0, 1.0, 0.7, 0.3],
    2: [1.0, 3.0, 8.0, 20.0, 25.0, 22.0, 12.0, 5.0, 2.0, 1.5, 0.5],
    3: [0.5, 1.0, 3.0, 10.0, 18.0, 30.0, 22.0, 9.0, 3.0, 2.5, 1.0],
    4: [0.0, 0.0, 1.0, 3.0, 8.0, 20.0, 45.0, 13.0, 5.0, 3.0, 2.0],
    5: [0.0, 0.0, 0.5, 2.0, 5.0, 12.0, 30.0, 25.0, 15.0, 7.0, 3.5],
    6: [0.0, 0.0, 0.5, 1.0, 3.0, 8.0, 22.0, 28.0, 22.0, 10.0, 5.5],
    7: [0.0, 0.0, 0.0, 0.5, 1.0, 4.0, 12.0, 22.0, 28.0, 20.0, 12.5],
    8: [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 5.0, 12.0, 25.0, 28.0, 28.5],
    9: [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 10.0, 18.0, 25.0, 23.0, 20.0],
}
"""Mean grain-size profiles by soil type; row totals approximately 100 percent."""

TYPE_GROUND_PROBS = {
    1: 0.06, 2: 0.10, 3: 0.16, 4: 0.20, 5: 0.18, 6: 0.14, 7: 0.09, 8: 0.05, 9: 0.02,
}
"""Reference soil-type sampling probabilities, favoring sands and sandy loams."""

PLAXIS_CLASSES = ("very fine", "fine", "medium", "coarse", "very coarse")
"""PLAXIS grain-size classes based on median diameter D50."""


def _interp_dp(p_asc: np.ndarray, d_asc: np.ndarray, p_target: float) -> np.ndarray:
    """Find Dp by log interpolation of the percent-passing curve.

    :param p_asc: increasing passing percentages, shape (n, 11)
    :param d_asc: increasing diameters, shape (11,)
    :param p_target: target passing percentage
    :return: Dp in mm, shape (n,)
    """
    n = p_asc.shape[0]
    mask = p_asc >= p_target
    any_mask = mask.any(axis=1)
    idx = np.argmax(mask, axis=1) # First index with percent passing >= p_target.

    rows = np.arange(n)
    idx_safe = np.clip(idx, 1, p_asc.shape[1] - 1)
    p1 = p_asc[rows, idx_safe - 1]
    p2 = p_asc[rows, idx_safe]
    d1 = d_asc[idx_safe - 1]
    d2 = d_asc[idx_safe]

    denom = np.where(np.abs(p2 - p1) < 1e-12, 1.0, p2 - p1)
    t = (p_target - p1) / denom
    log_d = np.log(d1) + t * (np.log(d2) - np.log(d1))
    result = np.exp(log_d)

    # Boundary cases.
    result = np.where(idx == 0, d_asc[0], result) # Target at or below the minimum passing percentage.
    result = np.where(~any_mask, d_asc[-1], result) # Target above the maximum passing percentage.
    return result


def plaxis_classification(fractions: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute grain-size diameters and the PLAXIS class.

    Calculate D10, D50, D60 and Cu = D60/D10 from the percent-passing curve.
    Classify D50 in mm as >10 very coarse, 2-10 coarse, 0.25-2 medium,
    0.075-0.25 fine, and otherwise very fine.

    :param fractions: mass percentages, shape (n, 11), in FRACTION_KEYS order
    :return: arrays D10, D50, D60, Cu and string plaxis_class
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    total = fractions.sum(axis=1, keepdims=True)
    total = np.where(total < 1e-6, 1.0, total)
    fractions = fractions / total * 100.0

    diameters = np.array([FRACTION_BOUNDS[k] for k in FRACTION_KEYS], dtype=np.float64) # Decreasing order.
    passing = np.cumsum(fractions[:, ::-1], axis=1)[:, ::-1] # Percentage finer than diameters[i].

    d_asc = diameters[::-1].copy() # Increasing order, 0.002-20 mm.
    p_asc = np.clip(passing[:, ::-1], 0.0, 100.0)
    p_asc = np.maximum.accumulate(p_asc, axis=1) # Enforce nondecreasing passing percentages.

    d10 = _interp_dp(p_asc, d_asc, 10.0)
    d50 = _interp_dp(p_asc, d_asc, 50.0)
    d60 = _interp_dp(p_asc, d_asc, 60.0)
    cu = d60 / np.maximum(d10, 1e-12)

    plaxis_class = np.select(
        [d50 > 10.0, d50 > 2.0, d50 > 0.25, d50 > 0.075],
        ["very coarse", "coarse", "medium", "fine"],
        default="very fine",
    )
    return {
        "D10": d10.astype(np.float32), "D50": d50.astype(np.float32),
        "D60": d60.astype(np.float32), "Cu": cu.astype(np.float32),
        "plaxis_class": plaxis_class.astype(object),
    }


def define_type_ground(fractions: np.ndarray, Ip: np.ndarray, Ir: np.ndarray) -> np.ndarray:
    """Assign GOST soil types 1-9 from grain size, plasticity and organic content.

    Ir >= 50 identifies peat. Sands (Ip < 1) are split by cumulative coarse fractions;
    1 <= Ip <= 7 identifies sandy loam, 7 < Ip <= 17 loam, and Ip > 17 clay.

    :param fractions: mass percentages, shape (n, 11)
    :param Ip: plasticity index, percent
    :param Ir: organic content, percent
    :return: integer soil-type codes 1-9, shape (n,)
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    Ip = np.asarray(Ip, dtype=np.float64)
    Ir = np.asarray(Ir, dtype=np.float64)
    acc = np.cumsum(fractions, axis=1) # Accumulate from the coarse fractions.

    sand_type = np.select(
        [acc[:, 2] > 25.0, acc[:, 4] > 50.0, acc[:, 5] > 50.0, acc[:, 6] >= 75.0],
        [1, 2, 3, 4],
        default=5,
    )
    type_ground = np.select(
        [Ir >= 50.0, Ip < 1.0, Ip <= 7.0, Ip <= 17.0],
        [9, sand_type, 6, 7],
        default=8,
    )
    return type_ground.astype(int)


def sample_grain_size(type_ground_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample grain-size profiles for target soil types.

    Apply multiplicative log-normal noise to the reference profile and normalize
    each row to 100 percent.

    :param type_ground_target: target types 1-9, shape (n,)
    :param rng: numpy random generator
    :return: mass percentages, shape (n, 11), with row sums of 100
    """
    n = len(type_ground_target)
    archetypes = np.array([TYPE_GROUND_ARCHETYPES[int(t)] for t in type_ground_target], dtype=np.float64)
    noise = np.exp(rng.normal(0.0, 0.35, size=archetypes.shape))
    fractions = archetypes * noise
    fractions = np.clip(fractions, 0.0, None)
    totals = fractions.sum(axis=1, keepdims=True)
    totals = np.where(totals < 1e-6, 1.0, totals)
    return (fractions / totals * 100.0).astype(np.float32)
