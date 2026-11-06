"""Empirical cyclic-resistance relations.

Hyperbolic, power, exponential and logarithmic families relate CSR and cycle
count at the triggering boundary. Functions accept scalars or arrays and retain
the input form; inverse N(CSR) functions are provided where analytic.
"""

from typing import List, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...], float, int]


def _to_numpy_1d(values: ArrayLike) -> np.ndarray:
    """:param values: scalar, list, tuple or numpy array
    :return: one-dimensional numpy.float64 array
    """
    arr: np.ndarray = np.asarray(values, dtype=np.float64)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    return arr


def _return_scalar_if_needed(
    original_values: ArrayLike,
    result: np.ndarray,
) -> Union[np.ndarray, float]:
    """Restore scalar output when the input was scalar.

    :param original_values: original input
    :param result: computed numpy array
    :return: float for scalar input, otherwise numpy array
    """
    if np.isscalar(original_values):
        return float(result[0])

    return result


def _validate_positive_cycles(n_cycles: ArrayLike) -> np.ndarray:
    """Validate strictly positive cycles for logarithms and power transformations.

    :param n_cycles: loading cycle count
    :return: one-dimensional cycle array
    """
    n_arr: np.ndarray = _to_numpy_1d(n_cycles)

    if np.any(n_arr <= 0.0):
        raise ValueError("All n_cycles values must be > 0.")

    return n_arr


def bilge_exponential_n_from_csr(
    csr: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """H. Bilge et al. exponential relation: N = alpha * exp(beta * CSR).

    :param csr: cyclic stress ratio
    :param alpha: scale coefficient
    :param beta: exponential coefficient
    :return: cycle count N
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")

    csr_arr: np.ndarray = _to_numpy_1d(csr)
    result: np.ndarray = alpha * np.exp(beta * csr_arr)

    return _return_scalar_if_needed(csr, result)


def bilge_exponential_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """H. Bilge et al. inverse relation: CSR = ln(N / alpha) / beta.

    :param n_cycles: cycle count N
    :param alpha: scale coefficient
    :param beta: exponential coefficient
    :return: CSR
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if beta == 0.0:
        raise ValueError("beta must be nonzero.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = np.log(n_arr / alpha) / beta

    return _return_scalar_if_needed(n_cycles, result)


def lentini_logarithmic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """V. Lentini et al. relation: CSR = alpha * log(N) + beta.

    Use the natural logarithm numpy.log.

    :param n_cycles: cycle count N
    :param alpha: log-cycle coefficient
    :param beta: intercept
    :return: CSR
    """
    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = alpha * np.log(n_arr) + beta

    return _return_scalar_if_needed(n_cycles, result)


def guoxing_power_n_from_csr(
    csr: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """C. Guoxing et al. power relation: N = alpha * CSR^beta.

    :param csr: cyclic stress ratio
    :param alpha: scale coefficient
    :param beta: power exponent
    :return: cycle count N
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")

    csr_arr: np.ndarray = _to_numpy_1d(csr)

    if np.any(csr_arr < 0.0):
        raise ValueError("CSR values must be >= 0 for the power relation.")

    result: np.ndarray = alpha * np.power(csr_arr, beta)

    return _return_scalar_if_needed(csr, result)


def guoxing_power_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """C. Guoxing et al. inverse relation: CSR = (N / alpha)^(1 / beta).

    :param n_cycles: cycle count N
    :param alpha: scale coefficient
    :param beta: power exponent
    :return: CSR
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0.")
    if beta == 0.0:
        raise ValueError("beta must be nonzero.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = np.power(n_arr / alpha, 1.0 / beta)

    return _return_scalar_if_needed(n_cycles, result)


def meziane_logarithmic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """E. Meziane et al. logarithmic relation: CSR = (alpha - ln(N)) / beta.

    :param n_cycles: cycle count N
    :param alpha: logarithmic intercept parameter
    :param beta: scale parameter
    :return: CSR
    """
    if beta == 0.0:
        raise ValueError("beta must be nonzero.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = (alpha - np.log(n_arr)) / beta

    return _return_scalar_if_needed(n_cycles, result)


def author_hyperbolic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """Author hyperbolic relation: CSR = beta / N^(1 - alpha).

    Equivalently, CSR = beta * N^(alpha - 1).

    :param n_cycles: cycle count N
    :param alpha: power-shape parameter
    :param beta: scale parameter
    :return: CSR
    """
    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = beta / np.power(n_arr, 1.0 - alpha)

    return _return_scalar_if_needed(n_cycles, result)
