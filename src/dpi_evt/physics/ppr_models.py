"""Pore-pressure growth and dissipation functions.

Includes the Ma-Wang exponential/hyperbolic CPT-inspired relation, its logarithmic
tail extension and a logarithmic growth/dissipation relation with an analytic
extremum. Functions accept scalars or arrays and provide PPR, pressure, peak cycle
or scale factors as appropriate.
"""

from typing import Optional, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple, float, int]


def _to_numpy_1d(values: ArrayLike) -> np.ndarray:
    """:param values: scalar, list, tuple or numpy array
    :return: one-dimensional numpy.float64 array
    """
    arr: np.ndarray = np.asarray(values, dtype=np.float64)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    return arr


def _return_scalar_if_needed(original_values: ArrayLike, result: np.ndarray) -> Union[np.ndarray, float]:
    """Restore scalar output when the input was scalar.

    :param original_values: original input
    :param result: computed numpy array
    :return: float for scalar input, otherwise numpy array
    """
    if np.isscalar(original_values):
        return float(result[0])

    return result


def compute_ppr(
    u: ArrayLike,
    sigma_3: float,
) -> Union[np.ndarray, float]:
    """Calculate PPR = u / sigma_3.

    :param u: pore pressure
    :param sigma_3: confining stress
    :return: pressure ratio
    """
    if sigma_3 == 0:
        raise ValueError("sigma_3 must be nonzero.")

    u_arr: np.ndarray = _to_numpy_1d(u)
    ppr: np.ndarray = u_arr / float(sigma_3)

    return _return_scalar_if_needed(u, ppr)


def cpt_pore_pressure_ma_wang(
    n_cycles: ArrayLike,
    a: float,
    k: float,
    A: float,
) -> Union[np.ndarray, float]:
    """Ma-Wang CPT-inspired pore-pressure relation.

    u(N) = a*A*(1-exp(-k*N/A)) + (1-a)*N/(1/k+N/A).
    The first term is exponential growth and the second hyperbolic saturation.

    :param n_cycles: loading cycle count N
    :param a: exponential weight, usually in [0, 1]
    :param k: initial-growth rate parameter
    :param A: limiting-level parameter
    :return: pore pressure u(N)
    """
    if A <= 0:
        raise ValueError("A must be > 0.")
    if k <= 0:
        raise ValueError("k must be > 0.")

    n: np.ndarray = _to_numpy_1d(n_cycles)

    exp_term: np.ndarray = a * A * (1.0 - np.exp(-k * n / A))
    hyp_term: np.ndarray = (1.0 - a) * n / ((1.0 / k) + (n / A))

    u: np.ndarray = exp_term + hyp_term
    return _return_scalar_if_needed(n_cycles, u)


def cpt_ppr_ma_wang(
    n_cycles: ArrayLike,
    a: float,
    k: float,
    A: float,
    sigma_3: Optional[float] = None,
) -> Union[np.ndarray, float]:
    """Calculate PPR from the Ma-Wang relation.

    With sigma_3, divide computed u(N) by confinement. Without it, treat the model
    output as already normalized.

    :param n_cycles: cycle count N
    :param a: exponential weight
    :param k: growth-rate parameter
    :param A: limiting-level parameter
    :param sigma_3: optional confining stress
    :return: PPR(N)
    """
    u: Union[np.ndarray, float] = cpt_pore_pressure_ma_wang(
        n_cycles=n_cycles,
        a=a,
        k=k,
        A=A,
    )

    if sigma_3 is None:
        return u

    return compute_ppr(u=u, sigma_3=sigma_3)


def extended_cpt_ppr(
    n_cycles: ArrayLike,
    a: float,
    b: float,
    k: float,
    A: float,
    n_max: float,
) -> Union[np.ndarray, float]:
    """Extended CPT-inspired pressure-ratio relation.

    f1(N) = a*A*(1-exp(-k*N/A)) + b*N/(1/k+N/A)
            + (1-a-b)*A*ln(k*N+1)/ln(k*N_max+1).
    The stated constraints are a+b<=1 and a,b in [0, 1].

    :param n_cycles: cycle count N
    :param a: exponential-term weight
    :param b: hyperbolic-term weight
    :param k: pressure-growth rate
    :param A: level parameter
    :param n_max: cycle range used to normalize the logarithmic term
    :return: PPR(N)
    """
    if A <= 0:
        raise ValueError("A must be > 0.")
    if k <= 0:
        raise ValueError("k must be > 0.")
    if n_max <= 0:
        raise ValueError("n_max must be > 0.")
    if not (0.0 <= a <= 1.0):
        raise ValueError("a must lie in [0, 1].")
    if not (0.0 <= b <= 1.0):
        raise ValueError("b must lie in [0, 1].")
    if a + b > 1.0:
        raise ValueError("a + b must not exceed 1.")

    n: np.ndarray = _to_numpy_1d(n_cycles)

    exp_term: np.ndarray = a * A * (1.0 - np.exp(-k * n / A))
    hyp_term: np.ndarray = b * n / ((1.0 / k) + (n / A))

    log_denominator: float = np.log(k * n_max + 1.0)
    if np.isclose(log_denominator, 0.0):
        raise ValueError("ln(k * n_max + 1) must be nonzero.")

    log_term: np.ndarray = (1.0 - a - b) * A * np.log(k * n + 1.0) / log_denominator

    f1: np.ndarray = exp_term + hyp_term + log_term
    return _return_scalar_if_needed(n_cycles, f1)


def logarithmic_ppr_raw(
    n_cycles: ArrayLike,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> Union[np.ndarray, float]:
    """Evaluate unscaled logarithmic growth/dissipation.

    g(N) = A1*ln(k1*N+1) - A2*ln(k2*N+1), before multiplication by C.

    :param n_cycles: cycle count N
    :param A1: growth-term coefficient
    :param A2: dissipation-term coefficient
    :param k1: first logarithmic rate
    :param k2: second logarithmic rate
    :return: g(N)
    """
    if k1 <= 0:
        raise ValueError("k1 must be > 0.")
    if k2 <= 0:
        raise ValueError("k2 must be > 0.")

    n: np.ndarray = _to_numpy_1d(n_cycles)
    g: np.ndarray = A1 * np.log(k1 * n + 1.0) - A2 * np.log(k2 * n + 1.0)

    return _return_scalar_if_needed(n_cycles, g)


def logarithmic_model_peak_cycle(
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> float:
    """Find the extremum of the unscaled logarithmic curve.

    Setting d/dN[A1*ln(k1*N+1)-A2*ln(k2*N+1)]=0 gives
    N_peak = (A2*k2-A1*k1)/(k1*k2*(A1-A2)).

    :param A1: first logarithmic coefficient
    :param A2: second logarithmic coefficient
    :param k1: first rate
    :param k2: second rate
    :return: N_peak
    """
    denominator: float = k1 * k2 * (A1 - A2)

    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot compute N_peak: zero denominator.")

    n_peak: float = (A2 * k2 - A1 * k1) / denominator
    return float(n_peak)


def logarithmic_model_scale_factor(
    A: float,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> float:
    """Choose scale C so the logarithmic curve reaches A at its extremum.

    For g(N)=A1*ln(k1*N+1)-A2*ln(k2*N+1), calculate N_peak and C=A/g(N_peak).

    :param A: target extremum level
    :param A1: first logarithmic coefficient
    :param A2: second logarithmic coefficient
    :param k1: first rate
    :param k2: second rate
    :return: scale C
    """
    n_peak: float = logarithmic_model_peak_cycle(
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    if n_peak <= -1.0 / max(k1, k2):
        raise ValueError("Invalid N_peak: logarithm argument outside its domain.")

    g_peak: float = float(
        logarithmic_ppr_raw(
            n_cycles=n_peak,
            A1=A1,
            A2=A2,
            k1=k1,
            k2=k2,
        )
    )

    if np.isclose(g_peak, 0.0):
        raise ValueError("Cannot compute C: the base function is zero at the extremum.")

    C: float = A / g_peak
    return float(C)


def logarithmic_ppr_normalized(
    n_cycles: ArrayLike,
    A: float,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> Union[np.ndarray, float]:
    """Evaluate normalized logarithmic PPR.

    f2(N) = [A1*ln(k1*N+1)-A2*ln(k2*N+1)]*C,
    with C chosen to place the extremum at level A.

    :param n_cycles: cycle count N
    :param A: target extremum level
    :param A1: first logarithmic coefficient
    :param A2: second logarithmic coefficient
    :param k1: first rate
    :param k2: second rate
    :return: PPR(N)
    """
    C: float = logarithmic_model_scale_factor(
        A=A,
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    raw: Union[np.ndarray, float] = logarithmic_ppr_raw(
        n_cycles=n_cycles,
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    raw_arr: np.ndarray = _to_numpy_1d(raw)
    f2: np.ndarray = raw_arr * C

    return _return_scalar_if_needed(n_cycles, f2)
