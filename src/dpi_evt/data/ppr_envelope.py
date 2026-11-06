"""PPR(N) envelopes from per-cycle pressure peaks.

Extract one upper peak per cycle, following DigitRock define_max_rude. Isotonic
regression defines the nondecreasing target envelope, which is smoothed and
resampled with a validity mask. Monotonicity describes this envelope rather
than every pressure measurement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["extract_upper_envelope", "extract_cycle_amplitude", "monotone_smooth",
           "causal_monotone_smooth", "resample_to_grid", "smooth_ppr_trajectory"]


def extract_cycle_amplitude(
    cycles: np.ndarray,
    signal: np.ndarray,
    points_in_cycle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract single amplitude (max - min)/2 in each cycle.

    Used to recover CSR(N) from deviator stress, including variable-amplitude loading.

    :param cycles: cycle coordinate of each measurement, shape (m,)
    :param signal: deviator/stress signal, shape (m,)
    :param points_in_cycle: fixed samples per cycle; otherwise use integer cycle bins
    :return: (cycle_centers, amplitude), each shape (L,)
    """
    cyc = np.asarray(cycles, dtype=float)
    sig = np.nan_to_num(np.asarray(signal, dtype=float))
    size = min(cyc.size, sig.size)
    cyc, sig = cyc[:size], sig[:size]
    if cyc.size == 0:
        return np.zeros(0), np.zeros(0)
    n_total = max(int(np.floor(np.nanmax(cyc))), 1)

    if points_in_cycle and points_in_cycle >= 2 and cyc.size >= points_in_cycle:
        centers, amps = [], []
        for i in range(0, cyc.size, int(points_in_cycle)):
            w = sig[i:i + int(points_in_cycle)]
            if w.size == 0:
                continue
            centers.append(float(cyc[min(i + w.size // 2, cyc.size - 1)]))
            amps.append((float(np.nanmax(w)) - float(np.nanmin(w))) / 2.0)
        c = np.asarray(centers, float); a = np.asarray(amps, float)
        order = np.argsort(c)
        return c[order], a[order]

    centers = np.arange(1, n_total + 1, dtype=float)
    amps = np.zeros(n_total)
    for k in range(1, n_total + 1):
        m = (cyc >= k) & (cyc < k + 1)
        if int(m.sum()) >= 2:
            amps[k - 1] = (float(np.nanmax(sig[m])) - float(np.nanmin(sig[m]))) / 2.0
        elif k > 1:
            amps[k - 1] = amps[k - 2]
    return centers, amps


def extract_upper_envelope(
    cycles: np.ndarray,
    ppr: np.ndarray,
    points_in_cycle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one upper PPR peak per cycle.

    Use fixed-length windows when points_in_cycle is supplied, otherwise integer
    cycle bins. Empty cycles inherit the preceding peak.

    :param cycles: cycle coordinate of each measurement, shape (m,)
    :param ppr: measured pressure ratio, shape (m,)
    :param points_in_cycle: optional fixed window length
    :return: (cycle_peaks, ppr_peaks), each shape (L,)
    """
    cyc = np.asarray(cycles, dtype=float)
    pp = np.nan_to_num(np.asarray(ppr, dtype=float))
    size = min(cyc.size, pp.size)
    cyc, pp = cyc[:size], pp[:size]
    n_total = int(np.floor(np.nanmax(cyc))) if cyc.size else 0
    n_total = max(n_total, 1)

    if points_in_cycle and points_in_cycle >= 2 and cyc.size >= points_in_cycle:
        cyc_peaks, ppr_peaks = [], []
        for i in range(0, cyc.size, int(points_in_cycle)):
            window = pp[i:i + int(points_in_cycle)]
            if window.size == 0:
                continue
            j = int(np.argmax(window)) + i
            cyc_peaks.append(cyc[j])
            ppr_peaks.append(pp[j])
        cyc_peaks = np.asarray(cyc_peaks, dtype=float)
        ppr_peaks = np.asarray(ppr_peaks, dtype=float)
        order = np.argsort(cyc_peaks)
        return cyc_peaks[order], ppr_peaks[order]

    # Default: maximum PPR within each integer cycle.
    bins = np.clip(np.ceil(cyc).astype(int), 1, n_total) - 1
    peaks = np.zeros(n_total)
    np.maximum.at(peaks, bins, pp)
    for k in range(1, n_total): # Fill an empty cycle with the preceding peak.
        if peaks[k] == 0.0 and peaks[k - 1] > 0.0:
            peaks[k] = peaks[k - 1]
    cyc_peaks = np.arange(1, n_total + 1, dtype=float)
    return cyc_peaks, peaks


def _isotonic_nondecreasing(y: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """Nondecreasing isotonic regression (PAVA), clipped to [y_min, y_max]."""
    try:
        from sklearn.isotonic import IsotonicRegression

        x = np.arange(len(y), dtype=float)
        fitted = IsotonicRegression(y_min=y_min, y_max=y_max, increasing=True).fit_transform(x, y)
        return np.asarray(fitted, dtype=float)
    except Exception:
        # PAVA implementation when sklearn is unavailable.
        y = np.clip(np.asarray(y, dtype=float), y_min, y_max)
        return np.maximum.accumulate(y)


def monotone_smooth(
    ppr_peaks: np.ndarray,
    smoothing: float = 1.0,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Construct a smooth nondecreasing curve from per-cycle PPR peaks.

    Apply isotonic regression, Gaussian smoothing and a cumulative maximum.

    :param ppr_peaks: per-cycle peaks, shape (L,)
    :param smoothing: Gaussian width in cycles; zero disables smoothing
    :param clip: PPR bounds
    :return: smooth nondecreasing PPR, shape (L,)
    """
    y = np.asarray(ppr_peaks, dtype=float)
    if y.size <= 2:
        return np.clip(np.maximum.accumulate(np.nan_to_num(y)), clip[0], clip[1])

    iso = _isotonic_nondecreasing(np.nan_to_num(y), clip[0], clip[1])
    if smoothing and smoothing > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            iso = gaussian_filter1d(iso, sigma=float(smoothing), mode="nearest")
        except Exception:
            kernel = max(int(round(smoothing)) * 2 + 1, 3)
            pad = kernel // 2
            padded = np.pad(iso, pad, mode="edge")
            iso = np.convolve(padded, np.ones(kernel) / kernel, mode="valid")
        iso = np.maximum.accumulate(iso) # Restore monotonicity after smoothing.
    return np.clip(iso, clip[0], clip[1])


def causal_monotone_smooth(
    ppr_peaks: np.ndarray,
    window: int = 3,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """One-sided envelope using the current and preceding peaks.

    Take a rolling median over window values, followed by a cumulative maximum.
    The value at N is independent of measurements after N. Used for causal prefixes
    and diagnostics; the full target envelope is constructed separately.
    """
    y = np.clip(np.nan_to_num(np.asarray(ppr_peaks, dtype=float)), clip[0], clip[1])
    if y.size == 0:
        return y
    w = max(int(window), 1)
    filtered = np.empty_like(y)
    for i in range(y.size):
        filtered[i] = np.median(y[max(0, i - w + 1):i + 1])
    return np.clip(np.maximum.accumulate(filtered), clip[0], clip[1])


def resample_to_grid(
    cycle_peaks: np.ndarray,
    ppr_smooth: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a smoothed PPR curve to seq_len grid nodes.

    Short records occupy the first nodes with an invalid padded tail. Longer records
    are evenly resampled to seq_len nodes.

    :param cycle_peaks: peak cycle coordinates, shape (L,)
    :param ppr_smooth: smoothed PPR, shape (L,)
    :param seq_len: output grid length
    :return: (cycles_grid, ppr_grid, valid_mask), each shape (seq_len,)
    """
    L = len(ppr_smooth)
    grid = np.zeros(seq_len, dtype=np.float32)
    vals = np.zeros(seq_len, dtype=np.float32)
    mask = np.zeros(seq_len, dtype=np.float32)
    if L == 0:
        return grid, vals, mask
    if L <= seq_len:
        grid[:L] = cycle_peaks
        vals[:L] = ppr_smooth
        mask[:L] = 1.0
        grid[L:] = cycle_peaks[-1]
    else:
        idx = np.linspace(0, L - 1, seq_len)
        grid = (np.interp(idx, np.arange(L), cycle_peaks)).astype(np.float32)
        vals = (np.interp(idx, np.arange(L), ppr_smooth)).astype(np.float32)
        mask[:] = 1.0
    return grid, vals, mask


def landmark_aware_cycles(last_cycle: float, seq_len: int, landmark_cycles: float,
                          k_early: int) -> np.ndarray:
    """Construct a grid with common early nodes through the landmark.

    The first k_early nodes use geomspace(1, N0, k_early) for every record.
    The remaining nodes use geomspace(N0, last_cycle), without duplicating N0.

    :param last_cycle: endpoint for the later grid
    :param seq_len: grid length
    :param landmark_cycles: physical landmark N0
    :param k_early: common early-node count, usually prefix_len
    :return: increasing cycle grid, shape (seq_len,)
    """
    n0 = max(float(landmark_cycles), 1.001)
    k = int(min(max(k_early, 2), seq_len - 1))
    early = np.geomspace(1.0, n0, k)
    last = max(float(last_cycle), n0 * 1.01)
    late = np.geomspace(n0, last, seq_len - k + 1)[1:] # Omit the duplicate N0 node.
    grid = np.concatenate([early, late]).astype(np.float32)
    return np.maximum.accumulate(grid)


def forecasting_cycles(horizon: float, seq_len: int, n0: float, prefix_len: int,
                       query_prefix_horizon: Optional[float] = None) -> np.ndarray:
    """Query times are independent of specimen duration and, optionally, of N0.

    The final prefix experiment uses one grid with exact 3/5/10/20-cycle landmarks.
    A zero-cycle prefix has no observations; it never creates repeated zero query times.
    """
    if query_prefix_horizon is None:
        return landmark_aware_cycles(horizon, seq_len, max(n0, 1.001), prefix_len)
    if prefix_len != 12 or float(query_prefix_horizon) != 20.0:
        raise ValueError("The common final-run grid requires prefix_len=12, query_prefix_horizon=20")
    if horizon <= 20 or seq_len <= 12:
        raise ValueError("The common grid requires horizon>20 and seq_len>12")
    early = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], dtype=float)
    late = np.geomspace(20, horizon, seq_len - 12 + 1)[1:]
    return np.r_[early, late].astype(np.float32)


def smooth_ppr_trajectory(
    raw_cycles: np.ndarray,
    raw_ppr: np.ndarray,
    seq_len: int,
    points_in_cycle: Optional[int] = None,
    smoothing: float = 1.0,
    clip: Tuple[float, float] = (0.0, 1.0),
    return_peaks: bool = False,
):
    """Convert raw cyclic PPR measurements to a smooth curve on a seq_len grid.

    Combine extract_upper_envelope, monotone_smooth and resample_to_grid.

    :param raw_cycles: cycle coordinates of the measurements
    :param raw_ppr: raw measured pressure ratios
    :param seq_len: output grid length
    :param points_in_cycle: optional fixed peak-extraction window
    :param smoothing: Gaussian width in cycles
    :param clip: PPR bounds
    :param return_peaks: also return the raw cycle_peaks and ppr_peaks
    :return: (cycles_grid, ppr_grid, valid_mask), optionally followed by
        cycle_peaks and ppr_peaks
    """
    cycle_peaks, ppr_peaks = extract_upper_envelope(raw_cycles, raw_ppr, points_in_cycle)
    ppr_smooth = monotone_smooth(ppr_peaks, smoothing=smoothing, clip=clip)
    grid, vals, mask = resample_to_grid(cycle_peaks, ppr_smooth, seq_len)
    if return_peaks:
        return grid, vals, mask, cycle_peaks, ppr_peaks
    return grid, vals, mask
