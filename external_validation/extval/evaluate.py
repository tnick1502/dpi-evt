"""Non-learned prefix references used in the study."""
import numpy as np

def _prefix(split):
    pm = split["prefix_mask"].numpy() > 0
    return split["cycles"].numpy(), split["prefix_obs"].numpy(), pm

def baseline_persistence(split):
    """Keep ru at the last observed prefix value, a lower bound for a nondecreasing forecast."""
    c, po, pm = _prefix(split)
    last = np.array([po[i][pm[i]][-1] if pm[i].any() else 0.0 for i in range(len(c))])
    return np.repeat(last[:, None], c.shape[1], 1)

def baseline_loglinear(split):
    """Fit ru = a + b ln N on the prefix (N ≤ N0), extrapolate and clip to [0, 1]."""
    c, po, pm = _prefix(split)
    out = np.zeros_like(c)
    for i in range(len(c)):
        x, y = np.log(c[i][pm[i]]), po[i][pm[i]]
        b, a = np.polyfit(x, y, 1) if len(x) >= 2 else (0.0, y.mean() if len(y) else 0.0)
        out[i] = np.clip(a + max(b, 0.0) * np.log(c[i]), 0, 1)
    return np.maximum.accumulate(out, axis=1)

def baseline_seed_arcsine(split, theta=0.7):
    """Seed, Martin & Lysmer (1975)/Booker et al. (1976): ru = 2/π·arcsin((N/N_L)^(1/(2θ))), θ=0.7.
    The single parameter N_L is fitted by least squares to the causal prefix (N ≤ N0).
    The reference uses no training data."""
    c, po, pm = _prefix(split)
    out = np.zeros_like(c)
    grid = np.logspace(0, 5, 400)
    for i in range(len(c)):
        x, y = c[i][pm[i]], po[i][pm[i]]
        best, bnl = 1e9, grid[-1]
        for nl in grid:
            pred = 2 / np.pi * np.arcsin(np.clip(x / nl, 0, 1) ** (1 / (2 * theta)))
            err = np.mean((pred - y) ** 2)
            if err < best:
                best, bnl = err, nl
        out[i] = 2 / np.pi * np.arcsin(np.clip(c[i] / bnl, 0, 1) ** (1 / (2 * theta)))
    return out

REFERENCE = {"Persistence": baseline_persistence, "Log-linear prefix": baseline_loglinear,
             "Seed–Booker arcsine (prefix-fit)": baseline_seed_arcsine}
