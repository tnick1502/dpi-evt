"""Soil and cyclic-loading categories with reference parameter ranges.

Soil parameters: e (void ratio), D_r (relative density, fraction), I_p
(plasticity index, percent), V_s (shear-wave velocity, m/s), xi (damping ratio,
fraction), sigma_eff (effective confinement, kPa), and log_perm (log10 permeability
in m/s). Loading parameters: CSR, frequency (Hz), amp_scale, N_max (observation
window in cycles), and nonstationarity.
"""

SOIL_CLASS_SPECS = {
    "loose_sand": {
        "e": (0.78, 1.05),
        "D_r": (0.25, 0.55),
        "I_p": (0.0, 3.0),
        "V_s": (110.0, 180.0),
        "xi": (0.02, 0.05),
        "sigma_eff": (70.0, 180.0),
        "log_perm": (-4.4, -3.2),
    },
    "silty_sand": {
        "e": (0.70, 0.98),
        "D_r": (0.35, 0.65),
        "I_p": (1.0, 6.0),
        "V_s": (125.0, 210.0),
        "xi": (0.025, 0.06),
        "sigma_eff": (80.0, 210.0),
        "log_perm": (-5.2, -3.8),
    },
    "low_plastic_silt": {
        "e": (0.62, 0.92),
        "D_r": (0.30, 0.60),
        "I_p": (4.0, 10.0),
        "V_s": (135.0, 225.0),
        "xi": (0.03, 0.07),
        "sigma_eff": (90.0, 240.0),
        "log_perm": (-6.0, -4.4),
    },
    "clayey_silt": {
        "e": (0.58, 0.88),
        "D_r": (0.28, 0.52),
        "I_p": (8.0, 20.0),
        "V_s": (145.0, 250.0),
        "xi": (0.04, 0.10),
        "sigma_eff": (95.0, 260.0),
        "log_perm": (-6.8, -5.2),
    },
    "dense_sand": {
        "e": (0.42, 0.72),
        "D_r": (0.65, 0.95),
        "I_p": (0.0, 2.0),
        "V_s": (210.0, 360.0),
        "xi": (0.015, 0.04),
        "sigma_eff": (120.0, 320.0),
        "log_perm": (-4.0, -2.8),
    },
}
"""Reference property ranges by soil class."""

LOAD_MODE_SPECS = {
    "storm": {
        "CSR": (0.10, 0.24),
        "frequency": (0.05, 0.35),
        "amp_scale": (0.85, 1.25),
        "N_max": (700.0, 1_500.0),
        "nonstationarity": (0.20, 0.60),
    },
    "seismic": {
        "CSR": (0.18, 0.48),
        "frequency": (0.8, 3.5),
        "amp_scale": (0.90, 1.35),
        "N_max": (60.0, 500.0),
        "nonstationarity": (0.45, 0.95),
    },
    "technogenic": {
        "CSR": (0.06, 0.18),
        "frequency": (6.0, 20.0),
        "amp_scale": (0.90, 1.10),
        "N_max": (250.0, 1_200.0),
        "nonstationarity": (0.08, 0.25),
    },
    "stationary_cyclic": {
        "CSR": (0.08, 0.22),
        "frequency": (0.8, 5.0),
        "amp_scale": (0.95, 1.10),
        "N_max": (350.0, 1_500.0),
        "nonstationarity": (0.05, 0.18),
    },
    "variable_amplitude": {
        "CSR": (0.10, 0.30),
        "frequency": (0.2, 10.0),
        "amp_scale": (0.85, 1.35),
        "N_max": (180.0, 1_500.0),
        "nonstationarity": (0.35, 0.85),
    },
}
"""Reference parameter ranges by cyclic-loading mode."""

# GOST soil types: keys follow type_ground codes 1-9.
SOIL_NAMES = [
    "gravelly_sand", "coarse_sand", "medium_sand", "fine_sand", "silty_sand",
    "sandy_loam", "loam", "clay", "peat",
]
"""Ordered soil identifiers corresponding to GOST codes 1-9."""

LOAD_NAMES = list(LOAD_MODE_SPECS.keys())
"""Ordered cyclic-loading mode identifiers."""

SOIL_DISPLAY_NAMES = {
    "gravelly_sand": "песок гравелистый",
    "coarse_sand": "песок крупный",
    "medium_sand": "песок средней крупности",
    "fine_sand": "песок мелкий",
    "silty_sand": "песок пылеватый",
    "sandy_loam": "супесь",
    "loam": "суглинок",
    "clay": "глина",
    "peat": "торф",
}
"""Russian soil-type display labels."""

LOAD_DISPLAY_NAMES = {
    "storm": "штормовой",
    "seismic": "сейсмический",
    "technogenic": "техногенный",
    "stationary_cyclic": "стационарно-циклический",
    "variable_amplitude": "переменная амплитуда",
}
"""Russian loading-mode display labels."""

GENERATOR_FAMILY_DISPLAY_NAMES = {
    "hyperbolic": "гиперболическое",
    "power": "степенное",
    "exponential": "экспоненциальное",
    "logarithmic": "логарифмическое",
}
"""Russian CRR-family display labels."""

SOIL_DISPLAY_NAMES_EN = {
    "gravelly_sand": "Gravelly sand",
    "coarse_sand": "Coarse sand",
    "medium_sand": "Medium sand",
    "fine_sand": "Fine sand",
    "silty_sand": "Silty sand",
    "sandy_loam": "Sandy loam",
    "loam": "Loam",
    "clay": "Clay",
    "peat": "Peat",
}
"""English soil-type display labels."""

RESPONSE_TYPE_DISPLAY_NAMES = {
    "contractive": "контрактантный",
    "transitional": "переходный",
    "dilative": "дилатантный",
    "plastic": "пластичный",
}
"""Russian cyclic-response display labels."""

RESPONSE_TYPE_DISPLAY_NAMES_EN = {
    "contractive": "Contractive",
    "transitional": "Transitional",
    "dilative": "Dilative",
    "plastic": "Plastic",
}
"""English cyclic-response display labels."""

LOAD_DISPLAY_NAMES_EN = {
    "storm": "Storm",
    "seismic": "Seismic",
    "technogenic": "Technogenic",
    "stationary_cyclic": "Stationary cyclic",
    "variable_amplitude": "Variable amplitude",
}
"""English loading-mode display labels."""

GENERATOR_FAMILY_DISPLAY_NAMES_EN = {
    "hyperbolic": "Hyperbolic",
    "power": "Power",
    "exponential": "Exponential",
    "logarithmic": "Logarithmic",
}
"""English CRR-family display labels."""

# English axis labels with units.
FEATURE_AXIS_LABELS_EN = {
    "e": "Void ratio, e (–)",
    "D_r": "Relative density, D_r (–)",
    "I_p": "Plasticity index, I_p (%)",
    "V_s": "Shear-wave velocity, V_s (m/s)",
    "xi": "Damping ratio, ξ (–)",
    "sigma_eff": "Effective stress, σ′ (kPa)",
    "log10_permeability": "log₁₀ permeability (m/s)",
    "permeability": "Permeability (m/s)",
    "CSR_base": "Cyclic stress ratio, CSR (–)",
    "CSR_max": "Peak CSR (–)",
    "frequency": "Loading frequency (Hz)",
    "amp_scale": "Amplitude scale (–)",
    "N_max": "Loading horizon, N_max (cycles)",
    "nonstationarity": "Non-stationarity (–)",
    "N_liq_true": "Cycles to liquefaction, N_liq (cycles)",
    "PPR_max_true": "Peak pore-pressure ratio (–)",
    "damage_max_true": "Peak damage state, z (–)",
    "risk_score_true": "Liquefaction risk (–)",
    "uncertainty_proxy": "Uncertainty proxy (–)",
    # Additional geotechnical properties.
    "rs": "Particle density, ρs (g/cm³)",
    "r": "Bulk density, ρ (g/cm³)",
    "rd": "Dry density, ρd (g/cm³)",
    "n_porosity": "Porosity, n (%)",
    "W": "Water content, W (%)",
    "Wl": "Liquid limit, WL (%)",
    "Wp": "Plastic limit, WP (%)",
    "Il": "Liquidity index, IL (–)",
    "Ir": "Organic content, Ir (%)",
    "fines_content": "Fines content (<0.075 mm), FC (%)",
    "clay_fraction": "Clay fraction (<0.002 mm) (%)",
    "D10": "Effective grain size, D10 (mm)",
    "D50": "Median grain size, D50 (mm)",
    "D60": "Grain size D60 (mm)",
    "Cu": "Uniformity coefficient, Cu (–)",
    "OCR": "Overconsolidation ratio, OCR (–)",
    "K0": "Earth-pressure coefficient, K0 (–)",
    "Vs1": "Corrected shear-wave velocity, Vs1 (m/s)",
    "static_shear_ratio": "Static shear ratio, α_static (–)",
    "cementation_index": "Cementation index (–)",
    "depth": "Sampling depth (m)",
    "phi": "Friction angle, φ (deg)",
    "cohesion": "Cohesion, c (kPa)",
    "E_modulus": "Deformation modulus, E (MPa)",
    "crr_ref": "CRR at 15 cycles, CRR15 (–)",
    "crr_alpha": "CRR curve parameter α (–)",
    "crr_betta": "CRR curve parameter β (–)",
    "crr_cycle_slope": "Cycle-degradation slope, s (–)",
}
"""English axis labels with units."""

__all__ = [
    "SOIL_CLASS_SPECS",
    "LOAD_MODE_SPECS",
    "SOIL_NAMES",
    "LOAD_NAMES",
    "SOIL_DISPLAY_NAMES",
    "LOAD_DISPLAY_NAMES",
    "GENERATOR_FAMILY_DISPLAY_NAMES",
    "SOIL_DISPLAY_NAMES_EN",
    "LOAD_DISPLAY_NAMES_EN",
    "GENERATOR_FAMILY_DISPLAY_NAMES_EN",
    "RESPONSE_TYPE_DISPLAY_NAMES",
    "RESPONSE_TYPE_DISPLAY_NAMES_EN",
    "FEATURE_AXIS_LABELS_EN",
]
