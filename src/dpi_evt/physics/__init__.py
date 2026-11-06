"""Empirical and semi-empirical CRR and PPR functions used as analytical references."""

from dpi_evt.physics.crr_physical import compute_crr_components, crr_curve
from dpi_evt.physics.crr_models import (
    author_hyperbolic_csr,
    bilge_exponential_csr,
    bilge_exponential_n_from_csr,
    guoxing_power_csr,
    guoxing_power_n_from_csr,
    lentini_logarithmic_csr,
    meziane_logarithmic_csr,
)
from dpi_evt.physics.ppr_models import (
    compute_ppr,
    cpt_pore_pressure_ma_wang,
    cpt_ppr_ma_wang,
    extended_cpt_ppr,
    logarithmic_model_peak_cycle,
    logarithmic_model_scale_factor,
    logarithmic_ppr_normalized,
    logarithmic_ppr_raw,
)

__all__ = [
    "compute_crr_components",
    "crr_curve",
    "author_hyperbolic_csr",
    "bilge_exponential_csr",
    "bilge_exponential_n_from_csr",
    "guoxing_power_csr",
    "guoxing_power_n_from_csr",
    "lentini_logarithmic_csr",
    "meziane_logarithmic_csr",
    "compute_ppr",
    "cpt_pore_pressure_ma_wang",
    "cpt_ppr_ma_wang",
    "extended_cpt_ppr",
    "logarithmic_model_peak_cycle",
    "logarithmic_model_scale_factor",
    "logarithmic_ppr_normalized",
    "logarithmic_ppr_raw",
]
