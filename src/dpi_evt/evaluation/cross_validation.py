"""Fixed model settings used by the study."""
from typing import Dict

def publication_model_kwargs(name: str, model_kwargs: Dict, config) -> Dict:
    """Return the model keyword arguments fixed by the study protocol.

    Settings that define the estimand and the uncertainty treatment are fixed here and recorded
    in every cross-validation row. Fold-specific grid choices are applied on top of them.
    """
    out = dict(model_kwargs)
    if name in {"dpi_evt", "evt_ssm"}:
        out["max_cycle_reference"] = float(config.max_cycle_reference)
    if name == "dpi_evt":
        out.update({
            "probabilistic": True,
            "use_flow": False,
            "nliq_from_curve": True,
            "report_nliq_from_curve": True,
            "calibration_steps": 0,
            "liq_threshold": float(config.liq_threshold),
            "traj_residual_span": 0.10,
            "use_free_increment": False,
            "use_analytical_layer": True,
            "use_monotone_clip": True,
            "use_discriminative_risk": True,
            "use_censored_nliq": True,
            "crossing_margin_weight": 0.10,
            "kl_weight": 0.02,
        })
    elif name == "evt_ssm":
        out["liq_threshold"] = float(config.liq_threshold)
    return out
