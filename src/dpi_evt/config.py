"""Experiment settings for splitting, training, evaluation and reproducibility.

Also provides random seeds and plot colors.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ExperimentConfig",
    "get_default_config",
    "set_global_seed",
    "DEMO_PALETTE",
    "LIQ_THRESHOLD",
]

# Shared ru=PPR threshold for labels, N_liq, observed g_obs
# and model crossings; all components use LIQ_THRESHOLD.
LIQ_THRESHOLD: float = 0.95


@dataclass
class ExperimentConfig:
    """Settings for data preparation, splitting, training and evaluation.

    :param seed: seed for numpy, torch and random
    :param n_scenarios: scenario count for the optional population configuration
    :param benchmark_subset: stratified benchmark subset size
    :param ablation_subset: subset size for short ablation runs
    :param seq_len: number of cycle-grid nodes
    :param prefix_len: observed PPR prefix length in grid nodes
    :param benchmark_train_fraction: benchmark training fraction
    :param benchmark_val_fraction: benchmark validation fraction
    :param batch_size: training mini-batch size
    :param baseline_epochs: epoch count for baseline smoke runs
    :param physics_epochs: epoch count for structured-model smoke runs
    :param ablation_epochs: epoch count for ablation and OOD runs
    :param learning_rate: AdamW learning rate
    :param weight_decay: AdamW weight decay
    :param mc_samples_eval: Monte Carlo sample count for prediction
    :param export_figures: whether to save figures
    :param figure_dir: figure output directory
    :param max_csr_clip: upper CSR clipping value
    :param max_cycle_reference: common horizon for log normalization and censoring
    :param risk_threshold: default event-probability classification threshold
    :param measured_crr_fraction: CRR supervision fraction in the optional population configuration
    :param dataset_source: data mode identifier, synthetic or real_objects
    :param liq_threshold: shared ru=PPR threshold for the event label, N_liq, g_obs and model crossings
    :param min_nonliq_complete_cycles: minimum record duration for the optional terminal-state rule
    :param publication_baseline_epochs: baseline epoch ceiling in the main experiment
    :param publication_physics_epochs: structured-model epoch ceiling in the main experiment
    :param early_stopping_patience: epochs without validation improvement; None/0 disables stopping
    :param early_stopping_min_delta: minimum validation improvement
    :param use_observed_aux_loss: enable auxiliary targets derived from the measured PPR curve;
        False selects the no-derived-threshold-auxiliary stress test
    """

    seed: int = 42
    n_scenarios: int = 24_000
    benchmark_subset: int = 8_000
    ablation_subset: int = 4_000
    seq_len: int = 72
    prefix_len: int = 12
    benchmark_train_fraction: float = 0.70
    benchmark_val_fraction: float = 0.15
    batch_size: int = 256
    baseline_epochs: int = 4 # Short smoke run.
    physics_epochs: int = 6 # Short smoke run.
    # Main-run epoch ceilings and validation-based early stopping.
    publication_baseline_epochs: int = 120 # Epoch ceiling; early stopping applies.
    publication_physics_epochs: int = 200 # Epoch ceiling; early stopping applies.
    grid_search_epochs: int = 50 # Same convergence procedure with a smaller epoch ceiling.
    early_stopping_patience: int = 25 # Epochs without validation improvement.
    early_stopping_min_delta: float = 1e-4
    ablation_epochs: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    mc_samples_eval: int = 8
    export_figures: bool = False
    figure_dir: str = "reports/liquefaction_demo_figures"
    max_csr_clip: float = 0.65
    max_cycle_reference: float = 3_000.0
    risk_threshold: float = 0.5
    measured_crr_fraction: float = 0.25
    dataset_source: str = "synthetic"
    liq_threshold: float = LIQ_THRESHOLD # Shared ru=PPR event threshold; see LIQ_THRESHOLD.
    min_nonliq_complete_cycles: float = 500.0
    use_observed_aux_loss: bool = True
    group_split_by_object: bool = True # Split by object/site: an object belongs to only one
    # of train, validation or test.
    # The preonset mode truncates the prefix before threshold crossing.
    # See data.real_adapter.strict_pre_onset_prefix_mask.
    prefix_strict_preonset: bool = True # Truncate before onset.
    prefix_onset_threshold: float = LIQ_THRESHOLD # Onset threshold shared with the event definition.
    prefix_onset_margin: int = 1 # Extra gap: last prefix index < onset_idx - margin.
    prefix_min_len: int = 3 # Minimum length, subject to ending before onset.
    # Confirm onset with a full onset_sustain_cycles window
    # of consecutive integer cycles above the threshold.
    onset_sustain_cycles: int = 3
    # A terminal crossing without the full sustained window is unconfirmed.
    # This flag excludes such records from the labeled cohort;
    # onset_terminal_ambiguous remains in the metadata.
    exclude_terminal_ambiguous: bool = True
    # Additional PPR interval-width multiplier; default 1.0.
    # Select a different value only within outer-train,
    # without using coverage on test sites.
    calibration_shift_inflation: float = 1.0
    # landmark: observations through physical cycle N0;
    # fixed_k: first K grid nodes; preonset: truncation before onset.
    # The last mode makes window length outcome-dependent.
    prefix_mode: str = "landmark" # "landmark" | "fixed_k" | "preonset"
    prefix_fixed_k: int = 6 # Prefix length in grid nodes for fixed_k mode.
    prefix_landmark_cycles: float = 10.0 # Physical landmark N0 in cycles for landmark mode.
    # Optional common query grid for a controlled prefix sweep; observations still stop at N0.
    query_prefix_horizon: float | None = None
    # Final cohort counts by N0 are recorded in
    # results/tables/cohort_counts.csv.


def get_default_config() -> ExperimentConfig:
    """Return the default experiment configuration.

    :return: a new ExperimentConfig instance
    """
    return ExperimentConfig()


def set_global_seed(seed: int) -> None:
    """Seed random, numpy and torch.

    Configure CUDA and cuDNN for reproducibility. Agreement across devices and library
    versions still depends on the operations used.

    :param seed: random seed
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # CPU reproducibility: one thread and deterministic torch algorithms
    # reduce numerical drift between repeated evaluations.
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


DEMO_PALETTE = {
    "primary": "#0b6efd",
    "secondary": "#6610f2",
    "accent": "#d63384",
    "success": "#198754",
    "warning": "#fd7e14",
    "danger": "#dc3545",
    "dark": "#1f2937",
    "sand": "#c99a3d",
    "silt": "#8b9dc3",
}
"""Shared plot-color palette."""
