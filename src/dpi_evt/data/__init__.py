"""Input preparation, causal prefixes and site-separated benchmark splits."""
from .io import load_population_artifact, save_population_artifact
from .ppr_envelope import extract_upper_envelope, monotone_smooth, smooth_ppr_trajectory
from .raw_loader import build_real_objects_population, find_cloud_root, DEFAULT_TEST_TYPES, read_statement, build_cohort_manifest
from .real_adapter import build_observed_prefix, build_population_from_experiments, compute_crr_features
from .splits import iterate_minibatches, make_benchmark_splits, prepare_benchmark_dataset, safe_strata, stratified_subset_indices
from .consistency import validate_forecasting_input_contract
