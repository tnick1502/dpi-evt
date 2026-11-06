"""DPI-EVT: prefix-conditioned pressure accumulation and threshold forecasting."""
from .accel import configure_performance, describe_device, resolve_device
from .config import ExperimentConfig, get_default_config, set_global_seed
from .data import load_population_artifact, prepare_benchmark_dataset, save_population_artifact
from .training import train_model
