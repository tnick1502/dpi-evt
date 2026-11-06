# Notebook guide

The 11 notebooks are saved with the outputs of a complete run. By default, they read
the included aggregate tables and public external extracts. Training cells run only
when their flags are enabled and the required local inputs are available.

| Notebook | Purpose | Additional inputs for a new numerical run |
|---|---|---|
| `1_data_analysis/1_1_data_preparation.ipynb` | Cohorts, eligibility and causal preparation | Authorized laboratory archive for preparation |
| `2_model_training/2_1_baseline_models.ipynb` | Fixed baseline recipe and recorded learning histories | Prepared internal data for refitting |
| `2_model_training/2_2_evt_neural_ssm.ipynb` | Structured state-space comparator | Prepared internal data for refitting |
| `2_model_training/2_3_dpi_evt.ipynb` | DPI-EVT and its anchored variant | Prepared internal data for refitting |
| `3_evaluations/3_1_full_evaluation.ipynb` | Main joint forecast comparison | Internal OOF predictions to recompute metrics |
| `3_evaluations/3_2_prefix_information.ipynb` | Fixed-cohort prefix experiment | Fits and OOF predictions for each landmark |
| `3_evaluations/3_3_onset_interval_calibration.ipynb` | Native and residual interval diagnostics | Fitted models and internal validation events |
| `3_evaluations/3_4_matched_controls.ipynb` | Matched supervision, anchoring and drift ablation | Prepared internal data and completed main run |
| `4_external_validation/4_1_external_data.ipynb` | Read public data and reproduce eligibility counts | None; processed public extracts are included |
| `4_external_validation/4_2_external_validation.ipynb` | Source-specific joint errors and paired intervals | Fitted internal models and fold-local transformations |
| `4_external_validation/4_3_ngl_validation.ipynb` | Focused NGL transfer and interval results | Same fitted models and NGL imputers for new inference |

To rerun the public notebooks from the project root:

```bash
poetry run python scripts/execute_notebooks.py
```

For a complete experiment with authorized laboratory data, follow the commands
in the [root README](../README.md). Training flags are `False` by default, so
“Run All” displays the recorded fit results.

The fitting cells default to CPU and reuse compatible local checkpoints. Set
`LIQ_FINAL_DEVICE` before starting the kernel to select a different device.
