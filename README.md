# DPI-EVT

**Early-cycle forecasting of pore-pressure build-up and threshold timing with a structured accumulation decoder.**

DPI-EVT conditions a two-regime accumulation decoder on soil descriptors, prescribed
cyclic loading and a short observed pressure prefix. The forecast pressure envelope
and its threshold-crossing time come from the same trajectory. The prefix-anchored
variant, **DPI-EVT +A**, continues accumulation from the observed pressure level.

This repository contains the code and notebooks of the study. It covers DPI-EVT,
EVT-NeuralSSM, the sequence and event baselines, the matched Transformer controls,
the reciprocal-drift ablation, the Transformer/GBM reconciliations and the residual
calibration of onset intervals.

## Installation

Use Python **3.11 or 3.12** and Poetry **2.x**. Python 3.11 was used for the recorded
calculations. `poetry.lock` records the dependency versions.

```bash
git clone https://github.com/tnick1502/dpi-evt.git
cd dpi-evt
poetry install
poetry run jupyter lab
```

For the numerical package without notebook and test tools:

```bash
poetry install --only main
```

The recorded calculations used CPU. The notebook fitting cells default to CPU;
the commands below select CPU for the full protocol.

## Project layout

```text
data/
  external/                 Public cycle-level extracts and download instructions
notebooks/
  1_data_analysis/           Data preparation and eligibility
  2_model_training/          Baselines, EVT-NeuralSSM and DPI-EVT
  3_evaluations/             Joint accuracy, prefix study, intervals and controls
  4_external_validation/     Public inputs and source-specific transfer results
src/dpi_evt/                Models, causal preprocessing, training and metrics
external_validation/extval/ Source-specific external adapters
scripts/                   Experiment, control and calibration entry points
results/tables/             Aggregate results used by the public notebooks
tests/                     Numerical contracts requiring no private observations
```

Install `dpi-evt` and import the models from `dpi_evt`. The `extval` package
contains the external-data adapters.

## Start with the notebooks

The [notebook guide](notebooks/README.md) gives the purpose and input requirements
of each notebook. The public notebooks recreate tables and figures from the
included [aggregate results](results/tables/README.md) and inspect the supplied
public external extracts. All 11 notebooks include saved outputs from a fresh run.

Training notebooks show the fitting settings and recorded first-fold learning
histories. To train with authorized laboratory data, prepare the inputs and enable
the fitting cells. Compatible local checkpoints are reused when available.

## Data access and reproducibility

**The internal laboratory data cannot be published and are not included.**
The release also excludes specimen-level internal predictions, internal sample
identifiers, prepared internal arrays and trained checkpoints. Internal numerical
results are supplied as aggregate tables, without individual laboratory records.

The public external extracts cover Karlsruhe fine sand (TCUI and TCUA), LEAP-2017
GWU, Adapazari and Christchurch. See [data sources and downloads](data/external/SOURCES.md)
for original providers, dataset identifiers, processing conventions and attribution.
The files supplied here are the cycle-level extracts read by the adapters, not the
original full-rate time histories.

The full experiment commands below require access to the restricted laboratory
archive. They fit the models before evaluating the public external extracts.

## Run the full protocol with authorized data

Keep restricted inputs outside Git. Point `LIQ_REAL_ROOT` at the authorized local
archive, then run the stages from the project root:

```bash
export LIQ_REAL_ROOT=/absolute/path/to/authorized/laboratory/archive
export LIQ_DEVICE=cpu
export LIQ_FINAL_DEVICE=cpu
poetry run python scripts/run_experiments.py all --device cpu
poetry run python scripts/calibrate_intervals.py
poetry run python scripts/train_controls.py --workers 3
poetry run python scripts/analyze_controls.py
```

The main run contains five site-held-out folds repeated three times. Prefix
experiments use landmarks 0, 3, 5, 10 and 20 on a common eligible cohort and score
continuations after cycle 20. The primary landmark is cycle 10; the horizon is
3000 cycles and the pressure threshold is 0.95. Preprocessing and model fitting
use the corresponding internal training and validation partitions.

`results/experiments/latest.json` selects the local main run used by the control
and calibration scripts. Override it with `DPI_PRIMARY_RUN` if several local runs
exist. Outputs and restricted inputs are ignored by Git. Changes to the numerical
code, project path or recorded environment can create a new run identity;
incompatible cached checkpoints are rejected.

## Interpreting the comparisons

Pressure continuation and event timing are evaluated separately. Event log-MAE
uses observed events, and censoring-aware distribution scores use a
training-supported time window. Internal intervals resample sites, and external
intervals resample tests within each source. Repeated predictions of a specimen
are kept together.

The matched controls have the same hidden width and maximum number of epochs as
DPI-EVT but different parameter counts. Residual interval calibration changes the
reported onset intervals and leaves point forecasts and native event distributions
unchanged. Full definitions and the scope of the experiments are given in
[the protocol](docs/PROTOCOL.md).

## Checks

```bash
poetry check --lock
poetry run pytest
poetry run python scripts/execute_notebooks.py
```

The runner starts a fresh kernel for each notebook and uses the default inputs
described above. Optional HTML previews can be written to a directory outside the
repository with `--preview-dir`.

## License and citation

Author: Nikita R. Tishin. The code is available under the [MIT License](LICENSE).
The license does not cover the external data. These datasets remain subject to
their providers' terms, listed in [data sources](data/external/SOURCES.md), and
should be cited separately.

To cite the software, use [CITATION.cff](CITATION.cff) or the "Cite this repository"
menu on GitHub.
