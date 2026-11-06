# Aggregate study results

These tables were recomputed on 6 November 2026 with the code in this repository:
285 neural fits, 30 GBM/Cox fits, 90 Gaussian-process jobs and 60 matched-control
fits on CPU. The main experiment uses all 15 site-held-out folds, the five prefix
lengths and the five external sources. The validation-residual intervals and the
paired bootstrap comparisons were recalculated from these predictions. The
tables reproduce the reported study results at their displayed precision.

The CSV files contain summaries by model, cohort and source, first-fold learning
histories and paired intervals. Internal specimen records, identifiers, predictions
and fitted weights remain in the local directories excluded from Git. Public notebooks
read these aggregate tables and the included external extracts.

`learning_history.csv` holds the validation losses of the first-fold models, and
`training_preview.csv` their evaluation. `prefix_summary.csv` scores the common
future window N > 20. Definitions are in [the protocol](../../docs/PROTOCOL.md).
