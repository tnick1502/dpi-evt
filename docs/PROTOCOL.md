# Forecasting protocol

## Forecast information

The inputs are soil and initial-state descriptors, the prescribed constant-amplitude
CSR and the pressure observations at or before the prediction landmark. The prefix
is truncated before smoothing. The termination cycle and the future measured stress
amplitude are not used as inputs. The output is a monotone pressure-accumulation
envelope on a fixed 72-node query grid ending at 3000 cycles.

The primary internal cohort has 812 eligible tests from 13 sites at cycle 10:
490 observed events, 263 known by-horizon non-events and 59 unresolved censored
records. The common prefix cohort contains 788 tests and 466 events. Internal
onset requires three consecutive peaks at or above 0.95 at the recorded cycle
resolution. The model point time is the interpolated threshold crossing of its
mean curve.

## Fitting and comparisons

Five site-held-out folds are repeated three times, and two sites from each fitting
pool provide validation. Neural networks are trained with AdamW (learning rate
0.002, weight decay 0.0001, batch size 256), gradient clipping at 1, cosine decay
and early stopping with patience 25. The maximum number of epochs is 200 for the
structured models and 120 for the baseline Transformer. The matched Transformers
use width 160 and a 200-epoch ceiling. The main comparison is between DPI-EVT +A
and the anchored, crossing-supervised matched Transformer.

The head- and crossing-supervised Transformer controls share their initialization
and all other loss terms. The drift ablation sets only the reciprocal-cycle pressure
terms of the two regimes to zero. Anchored variants are trained with anchoring in
their forward computation. Two Transformer/GBM references reconcile predictions
without future observations or event labels, and their onset is read again from the
final curve.

## Metrics

Continuation RMSE pools squared pressure errors over valid future query points.
Onset log-MAE averages `abs(log1p(T_pred) - log1p(T_true))` over observed events.
Reported internal metrics pool out-of-fold predictions within each repeat and
then average the three repeat estimates. Paired ratio intervals use 20,000 site
resamples. A ratio interval that includes one does not establish equivalence.

The Wilcoxon continuation contrast uses site means of individual-specimen RMSE,
which is a different estimand from the pooled-point RMSE. Bootstrap tail fractions
are reported as descriptive diagnostics, not as hypothesis-test p-values. External
intervals are conditional on the source and the fixed fits and retain all 15
predictions of each test.

Residual onset intervals use asymmetric validation-event residuals in log1p time,
with the same procedure for all comparators. Because the validation sites were also
used for checkpoint selection, these are empirical observed-event intervals without
an independent split-conformal or cross-site population-coverage guarantee. Native
event CDFs and integrated Brier scores are not recalibrated in this step.

## External transfer

The five external sources contain 127 eligible tests and 99 observed events.
Pressure continuation uses 125 tests: one Adapazari and one Christchurch test end
before the next future query node. There are no known by-horizon external negatives.
KFS-TCUA contains one observed event and cannot support a timing bootstrap interval.
Source-specific pressure normalization and event rules are retained.

NGL stiffness and strength imputers use only the internal training fold. Missing
grain-size curves are filled with the external source median before quality
filtering (five retained Adapazari tests and one Christchurch test), and the
completed curves keep the adapter's zero missingness flag. External outcomes are
not used to fit model weights, variance scales or residual interval offsets. KFS
and LEAP were examined during model development, so the external results assess
the transfer of fixed models and are not a prospective validation.

## Scope

The resistance-shaped coordinate is a learned decoder coordinate, distinct from
material CRR. The single-term ablation tests reciprocal drift within the fixed
architecture and protocol. The endpoint is a pressure-threshold event shared by the
mixed soils. Laboratory stopping rules, strain-based failure prediction, arbitrary
loading histories and scheduling savings are outside the scope of the study.
