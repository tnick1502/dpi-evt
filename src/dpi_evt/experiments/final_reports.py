"""Tables and figures for the article from completed final-run predictions, without training or selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .final_run import LANDMARKS, write_json
from .artifact_checks import file_digest

MODEL_STYLE = {
    "DPI-EVT": dict(color="#246aac", marker="o", linestyle="-"),
    "Transformer +cummax": dict(color="#c17a24", marker="s", linestyle="--"),
    "GP prefix (linear + SE)": dict(color="#8658a6", marker="^", linestyle="-."),
}


def projection_table(samples):
    columns = [c for c in samples if c.startswith("projection_")]
    rows = []
    for model, group in samples.groupby("model"):
        if not columns or not group[columns].notna().any().any():
            continue
        row = dict(model=model, forecast_rows=len(group))
        for key in columns:
            values = group[key].dropna().to_numpy()
            if len(values):
                row[key + "_mean"] = float(values.mean())
                row[key + "_p95"] = float(np.quantile(values, .95))
                row[key + "_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def event_reliability(samples):
    cal = samples[(samples.risk_label_observed > .5) & samples.event_tail_mass.notna()].copy()
    cal["event_probability"] = 1 - cal.event_tail_mass
    cal["bin"] = pd.cut(cal.event_probability, np.linspace(0, 1, 6), include_lowest=True)
    return cal.groupby(["model", "bin"], observed=True).agg(
        predicted=("event_probability", "mean"), observed=("liq_label", "mean"),
        rows=("liq_label", "size"), unique_specimens=("specimen_id", "nunique")).reset_index()


def export_prefix_tables(run, n_boot=2000):
    from dpi_evt.evaluation.significance import object_cluster_bootstrap
    raw, intervals = [], []
    for n0 in LANDMARKS:
        folder = run.out / "prefix" / f"n0_{n0:g}"
        raw.append(pd.read_csv(folder / "metrics.csv"))
        samples = pd.read_csv(folder / "per_test.csv")
        focus = samples[samples.model.isin(MODEL_STYLE)].copy()
        intervals.append(object_cluster_bootstrap(focus, ref="DPI-EVT", nboot=n_boot,
                         err_metrics=["traj_rmse_continuation", "event_log_err"]).assign(N0=n0))
    raw, ci = pd.concat(raw, ignore_index=True), pd.concat(intervals, ignore_index=True)
    folder = run.out / "tables"; folder.mkdir(parents=True, exist_ok=True)
    raw.to_csv(folder / "prefix_metrics.csv", index=False)
    ci.to_csv(folder / "prefix_pooled_site_bootstrap.csv", index=False)
    return raw, ci


def export_reports(run, n_boot=2000):
    from .run_checks import verify_run
    verify_run(run, require_reports=False)
    folder = run.out / "tables"; folder.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(run.out / "primary/n0_10/metrics.csv")
    samples = pd.read_csv(run.out / "primary/n0_10/per_test.csv")
    run.summarize(raw, samples, n_boot=n_boot)
    projection_table(samples).to_csv(folder / "projection_corrections.csv", index=False)
    reliability = event_reliability(samples)
    reliability.to_csv(folder / "event_reliability_H.csv", index=False)
    pooled = []
    for (repeat, model), g in samples.groupby(["repeat", "model"]):
        denom = g.continuation_points.sum()
        pooled.append(dict(repeat=repeat, model=model,
            Traj_RMSE_continuation=np.sqrt(g.traj_sse_continuation.sum() / denom) if denom else np.nan,
            N_liq_logMAE_liq=g.event_log_err.mean(), N_liq_MAE_liq=g.event_abs_err.mean(),
            N_liq_logMAE=g.nliq_log_err.mean(), N_liq_IBS=g.native_ibs.mean()))
    pd.DataFrame(pooled).to_csv(folder / "repeat_pooled_errors.csv", index=False)
    _, ci = export_prefix_tables(run, n_boot)
    from extval.final_run import summarize_external
    external = pd.read_csv(run.out / "external/primary/n0_10/metrics.csv")
    summarize_external(external).to_csv(folder / "external_summary.csv")
    erows = [pd.read_csv(run.out / "external/prefix" / f"n0_{n0:g}" / "metrics.csv") for n0 in LANDMARKS]
    eprefix = pd.concat(erows, ignore_index=True)
    eprefix.to_csv(run.out / "external/prefix_metrics.csv", index=False)
    _plot_reports(run, ci, reliability, eprefix)
    hashes = {str(p.relative_to(run.out)): file_digest(p) for p in folder.iterdir() if p.is_file()}
    hashes["external/prefix_error.png"] = file_digest(run.out / "external/prefix_error.png")
    sources = {}
    for role in ("primary", "prefix", "external/primary", "external/prefix"):
        for pattern in ("*/metrics.csv", "*/per_test.csv", "*/cohort_audit.csv"):
            for p in (run.out / role).glob(pattern):
                sources[str(p.relative_to(run.out))] = file_digest(p)
    write_json(run.out / "report_manifest.json", dict(identity=run.identity,
        n_boot=n_boot, reference="DPI-EVT", estimates="OOF metrics averaged across repeats; site bootstrap",
        files=hashes, sources=sources))
    return folder


def _plot_reports(run, ci, reliability, eprefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for font in (Path(matplotlib.get_data_path()) / "fonts/ttf").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "font.size": 11, "axes.grid": True, "grid.alpha": .2})
    stamp = "SMOKE · " if run.settings.smoke else ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in zip(axes, ("event_log_err", "traj_rmse_continuation"),
                              ("Event-time log-MAE (events)", "ru continuation RMSE, N > 20")):
        for model, style in MODEL_STYLE.items():
            d = ci[ci.model == model].sort_values("N0")
            ax.plot(d.N0, d[key], label=model, **style)
            ax.fill_between(d.N0, d[key+"_lo"], d[key+"_hi"], color=style["color"], alpha=.12)
        ax.set(xlabel="Observed prefix N₀ (cycles)", ylabel=ylabel, xticks=LANDMARKS)
    axes[0].legend(fontsize=8)
    fig.suptitle(stamp + "Common cohort; mean over repeats; 95% site-bootstrap intervals", fontsize=11)
    fig.tight_layout(); fig.savefig(run.out / "tables/prefix_error.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, d in reliability.groupby("model"):
        ax.plot(d.predicted, d.observed, "o-", label=model)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="Predicted P(T ≤ H)", ylabel="Observed event fraction", xlim=(0, 1), ylim=(0, 1),
           title=stamp + "Event calibration; known outcomes only")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(run.out / "tables/event_calibration_H.png", dpi=160); plt.close(fig)
    datasets = list(eprefix.dataset.unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.4 * len(datasets), 6), sharey="row", squeeze=False)
    for col, ds in enumerate(datasets):
        for row, key in enumerate(("N_liq_logMAE_liq", "Traj_RMSE_continuation")):
            ax = axes[row, col]
            for model, style in MODEL_STYLE.items():
                d = eprefix[(eprefix.dataset == ds) & (eprefix.model == model)].groupby("N0")[key].mean()
                ax.plot(d.index, d, label=model, **style)
            if not eprefix.loc[eprefix.dataset == ds, key].notna().any():
                ax.text(.5, .5, "No observed events\nin common cohort", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
            ax.set(title=ds, xlabel="N₀ (cycles)", xticks=LANDMARKS)
            if col == 0:
                ax.set_ylabel(("Event-time log-MAE (events)", "ru continuation RMSE (N > 20)")[row])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(stamp + "External common cohorts; mean individual-fold metrics", fontsize=11)
    fig.tight_layout(); fig.savefig(run.out / "external/prefix_error.png", dpi=180); plt.close(fig)
