#!/usr/bin/env python3
"""Evaluate validation-residual onset intervals for the fitted study models.

Validation-event log residuals adjust interval bounds. These bounds are diagnostics
conditional on observed events; native quantiles, the event CDF and IBS are unchanged."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "external_validation")]
from dpi_evt.experiments.paths import primary_run
MODELS = {
    "dpi_evt": "DPI-EVT", "dpi_evt_A": "DPI-EVT +A",
    "evt_ssm": "EVT-NeuralSSM", "transformer": "Transformer",
    "transformer_cummax": "Transformer +cummax", "gp_prefix": "GP prefix (linear + SE)",
    "gbm_distribution": "GBM horizon mixture", "cox_ph": "Cox PH",
    "dpi_evt_C": "DPI-EVT +C", "dpi_evt_AC": "DPI-EVT +AC",
    "transformer_A": "Transformer +A", "transformer_C": "Transformer +C",
    "transformer_AC": "Transformer +AC", "persistence": "Persistence",
    "loglinear_prefix": "Log-linear prefix", "seed_booker": "Seed–Booker arcsine (prefix-fit)",
}
REFERENCES = ("persistence", "loglinear_prefix", "seed_booker")
CORE_COMPARISON = tuple(MODELS)[:8]
SOURCES = ("KFS-TCUI", "KFS-TCUA", "LEAP-2017", "Adapazari", "Christchurch")
LEVELS = (.90, .95)
N0, H = 10., 3000.
METHOD = "validation_log_asymmetric"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def corrected_quantile(values, probability):
    """Order statistic ceil((n+1)p); absent ranks produce a vacuous bound."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a finite vector")
    if not 0 < probability < 1:
        raise ValueError("Probability must lie strictly between zero and one")
    rank = int(np.ceil((len(values) + 1) * probability))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else np.inf


@dataclass(frozen=True)
class OnsetCalibrator:
    level: float
    lower_log_offset: float
    upper_log_offset: float
    n_events: int

    @classmethod
    def fit(cls, truth, point, event, level=.90):
        if not 0 < level < 1:
            raise ValueError("Coverage level must lie strictly between zero and one")
        truth, point, event = (np.asarray(x) for x in (truth, point, event))
        if truth.ndim != 1 or truth.shape != point.shape or truth.shape != event.shape:
            raise ValueError("Truth, point and event must be aligned vectors")
        if not np.isin(event, [0, 1]).all():
            raise ValueError("Event indicators must be binary")
        e = event.astype(bool)
        if not np.isfinite(truth[e]).all() or ((truth[e] < N0) | (truth[e] > H)).any():
            raise ValueError("Observed onset must lie within [N0,H]")
        if np.isnan(point[e]).any() or np.isneginf(point[e]).any():
            raise ValueError("Invalid event point predictions")
        residual = np.log1p(truth[e]) - np.log1p(np.clip(point[e], N0, H))
        p = 1 - (1 - level) / 2
        return cls(level, -corrected_quantile(-residual, p),
                   corrected_quantile(residual, p), len(residual))

    def predict(self, point):
        point = np.asarray(point, dtype=float)
        if np.isnan(point).any() or np.isneginf(point).any():
            raise ValueError("Invalid point predictions")
        center = np.log1p(np.clip(point, N0, H))
        lower = np.maximum(center + self.lower_log_offset, np.log1p(N0))
        upper = np.maximum(center + self.upper_log_offset, lower)
        # Intervals that extend past H get an infinite upper bound.
        lo = np.where(lower > np.log1p(H), np.inf, np.expm1(np.minimum(lower, np.log1p(H))))
        hi = np.where(upper > np.log1p(H), np.inf, np.expm1(np.minimum(upper, np.log1p(H))))
        return lo, hi

    def parameters(self):
        d = asdict(self)
        d["vacuous"] = not np.isfinite([self.lower_log_offset, self.upper_log_offset]).all()
        # Null offsets and the vacuous flag keep the manifest valid JSON.
        for k in ("lower_log_offset", "upper_log_offset"):
            if not np.isfinite(d[k]):
                d[k] = None
        return d


def assert_disjoint(calibration, evaluation):
    for col in ("specimen_id", "site_id"):
        if set(calibration[col]) & set(evaluation[col]):
            raise ValueError(f"Calibration/evaluation overlap in {col}")


def checkpoint_name(name):
    if name in REFERENCES:
        return None
    return "transformer" if name == "transformer_cummax" else name


def normalized_outputs(outputs, name, split):
    """Keep the point definition and native interval availability of each model family."""
    from dpi_evt.evaluation.metrics import resolve_nliq_prediction
    if name == "transformer_cummax":
        from dpi_evt.experiments.final_run import project_outputs
        outputs, _ = project_outputs(outputs, split)
    out = dict(outputs)
    out["nliq"] = resolve_nliq_prediction(out, H)
    if ("nliq_q05" in out) != ("nliq_q95" in out):
        raise ValueError("Incomplete native interval pair")
    return out


def reference_outputs(name, split):
    import torch
    from extval.evaluate import REFERENCE
    from dpi_evt.training.losses import interpolated_crossing
    curve = REFERENCE[MODELS[name]](split)
    return dict(nliq=interpolated_crossing(torch.as_tensor(curve), split["cycles"], .95).numpy())


def interval_rows(a, calibrators, model, fold, source):
    rows = []
    # A missing native interval is recorded as missing, not as zero coverage.
    native = "nliq_q05" in a and "nliq_q95" in a
    methods = [("native", .90, a["nliq_q05"], a["nliq_q95"])] if native else []
    methods += [(METHOD, c.level, *c.predict(a["nliq"])) for c in calibrators]
    for method, level, lo, hi in methods:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
            raise ValueError("Invalid interval bounds")
        e = np.asarray(a["event"]).astype(bool)
        truth = np.asarray(a["truth"], dtype=float)
        frame = pd.DataFrame({
            "model": model, "fold": fold, "repeat": fold // 5, "source": source,
            "method": method, "target": level, "native_available": native, "specimen_id": a["specimen_id"],
            "site_id": a["site_id"], "programme": a["programme"],
            "event": e.astype(int), "truth": truth, "point": np.asarray(a["nliq"], dtype=float),
            "lower": lo, "upper": hi,
            "coverage": np.where(e, (lo <= truth) & (truth <= hi), np.nan),
            "width_H": np.where(e, np.maximum(0, np.minimum(hi, H) - np.minimum(lo, H)), np.nan),
            "upper_tail": np.where(e, hi > H, np.nan),
            "lower_tail": np.where(e, lo > H, np.nan),
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize(records, keys):
    events = records[records.event == 1]
    out = events.groupby(keys).agg(
        event_forecasts=("coverage", "size"), unique_events=("specimen_id", "nunique"),
        coverage=("coverage", "mean"), mean_width_H=("width_H", "mean"),
        median_width_H=("width_H", "median"), upper_tail=("upper_tail", "mean"),
        lower_tail=("lower_tail", "mean")).reset_index()
    sites = events.groupby(keys + ["site_id"]).coverage.mean().groupby(keys).agg(
        site_macro="mean", worst_site="min").reset_index()
    return out.merge(sites, on=keys)


def bootstrap_coverage(records, draws=3000):
    result = []
    for key, g in records[records.event == 1].groupby(["model", "method", "target"]):
        s = g.groupby("site_id").coverage.agg(["sum", "size"])
        rng = np.random.default_rng(11705)
        ix = rng.integers(0, len(s), (draws, len(s)))
        cov = s["sum"].to_numpy()[ix].sum(1) / s["size"].to_numpy()[ix].sum(1)
        result.append(dict(zip(["model", "method", "target"], key)) |
                      dict(coverage_lo=np.quantile(cov, .025), coverage_hi=np.quantile(cov, .975)))
    return pd.DataFrame(result)


def source_hashes(root, run):
    paths = [run / "manifest.json", run / "completed.json",
             root / "data/final_run/n0_10/arrays.npz", root / "data/final_run/n0_10/meta.parquet",
             root / "data/final_run/n0_10/preparation.json", run / "external/primary/n0_10/per_test.csv",
             run / "primary/n0_10/per_test.csv"]
    for fi in range(15):
        for name in MODELS:
            parent = checkpoint_name(name)
            if parent is None:
                continue
            base = run / f"primary/n0_10/fold_{fi}/{parent}"
            info = json.loads((base / "complete.json").read_text())
            weight = "weights.pt" if info["torch"] else "estimator.joblib"
            paths += [base / k for k in (weight, "complete.json", "predictions.npz", "predictions.json")]
            for source in SOURCES:
                p = run / f"external/primary/n0_10/fold_{fi}/{parent}/{source}.npz"
                paths += [p, p.with_suffix(".json")]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def validate_artifacts(out):
    out = Path(out)
    done = json.loads((out / "completed.json").read_text())
    if done["status"] != "complete":
        raise ValueError("Incomplete interval evaluation")
    for rel, digest in done["outputs"].items():
        if sha(out / rel) != digest:
            raise ValueError(f"Corrupt interval artifact: {rel}")
    return done


def ensure_onset_interval_evaluation(root=ROOT, base_run=None):
    """Build or check a content-addressed interval evaluation without training models."""
    root = Path(root).resolve()
    run = Path(base_run).resolve() if base_run else primary_run(root)
    for path in (root / "src", root / "external_validation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import joblib
    from dpi_evt.experiments.final_run import (
        RunSettings, code_digest, load_final_population, effective_config, model_spec,
        attach_cap, population_digest, split_digest)
    from dpi_evt.data.splits import prepare_benchmark_dataset, make_grouped_cv_folds
    from dpi_evt.evaluation.metrics import collect_outputs

    identity = json.loads((run / "manifest.json").read_text())
    completed = json.loads((run / "completed.json").read_text())
    if (completed["status"] != "complete" or completed["identity"] != identity or
            not completed["checksum_validation"] or identity["code"] != code_digest(root)):
        raise ValueError("Base run is incomplete or numerical source code changed")
    settings = RunSettings(**identity["settings"])
    if settings.smoke or (settings.n_splits, settings.n_repeats) != (5, 3):
        raise ValueError("This evaluation requires the complete 15-fold run")
    if str(torch.__version__) != identity["torch"]:
        raise ValueError("Frozen inference requires the recorded torch version")
    inputs = source_hashes(root, run)
    config = dict(protocol="onset-validation-residual-v2", base_run=str(run.relative_to(root)),
                  implementation_sha256=sha(__file__), inputs=inputs, levels=list(LEVELS),
                  models={name: dict(label=label, checkpoint=checkpoint_name(name)) for name, label in MODELS.items()},
                  N0=N0, horizon=H, bootstrap_draws=3000, bootstrap_seed=11705,
                  calibration="original internal validation events, fold-local",
                  interpretation="Empirical recalibration; validation also used for early stopping. No fresh split-conformal or cross-site guarantee.",
                  primary_target=.90, sensitivity_target=.95,
                  unchanged="trained weights, trajectories, point onset, event CDF, IBS, native quantiles")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = root / "results/onset_intervals" / fingerprint
    if (out / "completed.json").exists():
        validate_artifacts(out)
        write_json(out.parent / "latest.json", dict(path=str(out.relative_to(root)), fingerprint=fingerprint))
        return out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", config)
    pop, cfg0 = load_final_population(root, 10)
    if population_digest(pop) != identity["data"]["10"]:
        raise ValueError("Base population differs from the frozen run")
    cfg = effective_config(cfg0, settings)
    folds = make_grouped_cv_folds(pop["meta"], len(pop["meta"]), settings.seed, cfg,
                                 n_splits=5, n_repeats=3, val_objects=2)
    ext = pd.read_csv(run / "external/primary/n0_10/per_test.csv", low_memory=False)
    original_internal = pd.read_csv(run / "primary/n0_10/per_test.csv", low_memory=False)
    internal, external, parameters = [], [], {}
    for fi, fold in enumerate(folds):
        bm = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"), precomputed_split=fold)
        for part in ("train", "val", "test"):
            attach_cap(bm[part])
        assert_disjoint(bm["val"]["meta"], bm["test"]["meta"])
        for name, label in MODELS.items():
            parent = checkpoint_name(name)
            # Reference forecasters share the same frozen fold membership.
            base = run / f"primary/n0_10/fold_{fi}/{parent or 'dpi_evt'}"
            info = json.loads((base / "complete.json").read_text())
            weight = base / ("weights.pt" if info["torch"] else "estimator.joblib")
            if info["identity"] != identity or info["checkpoint_sha256"] != sha(weight):
                raise ValueError("Frozen checkpoint identity mismatch")
            for part in ("train", "val", "test"):
                if info[part + "_ids"] != bm[part]["meta"].specimen_id.tolist():
                    raise ValueError("Frozen specimen split mismatch")
            valpath = out / "validation" / f"{name}_fold_{fi}.npz"
            cache_identity = dict(checkpoint=info["checkpoint_sha256"] if parent else None,
                                  model=name, inputs=split_digest(bm["val"]),
                                  config=asdict(cfg), ids=info["val_ids"])
            if valpath.exists():
                cache = json.loads(valpath.with_suffix(".json").read_text())
                if cache["identity"] != cache_identity or cache["sha256"] != sha(valpath):
                    raise ValueError(f"Stale validation inference: {valpath}")
                a = dict(np.load(valpath))
            else:
                if parent is None:
                    pred = reference_outputs(name, bm["val"])
                else:
                    if info["torch"]:
                        cls, _, _ = model_spec(parent, bm["train"], cfg, settings)
                        model = cls(**info["kwargs"])
                        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
                    else:
                        model = joblib.load(weight)
                    pred = collect_outputs(model, bm["val"], cfg, torch.device("cpu"))
                    pred = normalized_outputs(pred, name, bm["val"])
                    del model
                a = dict(nliq=pred["nliq"], truth=bm["val"]["n_liq_true"].numpy(),
                         event=bm["val"]["label"].numpy(),
                         specimen_id=bm["val"]["meta"].specimen_id.to_numpy(dtype=str),
                         site_id=bm["val"]["meta"].site_id.to_numpy(dtype=str))
                valpath.parent.mkdir(exist_ok=True)
                np.savez_compressed(valpath, **a)
                write_json(valpath.with_suffix(".json"), dict(identity=cache_identity, sha256=sha(valpath)))
            calibrators = [OnsetCalibrator.fit(a["truth"], a["nliq"], a["event"], level) for level in LEVELS]
            parameters[f"{name}/{fi}"] = dict(validation_ids=info["val_ids"],
                validation_sites=sorted(set(a["site_id"].tolist())),
                levels=[c.parameters() for c in calibrators])
            test = bm["test"]
            frozen = original_internal[(original_internal.model == label) & (original_internal.fold == fi)]
            if frozen.specimen_id.tolist() != test["meta"].specimen_id.tolist():
                raise ValueError("Internal prediction/metadata identity mismatch")
            b = (normalized_outputs(dict(np.load(base / "predictions.npz")), name, test)
                 if parent else dict(nliq=frozen.nliq_pred.to_numpy()))
            if not np.allclose(frozen.nliq_pred, b["nliq"], rtol=1e-6, atol=1e-3):
                raise ValueError("Internal point prediction definition changed")
            b.update(truth=test["n_liq_true"].numpy(), event=test["label"].numpy(),
                     specimen_id=test["meta"].specimen_id.to_numpy(dtype=str),
                     site_id=test["meta"].site_id.to_numpy(dtype=str),
                     programme=test["meta"].load_mode.to_numpy(dtype=str))
            internal.append(interval_rows(b, calibrators, name, fi, "internal"))
            for source in SOURCES:
                g = ext[(ext.model == label) & (ext.fold == fi) & (ext.dataset == source)]
                if parent:
                    b = dict(np.load(run / f"external/primary/n0_10/fold_{fi}/{parent}/{source}.npz"))
                    # The external grid is the same fixed grid used by the frozen run.
                    ext_grid = torch.as_tensor(np.broadcast_to(pop["cycles"][0], b["traj_mean"].shape).copy()) if name == "transformer_cummax" else None
                    b = normalized_outputs(b, name, {"cycles": ext_grid})
                else:
                    b = dict(nliq=g.nliq_pred.to_numpy())
                if len(g) != len(b["nliq"]) or not np.allclose(g.nliq_pred, b["nliq"], atol=1e-3):
                    raise ValueError("External prediction/metadata alignment mismatch")
                b.update(truth=g.N_liq_true.to_numpy(), event=g.liq_label.to_numpy(),
                         specimen_id=g.specimen_id.to_numpy(dtype=str), site_id=np.repeat(source, len(g)),
                         programme=g.load_mode.to_numpy(dtype=str))
                external.append(interval_rows(b, calibrators, name, fi, source))
            print(f"Interval evaluation {name}, fold {fi}: {calibrators[0].n_events} validation events", flush=True)
    d, e = pd.concat(internal, ignore_index=True), pd.concat(external, ignore_index=True)
    d.to_csv(out / "internal_intervals.csv", index=False)
    e.to_csv(out / "external_intervals.csv", index=False)
    keys = ["model", "method", "target"]
    s = summarize(d, keys).merge(bootstrap_coverage(d), on=keys)
    s.to_csv(out / "internal_summary.csv", index=False)
    for group in ("repeat", "site_id", "programme"):
        # site_id is already a grouping key in the site-macro summary.
        if group == "site_id":
            d[d.event == 1].groupby(keys + [group]).agg(coverage=("coverage", "mean"),
                mean_width_H=("width_H", "mean"), event_forecasts=("coverage", "size")).reset_index().to_csv(
                    out / "internal_site_summary.csv", index=False)
        else:
            summarize(d, keys + [group]).to_csv(out / f"internal_{group}_summary.csv", index=False)
    summarize(e, keys + ["source"]).to_csv(out / "external_summary.csv", index=False)
    summarize(e, keys + ["source", "fold"]).to_csv(out / "external_fold_summary.csv", index=False)
    write_json(out / "calibration_parameters.json", parameters)
    inventory = d.groupby("model").agg(native_available=("native_available", "first"),
        folds=("fold", "nunique")).reset_index()
    inventory["label"] = inventory.model.map(MODELS)
    inventory["checkpoint"] = inventory.model.map(checkpoint_name)
    inventory.to_csv(out / "model_inventory.csv", index=False)
    if source_hashes(root, run) != inputs:
        raise RuntimeError("Frozen inputs changed during interval evaluation")
    outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "completed.json", dict(status="complete", fingerprint=fingerprint,
        frozen_inputs_unchanged=True, outputs=outputs, trained_models=0,
        internal_unique_records=int(d.specimen_id.nunique()),
        external_unique_records=int(e[["source", "specimen_id"]].drop_duplicates().shape[0])))
    write_json(out.parent / "latest.json", dict(path=str(out.relative_to(root)), fingerprint=fingerprint))
    return out


def plot_internal(out):
    import matplotlib.pyplot as plt
    s = pd.read_csv(Path(out) / "internal_summary.csv")
    names = [name for name in MODELS if name in set(s.model)]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .58*len(names)+1.8)),
                             sharey=True, constrained_layout=True)
    styles = [("native", .9, "Native 90% (where available)", -.20, "#777777", "s"),
              (METHOD, .9, "Validation target 90%", 0, "#2866a0", "o"),
              (METHOD, .95, "Target 95% (sensitivity)", .20, "#cf7523", "^")]
    for method, target, label, offset, color, marker in styles:
        g = s[(s.method == method) & np.isclose(s.target, target)].set_index("model")
        keep = [name for name in names if name in g.index]
        g = g.loc[keep]
        y = np.array([names.index(name) for name in keep]) + offset
        axes[0].errorbar(g.coverage, y,
            xerr=[g.coverage-g.coverage_lo, g.coverage_hi-g.coverage], fmt=marker,
            color=color, ms=5, capsize=2, lw=1, label=label)
        axes[1].plot(g.mean_width_H, y, marker, color=color, ms=5)
    axes[0].axvline(.9, ls="--", color="gray", lw=1)
    axes[0].set(xlabel="Observed-event coverage; 95% site-bootstrap CI", xlim=(-.02, 1.03))
    axes[0].set_yticks(range(len(names)), [MODELS[name] for name in names], fontsize=9)
    axes[0].invert_yaxis()
    axes[1].set(xlabel="Mean H-restricted width, cycles", xlim=(0, 3050))
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Same validation sites and calibration rule for every model; 490 events, 3 OOF repeats", fontsize=11)
    return fig


def plot_external(out, model=None):
    import matplotlib.pyplot as plt
    s = pd.read_csv(Path(out) / "external_summary.csv")
    if model is None:
        names = [name for name in MODELS if name in set(s.model)]
        fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .5*len(names)+1.8)),
                                 sharey=True, constrained_layout=True)
        for ax, target in zip(axes, LEVELS):
            g = s[(s.method == METHOD) & np.isclose(s.target, target)]
            matrix = g.pivot(index="model", columns="source", values="coverage").reindex(index=names, columns=SOURCES)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            for i in range(len(names)):
                for j in range(len(SOURCES)):
                    value = matrix.iloc[i, j]
                    ax.text(j, i, f"{100*value:.1f}%", ha="center", va="center", fontsize=8,
                            color="white" if value > .65 else "black")
            counts = g.groupby("source").unique_events.first().reindex(SOURCES)
            ax.set_xticks(range(len(SOURCES)), [f"{source}\n{n} event{'s' if n != 1 else ''}" for source, n in zip(SOURCES, counts)], fontsize=8)
            ax.set_yticks(range(len(names)), [MODELS[name] for name in names], fontsize=9)
            ax.set_title(f"Validation target {target:.0%}" + (" (sensitivity)" if target == .95 else ""))
            ax.grid(False)
        fig.colorbar(im, ax=axes, label="Observed-event coverage", shrink=.7)
        fig.suptitle("External transfer of internal calibration; 15 frozen fits per specimen", fontsize=11)
        return fig
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for method, target, label, offset in [("native", .9, "Native 90%", -.24),
            (METHOD, .9, "Validation target 90%", 0), (METHOD, .95, "Target 95% (sensitivity)", .24)]:
        g = s[(s.model == model) & (s.method == method) & np.isclose(s.target, target)].set_index("source").loc[list(SOURCES)]
        ax.bar(np.arange(len(SOURCES)) + offset, g.coverage, .23, label=label)
    counts = s[(s.model == model) & (s.method == METHOD) & np.isclose(s.target, .9)].set_index("source").loc[list(SOURCES)].unique_events
    ax.set_xticks(range(len(SOURCES)), [f"{source}\n{n} distinct events" for source, n in zip(SOURCES, counts)])
    ax.axhline(.9, ls="--", color="gray", lw=1)
    ax.set(ylabel="External event coverage (15 frozen fits)", ylim=(0, 1.08), title=MODELS[model])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    path = ensure_onset_interval_evaluation(args.root)
    print(path)
    print(pd.read_csv(path / "internal_summary.csv").to_string(index=False))
