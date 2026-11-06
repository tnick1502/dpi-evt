"""Check launch readiness and completion of the study experiment matrix."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

from .artifact_checks import file_digest, validate_predictions, validate_oof_rows


def internal_model_labels(n0, common=False, models=None):
    from .final_run import CORE_MODELS, PREFIX_MODELS, OPERATOR_MODELS, DISPLAY
    keys = models or (PREFIX_MODELS if common else CORE_MODELS + OPERATOR_MODELS)
    labels = [DISPLAY[k] for k in keys]
    if "transformer" in keys:
        labels += ["Transformer +cummax", "Transformer +isotonic (diagnostic)"]
    if n0 > 0:
        labels += reference_labels()
    return labels


def reference_labels():
    root = Path(__file__).resolve().parents[3]
    if str(root / "external_validation") not in sys.path:
        sys.path.insert(0, str(root / "external_validation"))
    from extval.evaluate import REFERENCE
    return list(REFERENCE)


def external_model_labels(n0, common=False):
    return [m for m in internal_model_labels(n0, common) if "isotonic" not in m]


def launch_audit(root):
    """Build all folds and external inputs without training a model."""
    from .final_run import (LANDMARKS, RunSettings, FinalRun, load_final_population,
                            common_prefix_ids, subset_population, dataset_dir, write_json)
    from dpi_evt.data import prepare_benchmark_dataset
    root = Path(root)
    run = FinalRun(root, RunSettings())
    common_ids = common_prefix_ids(root)
    populations = {n0: load_final_population(root, n0)[0] for n0 in LANDMARKS}
    base = subset_population(populations[0], common_ids)
    for n0, pop in populations.items():
        manifest = json.loads((dataset_dir(root, n0) / "preparation.json").read_text())
        if manifest["request"].get("max_objects") is not None:
            raise ValueError("The final run cannot use a max_objects limit")
        p = subset_population(pop, common_ids)
        for key in ("static_features", "cycles", "r_obs", "valid_mask", "liq_label", "n_liq_true"):
            if not np.array_equal(p[key], base[key]):
                raise ValueError(f"Common-cohort {key} changes across prefix settings, N0={n0}")
    if str(root / "external_validation") not in sys.path:
        sys.path.insert(0, str(root / "external_validation"))
    from extval.final_run import ExternalRun
    from extval.common import build_external_population
    records_cache, split_rows, rows = {}, [], []
    phases = [(10, False)] + [(n0, True) for n0 in LANDMARKS]
    for n0, common in phases:
        pop, cfg = run.populations(n0, common)
        seen = {}
        for fi, fold in enumerate(run.folds(pop, cfg)):
            bm = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"), precomputed_split=fold)
            parts = {p: bm[p]["meta"] for p in ("train", "val", "test")}
            sites = {p: set(m.site_id) for p, m in parts.items()}
            if any(sites[a] & sites[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
                raise ValueError(f"Site leakage in {'prefix' if common else 'primary'}, N0={n0}, fold={fi}")
            if any(len(m) == 0 for m in parts.values()):
                raise ValueError(f"Empty train/val/test split in fold {fi}")
            known = bm["train"]["risk_label_observed"].numpy() > .5
            if np.unique(bm["train"]["label"].numpy()[known]).size != 2:
                raise ValueError(f"GBM training lacks both observed outcome classes in fold {fi}")
            if int(bm["val"]["label"].sum()) < 2:
                raise ValueError(f"GBM scale calibration lacks validation events in fold {fi}")
            for part in parts:
                for key in ("static", "prefix_summary", "seq_in"):
                    if not torch.isfinite(bm[part][key]).all():
                        raise ValueError(f"Nonfinite standardized {part}/{key} in fold {fi}")
            rep = int(fold.get("repeat", 0))
            seen.setdefault(rep, []).extend(parts["test"].specimen_id)
            # NGL imputation must also be valid for every possible training-site combination.
            train_key = tuple(sorted(parts["train"].specimen_id))
            if train_key not in records_cache:
                records_cache[train_key] = ExternalRun.load_records(parts["train"])
            for ds, records in records_cache[train_key].items():
                external, _ = build_external_population(records, cfg)
                if not len(external["meta"]) or not np.isfinite(external["static_features"]).all():
                    raise ValueError(f"Invalid fold-local external inputs: {ds}, N0={n0}, fold={fi}")
            split_rows.append(dict(phase="prefix" if common else "primary", N0=n0, fold=fi, repeat=rep,
                train=len(parts["train"]), val=len(parts["val"]), test=len(parts["test"]),
                train_sites=len(sites["train"]), val_sites=len(sites["val"]), test_sites=len(sites["test"]),
                val_events=int(bm["val"]["label"].sum()),
                future_trajectory_subjects=int((bm["test"]["mask"] * (bm["test"]["cycles"] > (20 if common else n0))).sum(1).gt(0).sum())))
        for ids in seen.values():
            if len(ids) != len(set(ids)) or set(ids) != set(pop["meta"].specimen_id):
                raise ValueError("Incomplete or duplicated test specimens in a CV repeat")
        rows.append(dict(stage="prefix folds" if common else "primary folds", N0=n0,
                         status="ready: 15 disjoint folds + fold-local external inputs", path=str(run.out)))
    output = root / "data/final_run"
    pd.DataFrame(split_rows).to_csv(output / "split_audit.csv", index=False)
    free = shutil.disk_usage(root).free / 1024**3
    write_json(output / "launch_readiness.json", dict(status="ready_to_launch", full_training_executed=False,
        folds_audited=len(split_rows), neural_training_jobs=285, native_fits=30, gp_conditioning_jobs=90,
        free_disk_GiB=round(free, 2), primary_specimens=len(populations[10]["meta"]), common_specimens=len(common_ids),
        code_hash=run.code_hash, data_hashes=run.data_hashes))
    rows.append(dict(stage="workload", N0=np.nan, status=f"285 neural fits, 30 native fits, 90 GP jobs; {free:.1f} GiB free", path=str(output / "launch_readiness.json")))
    return pd.DataFrame(rows)


def verify_run(run, require_reports=True):
    """Accept completion only after checking every expected fold, model, input cohort and file hash."""
    from .final_run import CORE_MODELS, PREFIX_MODELS, OPERATOR_MODELS, LANDMARKS, write_json
    # A failed recheck must not leave an earlier success marker behind.
    (run.out / "completed.json").unlink(missing_ok=True)
    rows = []
    phases = [(10, False)] + [(n0, True) for n0 in LANDMARKS]
    n_folds = 1 if run.settings.smoke else run.settings.n_splits
    n_repeats = 1 if run.settings.smoke else run.settings.n_repeats
    for n0, common in phases:
        pop, cfg = run.populations(n0, common)
        role = "prefix" if common else "primary"
        location = run.out / role / f"n0_{n0:g}"
        raw = pd.read_csv(location / "metrics.csv")
        samples = pd.read_csv(location / "per_test.csv")
        folds = run.folds(pop, cfg)[:1] if run.settings.smoke else run.folds(pop, cfg)
        expected_ids = (set(samples.specimen_id) if run.settings.smoke else set(pop["meta"].specimen_id))
        keys = PREFIX_MODELS if common else CORE_MODELS + OPERATOR_MODELS
        validate_oof_rows(raw, samples, expected_ids, n_repeats, n_folds, internal_model_labels(n0, common))
        expected_pairs = {(fi, int(fold.get("repeat", 0))) for fi, fold in enumerate(folds)}
        for _, group in raw.groupby("model"):
            if set(zip(group.fold, group["repeat"])) != expected_pairs:
                raise ValueError("Wrong fold/repeat assignment in the OOF metric table")
        for fi, fold in enumerate(folds):
            test_idx = np.asarray(fold["benchmark_idx"])[fold["test_rel"]]
            ids = pop["meta"].iloc[test_idx].specimen_id.tolist()
            sample_rows = samples[(samples.fold == fi) & (samples["repeat"] == int(fold.get("repeat", 0)))]
            for _, group in sample_rows.groupby("model"):
                if set(group.specimen_id) != set(ids):
                    raise ValueError("OOF file does not match its assigned test fold")
            for name in keys:
                directory = location / f"fold_{fi}" / name
                checkpoint = json.loads((directory / "complete.json").read_text())
                train_idx = np.asarray(fold["benchmark_idx"])[fold["train_rel"]]
                val_idx = np.asarray(fold["benchmark_idx"])[fold["val_rel"]]
                if (checkpoint["identity"] != run.identity or checkpoint["test_ids"] != ids
                    or checkpoint["train_ids"] != pop["meta"].iloc[train_idx].specimen_id.tolist()
                    or checkpoint["val_ids"] != pop["meta"].iloc[val_idx].specimen_id.tolist()):
                    raise ValueError(f"Checkpoint has a different run/fold identity: {directory}")
                weights = directory / ("weights.pt" if checkpoint["torch"] else "estimator.joblib")
                if checkpoint["checkpoint_sha256"] != file_digest(weights):
                    raise ValueError(f"Checkpoint checksum failed: {weights}")
                _verify_prediction_file(directory / "predictions.npz", run, cfg, name, len(ids))
        rows.append(dict(stage=role, N0=n0, models=len(raw.model.unique()), folds=len(folds), status="complete"))
        ext = run.out / "external" / role / f"n0_{n0:g}"
        eraw, esamples = pd.read_csv(ext / "metrics.csv"), pd.read_csv(ext / "per_test.csv")
        audit = pd.read_csv(ext / "cohort_audit.csv")
        datasets = set(audit.dataset)
        if set(eraw.dataset) != datasets or len(datasets) != 5:
            raise ValueError("External source list is incomplete")
        if eraw.duplicated(["dataset", "model", "fold", "repeat"]).any():
            raise ValueError("Duplicate external evaluations")
        if not eraw.groupby(["dataset", "model"]).size().eq(len(folds)).all():
            raise ValueError("Missing external fold evaluations")
        if esamples.duplicated(["dataset", "model", "fold", "repeat", "specimen_id"]).any():
            raise ValueError("Duplicate external specimen predictions")
        group_keys = ["dataset", "model", "fold", "repeat"]
        if set(map(tuple, eraw[group_keys].to_numpy())) != set(map(tuple, esamples[group_keys].drop_duplicates().to_numpy())):
            raise ValueError("External per-test and fold metric tables contain different evaluation groups")
        for ds in datasets:
            expected = set(audit[(audit.dataset == ds) & audit.kept].specimen_id)
            source_rows = eraw[eraw.dataset == ds]
            if set(source_rows.model) != set(external_model_labels(n0, common)):
                raise ValueError(f"Missing models for external source {ds}")
            for _, group in source_rows.groupby("model"):
                if set(zip(group.fold, group["repeat"])) != expected_pairs:
                    raise ValueError("Wrong external fold/repeat assignment")
            for _, group in esamples[esamples.dataset == ds].groupby(["model", "fold", "repeat"]):
                if set(group.specimen_id) != expected:
                    raise ValueError(f"Wrong external analysis cohort for {ds}, N0={n0}")
            for fi in range(len(folds)):
                for name in keys:
                    _verify_prediction_file(ext / f"fold_{fi}" / name / f"{ds}.npz", run, cfg, name, len(expected))
        rows.append(dict(stage="external/" + role, N0=n0, models=eraw.model.nunique(), folds=len(folds), status="complete"))
    if require_reports:
        for name in ("site_bootstrap.csv", "paired_site_tests.csv", "prefix_pooled_site_bootstrap.csv",
                     "projection_corrections.csv", "event_reliability_H.csv", "external_summary.csv", "prefix_error.png"):
            if not (run.out / "tables" / name).is_file():
                raise FileNotFoundError(f"Missing final report artifact: {name}; run the report stage")
        manifest = json.loads((run.out / "report_manifest.json").read_text())
        if manifest["identity"] != run.identity:
            raise ValueError("Final report belongs to another run")
        for relative, checksum in {**manifest["files"], **manifest["sources"]}.items():
            if file_digest(run.out / relative) != checksum:
                raise ValueError(f"Final report or its inputs changed: {relative}; rerun the report stage")
    report = pd.DataFrame(rows)
    report.to_csv(run.out / "completion_audit.csv", index=False)
    if require_reports:
        write_json(run.out / "completed.json", dict(identity=run.identity, status="smoke_complete" if run.settings.smoke else "complete",
                   stages=len(rows), checksum_validation=True, full_training_executed=not run.settings.smoke))
    return report


def _verify_prediction_file(path, run, cfg, name, subjects):
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["identity"] != run.identity or meta["config"] != asdict(cfg) or meta["model"] != name:
        raise ValueError(f"Prediction identity mismatch: {path}")
    if meta["sha256"] != file_digest(path):
        raise ValueError(f"Prediction checksum failed: {path}")
    with np.load(path) as z:
        outputs = dict(z)
    from dpi_evt.data.ppr_envelope import forecasting_cycles
    c = forecasting_cycles(cfg.max_cycle_reference, cfg.seq_len, cfg.prefix_landmark_cycles,
                           cfg.prefix_len, cfg.query_prefix_horizon)
    split = {"cycles": torch.as_tensor(np.tile(c, (subjects, 1)))}
    validate_predictions(outputs, split, cfg.max_cycle_reference, name)
