"""External evaluation of internally fitted models on KFS, LEAP and NGL records.

Each prediction uses an internal fold model and that fold's training scalers.
External outcomes are excluded from model, seed, interval-scale and operator selection."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from .cv import EXTERNAL_DEV, EXTERNAL_CONF
from . import kfs, leap, ngl, cohort
from .common import build_external_population, make_external_split
from .evaluate import REFERENCE
from dpi_evt.data import prepare_benchmark_dataset
from dpi_evt.evaluation import collect_outputs
from dpi_evt.experiments.final_run import (
    FinalRun, CORE_MODELS, PREFIX_MODELS, OPERATOR_MODELS, LANDMARKS, DISPLAY,
    attach_cap, evaluate_outputs, evaluation_split, project_outputs, population_digest,
    subset_population, write_json, write_npz,
)
from dpi_evt.training.losses import interpolated_crossing
import torch


class ExternalRun:
    def __init__(self, run: FinalRun):
        self.run = run
        pop, _ = run.populations()
        # The full internal metadata serve only the descriptive input audit. NGL prediction
        # features are rebuilt below from the training metadata of each fold.
        self.records = self.load_records(pop["meta"])
        self.out = run.out / "external"

    @staticmethod
    def load_records(training_meta):
        out = {"KFS-TCUI": kfs.records(series=("TCUI",), control_amplitude="nominal"),
               "KFS-TCUA": [r for r in kfs.records(series=("TCUA",), control_amplitude="nominal")
                            if r["name"] not in cohort.EXTERNAL_EXCLUDE],
               "LEAP-2017": leap.records(control_amplitude="nominal")}
        for ds in EXTERNAL_CONF:
            out[ds] = ngl.records(ds, training_meta=training_meta)
        return out

    def populations(self, cfg, common=False):
        pops, audits = {}, []
        for dataset, records in self.records.items():
            pop, audit = build_external_population(records, cfg)
            if common:
                common_ids = set(pop["meta"].specimen_id)
                for n0 in LANDMARKS:
                    other, _ = build_external_population(records, replace(cfg, prefix_landmark_cycles=float(n0)))
                    common_ids &= set(other["meta"].specimen_id)
                pop = subset_population(pop, sorted(common_ids))
            audit["specimen_id"] = [f"{r.get('object', 'external')}::{r['name']}" for r in records]
            audit["landmark_eligible"] = audit["kept"]
            audit["kept"] = audit.specimen_id.isin(pop["meta"].specimen_id)
            pops[dataset] = pop
            audits.append(audit.assign(dataset=dataset, N0=cfg.prefix_landmark_cycles,
                                       prefix_peak_records=[int((np.asarray(r["N"]) <= cfg.prefix_landmark_cycles).sum())
                                                            for r in records]))
        return pops, pd.concat(audits, ignore_index=True)

    def evaluate(self, n0=10, common=False, models=None):
        run = self.run
        pop, cfg = run.populations(n0, common)
        external, audit = self.populations(cfg, common)
        models = models or (PREFIX_MODELS if common else CORE_MODELS + OPERATOR_MODELS)
        location = self.out / ("prefix" if common else "primary") / f"n0_{n0:g}"
        location.mkdir(parents=True, exist_ok=True)
        audit.to_csv(location / "cohort_audit.csv", index=False)
        # External materialization is part of the cache identity, including event rules/targets.
        ext_hash = {k: population_digest(v) for k, v in external.items()}
        marker = location / "external_manifest.json"
        expected = dict(run_identity=run.identity, external_hashes=ext_hash,
                        status="external transfer evaluation",
                        interval_calibration="internal validation only")
        if marker.exists() and json.loads(marker.read_text()) != expected:
            raise ValueError(f"External data/config changed: use a new result namespace: {location}")
        write_json(marker, expected)
        rows, frames = [], []
        for fi, fold in enumerate(run.folds(pop, cfg)):
            bm = prepare_benchmark_dataset(pop, cfg, run.device, precomputed_split=fold)
            for part in ("train", "val", "test"):
                attach_cap(bm[part])
            fold_records = self.load_records(bm["train"]["meta"])
            fold_pops = {}
            for ds, records in fold_records.items():
                p, _ = build_external_population(records, cfg)
                fold_pops[ds] = subset_population(p, external[ds]["meta"].specimen_id.tolist())
            fold_hashes = {ds: population_digest(p) for ds, p in fold_pops.items()}
            fold_marker = location / f"fold_{fi}" / "inputs.json"
            if fold_marker.exists() and json.loads(fold_marker.read_text()) != fold_hashes:
                raise ValueError(f"Changed fold-local external inputs: {fold_marker}")
            write_json(fold_marker, fold_hashes)
            splits = {ds: evaluation_split(attach_cap(make_external_split(p, bm["scalers"], cfg), external=True),
                                          20 if common else None) for ds, p in fold_pops.items()}
            for name in models:
                model = None
                for ds, split in splits.items():
                    cache = location / f"fold_{fi}" / name / f"{ds}.npz"
                    def load_model():
                        nonlocal model
                        if model is None:
                            model = run.fold_model(name, bm, cfg, fi, n0, common)
                        return model
                    outputs = run.predictions(name, split, cfg, cache, load_model)
                    variants = [(DISPLAY[name], outputs, {})]
                    if name == "transformer":
                        projected, diag = project_outputs(outputs, split)
                        variants.append(("Transformer +cummax", projected, diag))
                    for label, out, diagnostic in variants:
                        met, samples = evaluate_outputs(label, out, split, bm["train"], cfg)
                        samples["prefix_cap_conflict"] = ((split["prefix_obs"] * split["prefix_mask"]).max(1).values > split["ru_cap"]).numpy()
                        rules = {f"{r.get('object', 'external')}::{r['name']}": r["event"].get("rule", "")
                                 for r in self.records[ds]}
                        samples["event_rule"] = split["meta"].specimen_id.map(rules).to_numpy()
                        rows.append(dict(dataset=ds, model=label, fold=fi, repeat=int(fold.get("repeat", 0)),
                                         N0=n0, **met, **diagnostic))
                        frames.append(samples.assign(dataset=ds, model=label, fold=fi,
                                                     repeat=int(fold.get("repeat", 0)), N0=n0))
                print(f"external N0={n0} fold={fi} {name}: done", flush=True)
            # Prefix references are deterministic and need no training. They keep the fold and
            # repeat indices so that they are compared on the same specimens.
            for ds, split in splits.items():
                if n0 == 0:
                    continue  # Too few prefix points for the prefix-fitted references.
                for label, func in REFERENCE.items():
                    r = func(split)
                    out = dict(traj_mean=r, nliq=interpolated_crossing(torch.as_tensor(r), split["cycles"], .95).numpy(),
                               risk_prob=(r[:, -1] >= .95).astype(float))
                    met, samples = evaluate_outputs(label, out, split, bm["train"], cfg)
                    rows.append(dict(dataset=ds, model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0, **met))
                    frames.append(samples.assign(dataset=ds, model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0))
            if run.settings.smoke:
                break
        raw, samples = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
        raw.to_csv(location / "metrics.csv", index=False)
        samples.to_csv(location / "per_test.csv", index=False)
        raw.groupby(["dataset", "model"]).mean(numeric_only=True).to_csv(location / "fold_mean.csv")
        return raw, samples


def summarize_external(raw):
    """Average fold metrics by dataset and model."""
    keys = [k for k in ("Traj_RMSE_continuation", "N_liq_logMAE_liq", "N_liq_logMAE",
                        "N_liq_Cindex_ipcw", "N_liq_IBS", "Coverage_90", "Interval_Width_90",
                        "Physics_Violation_Rate", "Event_Brier_H", "Event_Cindex_ipcw", "Event_Coverage_50",
                        "Event_Coverage_90", "Event_Width_90_restricted_H", "Event_Upper_Tail_90", "IBS_start", "IBS_tau") if k in raw]
    return raw.groupby(["dataset", "model"])[keys].mean()
