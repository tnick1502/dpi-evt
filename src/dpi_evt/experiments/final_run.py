"""Site-held-out forecasting experiments and shared notebook entry points.

Preprocessing and fitting use the training and validation partitions of each fold.
Run identities distinguish cohorts, settings and cached model artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import torch

from dpi_evt.config import get_default_config, set_global_seed
from dpi_evt.data import load_population_artifact, prepare_benchmark_dataset, save_population_artifact
from dpi_evt.data.raw_loader import build_real_objects_population, DEFAULT_TEST_TYPES, find_cloud_root
from dpi_evt.data.splits import make_grouped_cv_folds
from dpi_evt.evaluation import collect_outputs, compute_metrics, fit_interval_scale
from dpi_evt.evaluation.cross_validation import publication_model_kwargs
from dpi_evt.evaluation.event_distribution import native_survival_scores, concordance_ipcw
from dpi_evt.evaluation.metrics import resolve_nliq_prediction
from dpi_evt.models import DPIEvtNet, EVTNeuralSSM, TransformerBaseline
from dpi_evt.models.forecast_operators import make_transfer_class, CalibratedTransformer
from dpi_evt.models.prefix_baselines import GBMHorizonMixture, PrefixGP
from dpi_evt.models.survival import CoxPHBaseline
from dpi_evt.training import train_model
from dpi_evt.training.losses import interpolated_crossing
from dpi_evt.experiments.artifact_checks import (
    file_digest, temporary_path, job_lock, split_digest, validate_predictions, validate_oof_rows,
)

PROTOCOL = "dpi-evt-forecast-v1"
LANDMARKS = (0, 3, 5, 10, 20)
CORE_MODELS = ("dpi_evt", "evt_ssm", "transformer", "gp_prefix", "gbm_distribution", "cox_ph")
PREFIX_MODELS = ("dpi_evt", "transformer", "gp_prefix")
OPERATOR_MODELS = tuple(f"{base}_{op}" for base in ("dpi_evt", "transformer") for op in ("A", "C", "AC"))
DISPLAY = {"dpi_evt": "DPI-EVT", "evt_ssm": "EVT-NeuralSSM",
           "transformer": "Transformer", "gp_prefix": "GP prefix (linear + SE)",
           "gbm_distribution": "GBM horizon mixture", "cox_ph": "Cox PH"}
for _base in ("dpi_evt", "transformer"):
    for _op in ("A", "C", "AC"):
        DISPLAY[f"{_base}_{_op}"] = DISPLAY[_base] + " +" + _op


@dataclass(frozen=True)
class RunSettings:
    smoke: bool = False
    n_splits: int = 5
    n_repeats: int = 3
    seed: int = 42
    mc_samples: int = 64
    baseline_epochs: int = 120
    physics_epochs: int = 200
    patience: int = 25

    def __post_init__(self):
        frozen = dict(n_splits=5, n_repeats=3, seed=42, mc_samples=64,
                      baseline_epochs=120, physics_epochs=200, patience=25)
        if not self.smoke and any(getattr(self, k) != v for k, v in frozen.items()):
            raise ValueError("Final protocol is frozen at 5x3 folds, seed=42, MC=64, epochs=120/200, patience=25; use smoke=True for checks")


def digest_files(paths):
    h = hashlib.sha256()
    for p in sorted(map(Path, paths)):
        h.update(str(p).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def code_digest(root):
    root = Path(root)
    return digest_files([*root.glob("src/dpi_evt/**/*.py"),
                         *root.glob("external_validation/extval/*.py")])


def preparation_code_digest(root):
    root = Path(root)
    return digest_files([root / "src/dpi_evt/config.py",
                         *root.glob("src/dpi_evt/data/*.py"),
                         *root.glob("src/dpi_evt/physics/*.py")])


def population_digest(pop):
    h = hashlib.sha256()
    h.update(pop["meta"].to_json(orient="split", double_precision=15).encode())
    for key in sorted(pop):
        if isinstance(pop[key], np.ndarray):
            a = np.ascontiguousarray(pop[key])
            h.update(key.encode()); h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temporary_path(path)
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def write_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temporary_path(path)
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def dataset_dir(root, n0=10):
    return Path(root) / "data" / "final_run" / f"n0_{n0:g}"


def prepare_populations(root, landmarks=LANDMARKS, raw_root=None, max_objects=None):
    """Rebuild each prefix from RAW observations truncated before smoothing, on one query grid."""
    root = Path(root)
    source = find_cloud_root([raw_root or "", root / "data" / "sites"])
    if source is None:
        raise FileNotFoundError("Local source data not found: set LIQ_REAL_ROOT or populate data/sites")
    source_files = [p for p in source.rglob("*") if p.is_file() and p.suffix in (".pickle", ".pkl", ".xls", ".xlsx")]
    if not source_files:
        raise FileNotFoundError(f"No local source files under {source}")
    source_hash = digest_files(source_files)
    code_hash = preparation_code_digest(root)
    audit = []
    for n0 in landmarks:
        cfg = replace(get_default_config(), dataset_source="real_objects", prefix_mode="landmark",
                      prefix_landmark_cycles=float(n0), query_prefix_horizon=20., seed=42)
        out = dataset_dir(root, n0)
        request = dict(protocol=PROTOCOL, source_hash=source_hash, code_hash=code_hash,
                       config=asdict(cfg), max_objects=max_objects)
        manifest = out / "preparation.json"
        if not manifest.exists() or json.loads(manifest.read_text()).get("request") != request:
            pop = build_real_objects_population([(source, DEFAULT_TEST_TYPES)], cfg, max_objects=max_objects)
            validate_population(pop, cfg)
            save_population_artifact(out, pop, cfg)
            write_json(manifest, {"request": request, "population_hash": population_digest(pop)})
        pop, cfg = load_final_population(root, n0)
        audit.append(dict(N0=n0, specimens=len(pop["meta"]), sites=pop["meta"].site_id.nunique(),
                          events=int(pop["liq_label"].sum()), prefix_nodes=float(pop["prefix_mask"].sum(1).mean()),
                          data=str(out)))
    table = pd.DataFrame(audit)
    table.to_csv(root / "data" / "final_run" / "cohort_by_prefix.csv", index=False)
    return table


def validate_population(pop, cfg):
    from dpi_evt.data.consistency import validate_forecasting_input_contract
    validate_forecasting_input_contract(pop)
    for key in ("static_features", "prefix_summary", "seq_inputs", "cycles", "delta_cycles", "r_obs", "prefix_obs"):
        if not np.isfinite(pop[key]).all():
            raise ValueError(f"Nonfinite values in prepared {key}")
    expected = pop["meta"]["CSR_base"].to_numpy()[:, None]
    if not np.allclose(pop["csr"], expected, atol=1e-7):
        raise ValueError("Final-run controls must use the prescribed CSR_base throughout the horizon")
    if not np.allclose(pop["cycles"], pop["cycles"][:1]) or (np.diff(pop["cycles"], axis=1) <= 0).any():
        raise ValueError("Query grid must be common, strictly increasing and duration-independent")
    if not np.isclose(pop["cycles"][0, -1], cfg.max_cycle_reference):
        raise ValueError("Wrong forecast horizon")
    pm = pop["prefix_mask"] > 0
    if np.any(pm & (pop["cycles"] > cfg.prefix_landmark_cycles + 1e-6)):
        raise ValueError("Prefix contains future observations")
    if "specimen_id" not in pop["meta"] or pop["meta"].specimen_id.duplicated().any():
        raise ValueError("Stable unique specimen_id is required")
    if cfg.prefix_landmark_cycles == 0 and any(np.any(pop[k]) for k in ("prefix_mask", "prefix_obs", "prefix_summary")):
        raise ValueError("N0=0 requires empty prefix inputs")


def load_final_population(root, n0=10):
    out = dataset_dir(root, n0)
    if not (out / "preparation.json").exists():
        raise FileNotFoundError(f"Prepare the authorized internal data with scripts/run_experiments.py prepare: missing {out}")
    pop, cfg = load_population_artifact(out)
    validate_population(pop, cfg)
    manifest = json.loads((out / "preparation.json").read_text())
    if manifest["request"]["config"] != asdict(cfg):
        raise ValueError(f"Dataset configuration differs from the preparation manifest: {out}")
    if manifest["population_hash"] != population_digest(pop):
        raise ValueError(f"Dataset content differs from preparation manifest: {out}")
    if manifest["request"]["code_hash"] != preparation_code_digest(root):
        raise ValueError("Preparation code has changed; rerun scripts/run_experiments.py prepare")
    return pop, cfg


def subset_population(pop, ids):
    idx = pop["meta"].set_index("specimen_id").index.get_indexer(ids)
    if (idx < 0).any():
        raise ValueError("Common prefix cohort contains missing specimens")
    out = dict(pop)
    n = len(pop["meta"])
    for k, v in pop.items():
        if isinstance(v, np.ndarray) and v.ndim and len(v) == n:
            out[k] = v[idx]
    out["meta"] = pop["meta"].iloc[idx].reset_index(drop=True)
    # New folds are always supplied explicitly after selecting the cohort.
    return out


def common_prefix_ids(root):
    populations = [load_final_population(root, n0)[0] for n0 in LANDMARKS]
    common = set(populations[0]["meta"].specimen_id)
    for pop in populations[1:]:
        common &= set(pop["meta"].specimen_id)
    return sorted(common)


def effective_config(cfg, settings):
    return replace(cfg, seed=settings.seed, mc_samples_eval=4 if settings.smoke else settings.mc_samples,
                   publication_baseline_epochs=1 if settings.smoke else settings.baseline_epochs,
                   publication_physics_epochs=1 if settings.smoke else settings.physics_epochs,
                   early_stopping_patience=2 if settings.smoke else settings.patience)


def model_spec(name, train, cfg, settings):
    base, _, op = name.partition("_") if name.startswith("transformer_") else (name, "", "")
    if name.startswith("dpi_evt_"):
        base, op = "dpi_evt", name[len("dpi_evt_"):]
    dims = dict(static_dim=train["static"].shape[1], prefix_dim=train["prefix_summary"].shape[1],
                seq_dim=train["seq_in"].shape[-1], seq_len=cfg.seq_len, prefix_len=cfg.prefix_len,
                max_cycle_reference=cfg.max_cycle_reference)
    classes = {"dpi_evt": DPIEvtNet, "evt_ssm": EVTNeuralSSM, 
               "transformer": CalibratedTransformer, "cox_ph": CoxPHBaseline,
               "gp_prefix": PrefixGP, "gbm_distribution": GBMHorizonMixture}
    if base not in classes:
        raise ValueError(f"Unknown model: {name}")
    cls = classes[base]
    accepted = set(inspect.signature(cls).parameters)
    kw = {k: int(v) if k.endswith("dim") or k.endswith("len") else v for k, v in dims.items() if k in accepted}
    if base in ("dpi_evt", "evt_ssm"):
        kw = publication_model_kwargs(base, kw, cfg)
        kw["hidden_dim"] = 192 if base == "evt_ssm" else 160
    if base == "dpi_evt":
        kw.update(crr_mode="hybrid", use_traj_residual=False, mc_train_samples=0, nliq_head_aux_weight=.1)
    if base == "evt_ssm":
        kw.update(integrator="heun", use_crr_damage=True, structured_post_event=True, use_trigger_head=True)
    if base == "transformer":
        kw.update(hidden_dim=96)
    if base in ("gp_prefix", "gbm_distribution"):
        kw.update(seed=cfg.seed)
        if base == "gp_prefix":
            kw.update(samples=cfg.mc_samples_eval)
        elif settings.smoke:
            kw.update(iterations=8, depth=3)
    if op:
        cls = make_transfer_class(cls)
        kw.update(t_anchor="A" in op, t_cap="C" in op)
    return cls, kw, base


def train_estimator(name, benchmark, cfg, settings, seed, device):
    cfg = replace(cfg, seed=seed)
    set_global_seed(seed)
    cls, kw, base = model_spec(name, benchmark["train"], cfg, settings)
    if "seed" in kw:
        kw["seed"] = seed
    model = cls(**kw).to(device)
    history = pd.DataFrame()
    if isinstance(model, torch.nn.Module):
        epochs = cfg.publication_physics_epochs if base in ("dpi_evt", "evt_ssm") else cfg.publication_baseline_epochs
        model, history = train_model(model, benchmark["train"], benchmark["val"], epochs=epochs,
                                     model_name=DISPLAY[name], config=cfg, device=device,
                                     verbose=False, scheduler="cosine", ema_decay=0.)
        fit_interval_scale(model, benchmark["val"], cfg, device, level=.90)
    else:
        model.fit(benchmark["train"], benchmark["val"])
    return model, history, kw


def attach_cap(split, external=False):
    # The same operator is used by the external validation adapters.
    import sys
    root = Path(__file__).resolve().parents[3]
    if str(root / "external_validation") not in sys.path:
        sys.path.insert(0, str(root / "external_validation"))
    from extval.cohort import ru_cap
    split["ru_cap"] = torch.as_tensor(ru_cap(split["meta"], c_to_kpa=1. if external else 1000.))
    return split


def evaluation_split(split, min_cycle=None):
    out = dict(split)
    if min_cycle is not None:
        out["continuation_mask"] = split["mask"] * (split["cycles"] > min_cycle).to(torch.float32)
    return out


def evaluate_outputs(name, outputs, split, train, cfg):
    met, samples = compute_metrics(name, outputs, split, cfg)
    met.pop("model", None)
    # An IBS of the deterministic point step is only a diagnostic, not a distributional IBS.
    met["N_liq_IBS_point_diagnostic"] = met.get("N_liq_IBS", np.nan)
    met["N_liq_IBS"] = np.nan
    event = split["label"].numpy() > .5
    samples["event_log_err"] = np.where(event, samples["nliq_log_err"], np.nan)
    samples["event_abs_err"] = np.where(event, samples["nliq_abs_err"], np.nan)
    if "traj_mean" in outputs and "traj_logvar" in outputs:
        widths = 2 * 1.6449 * np.sqrt(np.exp(outputs["traj_logvar"]))
        mask = split["continuation_mask"].numpy()
        samples["traj_interval_width_sum"] = (widths * mask).sum(1)
        samples["traj_interval_width"] = np.where(mask.sum(1) > 0,
            samples["traj_interval_width_sum"] / np.maximum(mask.sum(1), 1), np.nan)
    if "event_cdf" in outputs:
        native, per = native_survival_scores(outputs["event_cdf"], split["cycles"].numpy(),
                                            split["n_liq_true"].numpy(), split["label"].numpy(),
                                            train["n_liq_true"].numpy(), train["label"].numpy(),
                                            landmark=cfg.prefix_landmark_cycles)
        met.update(native); samples["native_ibs"] = per
        samples["event_tail_mass"] = outputs["event_tail_mass"]
        known = split["risk_label_observed"].numpy() > .5
        prob = 1 - outputs["event_tail_mass"]
        samples["event_brier_H"] = np.where(known, (prob - split["label"].numpy()) ** 2, np.nan)
        if known.any():
            met["Event_Brier_H"] = float(np.mean((prob[known] - split["label"].numpy()[known]) ** 2))
        # Coverage and restricted widths use the same event distribution for every family.
        q = split["cycles"].numpy()
        F = outputs["event_cdf"]
        # Rank the full survival distribution by restricted mean survival, avoiding the
        # artificial ties introduced when many Cox/GBM medians are censored at H.
        after = q[0] >= cfg.prefix_landmark_cycles
        x = q[0, after]
        S = 1 - F[:, after]
        if len(x) and x[0] > cfg.prefix_landmark_cycles:
            x = np.r_[cfg.prefix_landmark_cycles, x]
            S = np.c_[np.ones(len(S)), S]
        if len(x) > 1:
            rmst = (.5 * (S[:, 1:] + S[:, :-1]) * np.diff(x)).sum(1)
            met["Event_Cindex_ipcw"] = concordance_ipcw(rmst, split["n_liq_true"].numpy(),
                split["label"].numpy(), train["n_liq_true"].numpy(), train["label"].numpy())
            samples["event_rmst"] = rmst
        event = split["label"].numpy() > .5
        truth = split["n_liq_true"].numpy()
        for level in (.5, .9):
            quantiles = []
            names = ("event_q25", "event_q75") if level == .5 else ("nliq_q05", "nliq_q95")
            for alpha, key in zip(((1 - level) / 2, 1 - (1 - level) / 2), names):
                if key in outputs:
                    quantiles.append(outputs[key])
                else:
                    j = (F >= alpha).argmax(1)
                    quantiles.append(np.where((F >= alpha).any(1), q[np.arange(len(q)), j], np.inf))
            lo, hi = quantiles
            if event.any():
                suffix = str(int(level * 100))
                covered = (truth >= lo) & (truth <= hi)
                width = np.minimum(hi, cfg.max_cycle_reference) - np.minimum(lo, cfg.max_cycle_reference)
                tail = np.isinf(hi)
                met[f"Event_Coverage_{suffix}"] = float(covered[event].mean())
                met[f"Event_Width_{suffix}_restricted_H"] = float(width[event].mean())
                met[f"Event_Upper_Tail_{suffix}"] = float(tail[event].mean())
                samples[f"event_coverage{suffix}"] = np.where(event, covered, np.nan)
                samples[f"event_width{suffix}_restricted_H"] = np.where(event, width, np.nan)
                samples[f"event_upper_tail{suffix}"] = np.where(event, tail, np.nan)
    # The concordance with a censoring KM estimated on the test fold is kept as a labelled
    # diagnostic. The reported IPCW concordance uses the training censoring distribution.
    met["Cindex_ipcw_testKM_diagnostic"] = met.pop("N_liq_Cindex_ipcw", np.nan)
    met["N_liq_Cindex_ipcw"] = concordance_ipcw(
        resolve_nliq_prediction(outputs, cfg.max_cycle_reference), split["n_liq_true"].numpy(),
        split["label"].numpy(), train["n_liq_true"].numpy(), train["label"].numpy())
    if "ru_cap" in split:
        conflict = (split["prefix_obs"].max(1).values > split["ru_cap"]).numpy()
        met["Prefix_cap_conflict_rate"] = float(conflict.mean())
    if "specimen_id" in split["meta"]:
        samples["specimen_id"] = split["meta"].specimen_id.to_numpy()
    for key in outputs:
        if key.startswith("projection_"):
            samples[key] = outputs[key]
    return met, samples


def project_outputs(outputs, split, isotonic=False):
    from sklearn.isotonic import IsotonicRegression
    out = dict(outputs)
    raw = outputs["traj_mean"]
    if isotonic:
        r = np.stack([IsotonicRegression(y_min=0, y_max=1).fit_transform(c, y)
                      for c, y in zip(split["cycles"].numpy(), raw)])
    else:
        r = np.maximum.accumulate(np.clip(raw, 0, 1), axis=1)
    out["traj_mean"] = r
    out["nliq"] = interpolated_crossing(torch.as_tensor(r, dtype=torch.float32), split["cycles"], .95).numpy()
    for key in ("nliq_norm", "nliq_pred", "nliq_q05", "nliq_q95"):
        out.pop(key, None)
    delta = np.abs(r - raw)
    out["projection_l1"] = delta.mean(1)
    out["projection_changed_fraction"] = (delta > 1e-7).mean(1)
    out["projection_max"] = delta.max(1)
    out["projection_onset_shift"] = np.abs(out["nliq"] - resolve_nliq_prediction(outputs, float(split["cycles"][0, -1])))
    out["projection_integral_cycles"] = (.5 * (delta[:, 1:] + delta[:, :-1]) *
                                        np.diff(split["cycles"].numpy(), axis=1)).sum(1)
    diagnostic = dict(Projection_L1=float(delta.mean()), Projection_changed_fraction=float((delta > 1e-7).mean()),
                      Projection_max=float(delta.max()),
                      Projection_onset_shift=float(np.abs(out["nliq"] - resolve_nliq_prediction(outputs, float(split["cycles"][0, -1]))).mean()))
    diagnostic["Projection_integral_cycles"] = float((.5 * (delta[:, 1:] + delta[:, :-1]) *
                                                       np.diff(split["cycles"].numpy(), axis=1)).sum(1).mean())
    return out, diagnostic


class FinalRun:
    def __init__(self, root, settings=None, device=None):
        self.root = Path(root).resolve()
        self.settings = settings or RunSettings()
        from dpi_evt.accel import resolve_device, configure_performance
        requested = device if device not in (None, "auto") else None
        self.device = configure_performance(resolve_device(requested))
        self.code_hash = code_digest(root)
        from importlib.metadata import version
        packages = {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib")}
        self.data_hashes = {str(n0): json.loads((dataset_dir(root, n0) / "preparation.json").read_text())["population_hash"]
                            for n0 in LANDMARKS if (dataset_dir(root, n0) / "preparation.json").exists()}
        identity = dict(protocol=PROTOCOL, settings=asdict(self.settings), code=self.code_hash, data=self.data_hashes,
                        device=str(self.device), python=platform.python_version(), torch=torch.__version__, packages=packages)
        self.identity = identity
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.out = self.root / "results" / ("smoke" if self.settings.smoke else "experiments") / fingerprint
        write_json(self.out / "manifest.json", identity)
        if not self.settings.smoke:
            write_json(self.root / "results/experiments/latest.json", {"path": str(self.out.relative_to(self.root))})

    def populations(self, n0=10, common=False):
        pop, cfg = load_final_population(self.root, n0)
        if common:
            pop = subset_population(pop, common_prefix_ids(self.root))
        return pop, effective_config(cfg, self.settings)

    def folds(self, pop, cfg):
        return make_grouped_cv_folds(pop["meta"], len(pop["meta"]), self.settings.seed, cfg,
                                     n_splits=self.settings.n_splits, n_repeats=self.settings.n_repeats,
                                     val_objects=2)

    def fold_model(self, name, bm, cfg, fold_id, n0=10, common=False):
        path = self.out / ("prefix" if common else "primary") / f"n0_{n0:g}" / f"fold_{fold_id}" / name
        with job_lock(path):
            return self._fold_model(name, bm, cfg, fold_id, path)

    def _fold_model(self, name, bm, cfg, fold_id, path):
        marker = path / "complete.json"
        cls, expected_kwargs, _ = model_spec(name, bm["train"],
            replace(cfg, seed=self.settings.seed + fold_id), self.settings)
        expected = dict(identity=self.identity, kwargs=expected_kwargs,
                        train_ids=bm["train"]["meta"].specimen_id.tolist(),
                        val_ids=bm["val"]["meta"].specimen_id.tolist(),
                        test_ids=bm["test"]["meta"].specimen_id.tolist())
        if marker.exists():
            info = json.loads(marker.read_text())
            if any(info.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Stale checkpoint identity, kwargs or specimen split: {path}")
            checkpoint = path / ("weights.pt" if info["torch"] else "estimator.joblib")
            if not checkpoint.exists() or file_digest(checkpoint) != info.get("checkpoint_sha256"):
                raise ValueError(f"Missing or corrupt checkpoint: {checkpoint}")
            if info["torch"]:
                model = cls(**info["kwargs"])
                model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True))
            else:
                model = joblib.load(path / "estimator.joblib")
            return model.to(self.device)
        model, history, kw = train_estimator(name, bm, cfg, self.settings,
                                            self.settings.seed + fold_id, self.device)
        path.mkdir(parents=True, exist_ok=True)
        is_torch = isinstance(model, torch.nn.Module)
        checkpoint = path / ("weights.pt" if is_torch else "estimator.joblib")
        tmp = temporary_path(checkpoint)
        if is_torch:
            torch.save(model.to("cpu").state_dict(), tmp)
        else:
            joblib.dump(model, tmp)
        tmp.replace(checkpoint)
        model.to(self.device)
        history.to_csv(path / "history.csv", index=False)
        scale = float(model.calib_log_scale.exp().item()) if hasattr(model, "calib_log_scale") else None
        write_json(marker, dict(**expected, torch=is_torch, checkpoint_sha256=file_digest(checkpoint),
                               trained_epochs=len(history), variance_scale=scale,
                               variance_calibration=getattr(model, "_variance_scale_validation", None)))
        return model

    def predictions(self, name, split, cfg, path, load_model):
        """Accept a cached prediction only with matching inputs, identity and file digest."""
        path = Path(path)
        marker = path.with_suffix(".json")
        expected = dict(identity=self.identity, model=name, inputs=split_digest(split),
                        config=asdict(cfg))
        if path.exists() and marker.exists():
            meta = json.loads(marker.read_text())
            if any(meta.get(k) != v for k, v in expected.items()) or meta.get("sha256") != file_digest(path):
                raise ValueError(f"Stale or corrupt prediction cache: {path}")
            with np.load(path) as z:
                outputs = dict(z)
            validate_predictions(outputs, split, cfg.max_cycle_reference, name)
            return outputs
        outputs = collect_outputs(load_model(), split, cfg, self.device)
        validate_predictions(outputs, split, cfg.max_cycle_reference, name)
        write_npz(path, outputs)
        write_json(marker, dict(**expected, sha256=file_digest(path)))
        return outputs

    def train_preview(self, names):
        pop, cfg = self.populations()
        fold = self.folds(pop, cfg)[0]
        bm = prepare_benchmark_dataset(pop, cfg, self.device, precomputed_split=fold)
        for part in ("train", "val", "test"):
            attach_cap(bm[part])
        rows = []
        for name in names:
            model = self.fold_model(name, bm, cfg, 0)
            out = collect_outputs(model, bm["test"], cfg, self.device)
            met, _ = evaluate_outputs(DISPLAY[name], out, bm["test"], bm["train"], cfg)
            rows.append(dict(model=DISPLAY[name], **met))
        return pd.DataFrame(rows)

    def grouped(self, n0=10, common=False, models=None):
        pop, cfg = self.populations(n0, common)
        models = models or (PREFIX_MODELS if common else CORE_MODELS + OPERATOR_MODELS)
        rows, frames = [], []
        for fi, fold in enumerate(self.folds(pop, cfg)):
            bm = prepare_benchmark_dataset(pop, cfg, self.device, precomputed_split=fold)
            for part in ("train", "val", "test"):
                attach_cap(bm[part])
            test = evaluation_split(bm["test"], 20 if common else None)
            tag = "prefix" if common else "primary"
            for name in models:
                cache = self.out / tag / f"n0_{n0:g}" / f"fold_{fi}" / name / "predictions.npz"
                outputs = self.predictions(name, test, cfg, cache,
                    lambda: self.fold_model(name, bm, cfg, fi, n0, common))
                variants = [(DISPLAY[name], outputs, {})]
                if name == "transformer":
                    for iso, label in ((False, " +cummax"), (True, " +isotonic (diagnostic)")):
                        po, diag = project_outputs(outputs, test, iso)
                        variants.append((DISPLAY[name] + label, po, diag))
                for label, out, diagnostic in variants:
                    met, samples = evaluate_outputs(label, out, test, bm["train"], cfg)
                    rows.append(dict(model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0, **met, **diagnostic))
                    frames.append(samples.assign(model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0))
                print(f"{tag} N0={n0} fold={fi} {name}: done", flush=True)
            if n0 > 0:
                import sys
                ext_root = self.root / "external_validation"
                if str(ext_root) not in sys.path:
                    sys.path.insert(0, str(ext_root))
                from extval.evaluate import REFERENCE
                for label, func in REFERENCE.items():
                    r = func(test)
                    out = dict(traj_mean=r, nliq=interpolated_crossing(torch.as_tensor(r), test["cycles"], .95).numpy(),
                               risk_prob=(r[:, -1] >= .95).astype(float))
                    met, samples = evaluate_outputs(label, out, test, bm["train"], cfg)
                    rows.append(dict(model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0, **met))
                    frames.append(samples.assign(model=label, fold=fi, repeat=int(fold.get("repeat", 0)), N0=n0))
            if self.settings.smoke:
                break
        raw, samples = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
        if not self.settings.smoke:
            from dpi_evt.experiments.run_checks import internal_model_labels
            validate_oof_rows(raw, samples, pop["meta"].specimen_id, self.settings.n_repeats,
                              self.settings.n_splits, internal_model_labels(n0, common, models))
        table_dir = self.out / ("prefix" if common else "primary") / f"n0_{n0:g}"
        raw.to_csv(table_dir / "metrics.csv", index=False)
        samples.to_csv(table_dir / "per_test.csv", index=False)
        summary = raw.groupby("model").mean(numeric_only=True).drop(columns=["fold", "repeat", "N0"], errors="ignore")
        summary.to_csv(table_dir / "fold_mean.csv")
        return raw, samples

    def summarize(self, raw, samples, n_boot=2000):
        from dpi_evt.evaluation.significance import object_cluster_bootstrap, paired_significance
        metrics = [m for m in ("traj_rmse_continuation", "nliq_log_err", "event_log_err", "event_abs_err",
                    "coverage90", "traj_interval_width", "native_ibs", "event_brier_H", "event_coverage50",
                    "event_coverage90", "event_width90_restricted_H", "event_upper_tail90") if m in samples]
        ci = object_cluster_bootstrap(samples, ref="DPI-EVT", nboot=n_boot, err_metrics=metrics)
        paired = paired_significance(samples, ref="DPI-EVT", nboot=n_boot,
            metrics=["traj_rmse_continuation", "event_log_err", "nliq_log_err"])
        stability = raw.groupby(["repeat", "model"]).mean(numeric_only=True).reset_index()
        directory = self.out / "tables"
        directory.mkdir(parents=True, exist_ok=True)
        ci.to_csv(directory / "site_bootstrap.csv", index=False)
        paired.to_csv(directory / "paired_site_tests.csv", index=False)
        stability.to_csv(directory / "repeat_stability.csv", index=False)
        return ci, paired, stability


def preflight(root, require_data=True, deep=False):
    """Quick readiness checks without fitting, downloads or reuse of earlier outputs."""
    root = Path(root)
    rows = []
    for n0 in LANDMARKS:
        path = dataset_dir(root, n0)
        status = "not prepared"
        if (path / "preparation.json").exists():
            pop, cfg = load_final_population(root, n0)
            status = f"ready: {len(pop['meta'])} specimens"
        elif require_data:
            raise FileNotFoundError(f"Prepare final-run data first: {path}")
        rows.append(dict(stage="data", N0=n0, status=status, path=str(path)))
    report = pd.DataFrame(rows)
    if deep:
        from dpi_evt.experiments.run_checks import launch_audit
        report = pd.concat([report, launch_audit(root)], ignore_index=True)
    return report


def notebook_run(root):
    import os
    smoke = os.environ.get("LIQ_FINAL_SMOKE", "0") == "1"
    settings = RunSettings(smoke=True, n_splits=2, n_repeats=1, mc_samples=4) if smoke else RunSettings()
    return FinalRun(root, settings, device=os.environ.get("LIQ_FINAL_DEVICE", "auto"))
