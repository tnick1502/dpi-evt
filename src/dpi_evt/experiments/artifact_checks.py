"""Checks for resumable final-run artifacts. No model fitting or target-domain tuning."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def temporary_path(path):
    path = Path(path)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


@contextmanager
def job_lock(directory):
    """OS releases the lock after a crash; two notebooks cannot train the same job together."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".job.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is running this job: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def split_digest(split):
    h = hashlib.sha256()
    h.update(json.dumps(split["meta"].specimen_id.tolist(), ensure_ascii=False).encode())
    for key in ("static", "prefix_summary", "seq_in", "cycles", "delta_cycles", "csr",
                "prefix_obs", "prefix_mask", "ru_cap"):
        if key in split:
            value = split[key].detach().cpu().numpy()
            value = np.ascontiguousarray(value)
            h.update(key.encode()); h.update(str((value.shape, value.dtype)).encode()); h.update(value.tobytes())
    return h.hexdigest()


def validate_predictions(outputs, split, horizon, name=""):
    from dpi_evt.evaluation.metrics import resolve_nliq_prediction
    n, steps = split["cycles"].shape
    p = np.asarray(outputs["risk_prob"])
    if p.shape != (n,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f"{name}: invalid horizon-risk probabilities")
    t = resolve_nliq_prediction(outputs, horizon)
    if t.shape != (n,) or not np.isfinite(t).all() or np.any((t < 0) | (t > horizon + 1e-3)):
        raise ValueError(f"{name}: invalid point event times")
    if "traj_mean" in outputs:
        r = np.asarray(outputs["traj_mean"])
        if r.shape != (n, steps) or not np.isfinite(r).all():
            raise ValueError(f"{name}: invalid forecast trajectories")
        structured = name.startswith(("dpi_evt", "evt_ssm")) or name.endswith(("_A", "_C", "_AC"))
        if structured and (np.any(np.diff(r, axis=1) < -1e-5) or np.any((r < -1e-5) | (r > 1 + 1e-5))):
            raise ValueError(f"{name}: an admissible-model trajectory violates its output contract")
    if "traj_logvar" in outputs:
        v = np.asarray(outputs["traj_logvar"])
        if v.shape != (n, steps) or not np.isfinite(v).all() or np.any(v > 80):
            raise ValueError(f"{name}: invalid predictive variances")
    requires_distribution = name.startswith("dpi_evt") or name in ("gp_prefix", "gbm_distribution", "cox_ph")
    if requires_distribution and "event_cdf" not in outputs:
        raise ValueError(f"{name}: missing native event distribution")
    if "event_cdf" in outputs:
        F = np.asarray(outputs["event_cdf"])
        tail = np.asarray(outputs["event_tail_mass"])
        if (F.shape != (n, steps) or not np.isfinite(F).all() or np.any((F < -1e-6) | (F > 1+1e-6))
                or np.any(np.diff(F, axis=1) < -1e-6)):
            raise ValueError(f"{name}: invalid event CDF")
        if tail.shape != (n,) or not np.allclose(tail, 1-F[:, -1], atol=1e-6):
            raise ValueError(f"{name}: inconsistent no-hit probability")
        if "event_survival" in outputs and not np.allclose(outputs["event_survival"], 1-F, atol=1e-6):
            raise ValueError(f"{name}: inconsistent survival probabilities")
        if "event_time_samples" in outputs:
            times = np.asarray(outputs["event_time_samples"])
            if times.ndim != 2 or times.shape[0] != n or np.isnan(times).any() or (times < 0).any():
                raise ValueError(f"{name}: invalid pathwise event times")
            if not np.allclose(np.isinf(times).mean(1), tail, atol=1e-6):
                raise ValueError(f"{name}: pathwise no-hit samples disagree with tail probability")


def validate_oof_rows(raw, samples, specimen_ids, repeats, folds, models):
    expected = set(specimen_ids)
    if set(raw.model) != set(models) or set(samples.model) != set(models):
        raise ValueError("The model list in the OOF report is incomplete")
    if set(raw["repeat"]) != set(range(repeats)):
        raise ValueError("Missing CV repeats")
    if raw.duplicated(["model", "fold", "repeat"]).any():
        raise ValueError("Duplicate fold results")
    if not raw.groupby("model").size().eq(folds * repeats).all():
        raise ValueError("Missing model/fold evaluations")
    if samples.duplicated(["model", "repeat", "specimen_id"]).any():
        raise ValueError("A specimen has multiple OOF predictions in one repeat")
    for _, rows in samples.groupby(["model", "repeat"]):
        if set(rows.specimen_id) != expected:
            raise ValueError("OOF predictions omit specimens or contain a different cohort")
