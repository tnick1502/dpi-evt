"""Dataset splitting, normalized benchmark tensors and mini-batches.

Select a stratified benchmark subset, form train/validation/test partitions and
assemble scaled tensors. Stratification starts from soil type, loading mode and
event label; sparse combinations are progressively merged.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dpi_evt.config import ExperimentConfig

__all__ = [
    "safe_strata",
    "stratified_subset_indices",
    "make_benchmark_splits",
    "make_grouped_cv_folds",
    "make_loo_object_folds",
    "prepare_benchmark_dataset",
    "iterate_minibatches",
    "audit_horizon_negatives",
]


def audit_horizon_negatives(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    cycles: np.ndarray,
    liq_label: np.ndarray,
    threshold: float = 0.95,
    early_window: tuple = (400.0, 500.0),
    rise_eps: float = 0.03,
) -> Dict[str, float]:
    """Check later pressure growth after an early flat PPR window.

    Select records without an event and with a range below rise_eps over
    [early_lo, early_hi]. Count later crossings of 0.95 and increases exceeding
    rise_eps above the early level, separating absence of an event from continued
    pressure accumulation.

    :return: n_early_flat, n_later_cross, n_later_rise and frac_later_rise
    """
    lo, hi = float(early_window[0]), float(early_window[1])
    n_flat = n_cross = n_rise = 0
    n_final_liq = 0
    for i in range(len(liq_label)):
        v = valid_mask[i] > 0
        in_win = v & (cycles[i] >= lo) & (cycles[i] <= hi)
        after = v & (cycles[i] > hi)
        through_win = v & (cycles[i] <= hi)
        if int(in_win.sum()) < 2 or int(after.sum()) < 2:
            continue
        # Select using observations through hi only.
        # The final liq_label does not select early flat records.
        if through_win.any() and float(r_obs[i, through_win].max()) >= threshold:
            continue
        win_vals = r_obs[i, in_win]
        if float(win_vals.max() - win_vals.min()) >= rise_eps: # Skip records that are not flat in the window.
            continue
        n_flat += 1
        n_final_liq += int(liq_label[i] >= 0.5)
        win_level = float(win_vals.mean())
        after_vals = r_obs[i, after]
        if float(after_vals.max()) >= threshold:
            n_cross += 1
        if float(after_vals.max() - win_level) >= rise_eps:
            n_rise += 1
    return {"n_early_flat": float(n_flat), "n_later_cross": float(n_cross),
            "n_later_rise": float(n_rise),
            "n_final_liq": float(n_final_liq),
            "frac_later_rise": float(n_rise / max(n_flat, 1))}


def safe_strata(meta: pd.DataFrame, fine_columns: List[str]) -> np.ndarray:
    """Build stratum labels that support stratified splitting.

    Start with fine_columns; merge singleton strata into loading-mode/event strata,
    then event-only strata if needed.

    :param meta: dataset metadata
    :param fine_columns: columns defining the finest strata
    :return: string stratum labels of length len(meta)
    """
    fine = meta[fine_columns].astype(str).agg("|".join, axis=1)
    fine_counts = fine.value_counts()
    if fine_counts.min() >= 2:
        return fine.to_numpy()

    medium = meta[["load_mode", "liq_label"]].astype(str).agg("|".join, axis=1)
    medium = fine.where(fine.map(fine_counts) >= 2, medium)
    medium_counts = medium.value_counts()
    if medium_counts.min() >= 2:
        return medium.to_numpy()

    # Merge into event-only strata.
    coarse = meta["liq_label"].astype(str)
    coarse_counts = coarse.value_counts()
    if coarse_counts.min() >= 2:
        return coarse.to_numpy()

    # If the minority class has fewer than two records,
    # use one stratum and split without stratification.
    return np.zeros(len(meta), dtype=int)


def stratified_subset_indices(meta: pd.DataFrame, subset_size: int, seed: int) -> np.ndarray:
    """Select a stratified subset of dataset indices.

    :param meta: dataset metadata
    :param subset_size: requested size, bounded by the dataset size
    :param seed: selection seed
    :return: sorted absolute indices
    """
    subset_size = min(subset_size, len(meta))
    idx = np.arange(len(meta))
    if subset_size >= len(meta): # The requested subset covers the full dataset.
        return idx
    strata = safe_strata(meta, ["soil_type", "load_mode", "liq_label"])
    keep_idx, _ = train_test_split(idx, train_size=subset_size, stratify=strata, random_state=seed)
    return np.sort(keep_idx)


def make_benchmark_splits(meta: pd.DataFrame, subset_size: int, seed: int, config: ExperimentConfig) -> Dict[str, np.ndarray]:
    """Select a benchmark subset and form train/validation/test partitions.

    Use the configured benchmark_train_fraction and benchmark_val_fraction.

    :param meta: full metadata table
    :param subset_size: benchmark size
    :param seed: splitting seed
    :param config: experiment settings
    :return: absolute benchmark_idx and benchmark-relative train_rel, val_rel, test_rel
    """
    benchmark_idx = stratified_subset_indices(meta, subset_size, seed)
    benchmark_meta = meta.iloc[benchmark_idx].reset_index(drop=True)
    benchmark_rel = np.arange(len(benchmark_idx))

    # Keep objects/sites separate across training, validation and test.
    # Specimens from one site remain together. Where possible,
    # include CRR and non-CRR objects in validation/test
    # to support CRR and binary-risk diagnostics.
    if getattr(config, "group_split_by_object", False) and "object" in benchmark_meta.columns:
        gcol = _group_col(benchmark_meta) # Group by site_id when available.
        objects = benchmark_meta[gcol].to_numpy()
        obj_stats = (
            benchmark_meta.groupby(gcol)
            .agg(
                n=(gcol, "size"),
                liq_rate=("liq_label", "mean"),
                has_crr=("has_measured_crr", "max") if "has_measured_crr" in benchmark_meta.columns else ("liq_label", "min"),
            )
            .reset_index().rename(columns={gcol: "object"})
        )
        uniq = np.array(sorted(obj_stats["object"].tolist()))
        rng = np.random.default_rng(seed)
        perm = list(uniq[rng.permutation(len(uniq))])
        n_obj = len(perm)
        test_fraction = max(0.0, 1.0 - config.benchmark_train_fraction - config.benchmark_val_fraction)
        n_te = max(1, int(round(test_fraction * n_obj)))
        if n_obj >= 6:
            n_te = max(2, n_te)
        n_va = max(1, int(round(config.benchmark_val_fraction * n_obj)))
        n_va = min(n_va, max(1, n_obj - n_te - 1))
        n_tr = max(1, n_obj - n_va - n_te)

        stats = obj_stats.set_index("object")

        def take_balanced(pool: List[str], k: int) -> List[str]:
            """Select k objects while retaining CRR/non-CRR coverage when possible."""
            chosen: List[str] = []
            crr = [o for o in pool if bool(stats.loc[o, "has_crr"])]
            non = [o for o in pool if not bool(stats.loc[o, "has_crr"])]
            if k >= 2 and crr and non:
                chosen.extend([crr[0], non[0]])
            elif crr:
                chosen.append(crr[0])
            elif non:
                chosen.append(non[0])
            for o in pool:
                if len(chosen) >= k:
                    break
                if o not in chosen:
                    chosen.append(o)
            return chosen[:k]

        te_obj = set(take_balanced(perm, n_te))
        remaining = [o for o in perm if o not in te_obj]
        va_obj = set(take_balanced(remaining, n_va))
        tr_obj = set(o for o in remaining if o not in va_obj)
        if len(tr_obj) < n_tr:
            deficit = n_tr - len(tr_obj)
            move = list(va_obj)[:deficit]
            tr_obj.update(move)
            va_obj.difference_update(move)
        train_rel = benchmark_rel[np.isin(objects, list(tr_obj))]
        val_rel = np.sort(benchmark_rel[np.isin(objects, list(va_obj))])
        test_rel = np.sort(benchmark_rel[np.isin(objects, list(te_obj))])
        return {"benchmark_idx": benchmark_idx, "train_rel": np.sort(train_rel),
                "val_rel": val_rel, "test_rel": test_rel}

    strata = safe_strata(benchmark_meta, ["soil_type", "load_mode", "liq_label"])

    train_rel, temp_rel = train_test_split(
        benchmark_rel,
        train_size=config.benchmark_train_fraction,
        stratify=strata,
        random_state=seed,
    )
    temp_meta = benchmark_meta.iloc[temp_rel].reset_index(drop=True)
    temp_strata = safe_strata(temp_meta, ["soil_type", "load_mode", "liq_label"])
    val_fraction_relative = config.benchmark_val_fraction / (1.0 - config.benchmark_train_fraction)
    val_rel_local, test_rel_local = train_test_split(
        np.arange(len(temp_rel)),
        train_size=val_fraction_relative,
        stratify=temp_strata,
        random_state=seed,
    )
    val_rel = np.sort(temp_rel[val_rel_local])
    test_rel = np.sort(temp_rel[test_rel_local])

    return {
        "benchmark_idx": benchmark_idx,
        "train_rel": np.sort(train_rel),
        "val_rel": val_rel,
        "test_rel": test_rel,
    }


def _group_col(meta: pd.DataFrame) -> str:
    """Use canonical site_id for grouped splitting, or object when site_id is absent."""
    return "site_id" if "site_id" in meta.columns else "object"


def _object_level_table(benchmark_meta: pd.DataFrame, uniq: np.ndarray,
                        group_col: str = "object") -> pd.DataFrame:
    """Summarize group size, event fraction and CRR availability for fold allocation."""
    g = benchmark_meta.groupby(group_col)
    n = g.size().reindex(uniq).fillna(0).astype(int)
    if "has_measured_crr" in benchmark_meta.columns:
        crr = g["has_measured_crr"].max().reindex(uniq).fillna(0).astype(int)
    else:
        crr = pd.Series(0, index=uniq, dtype=int)
    if "liq_label" in benchmark_meta.columns:
        liq = g["liq_label"].mean().reindex(uniq).fillna(0.0)
    else:
        liq = pd.Series(0.0, index=uniq)
    return pd.DataFrame({"object": uniq, "n": n.values, "has_crr": crr.values,
                         "liq_bin": (liq.values >= 0.5).astype(int)}).set_index("object")


def _pick_balanced_val(pool: List[str], k: int, stats: pd.DataFrame, rng) -> List[str]:
    """Select k validation objects with class and CRR coverage when available.

    Prefer objects with different majority labels and include a measured-CRR object
    when the pool contains one.
    """
    if not pool:
        return []
    perm = list(np.array(pool)[rng.permutation(len(pool))])
    pos = [o for o in perm if int(stats.loc[o, "liq_bin"]) == 1]
    neg = [o for o in perm if int(stats.loc[o, "liq_bin"]) == 0]
    chosen: List[str] = []
    if pos:
        chosen.append(pos[0])
    if neg:
        chosen.append(neg[0])
    # Include a CRR object for validation diagnostics when available.
    if not any(int(stats.loc[o, "has_crr"]) for o in chosen):
        crr = [o for o in perm if int(stats.loc[o, "has_crr"]) and o not in chosen]
        if crr:
            chosen.append(crr[0])
    for o in perm: # Fill the remaining slots up to k.
        if len(chosen) >= max(k, 2):
            break
        if o not in chosen:
            chosen.append(o)
    return chosen


def _balanced_object_folds(uniq: np.ndarray, stats: pd.DataFrame, n_splits: int, rng) -> Dict[str, int]:
    """Allocate objects to folds, prioritizing CRR coverage.

    Assign CRR objects first by round-robin, then the remaining objects. Alternate
    event-majority and non-event-majority objects within each group.
    """
    def interleave(objs: List[str]) -> List[str]:
        objs = list(objs); rng.shuffle(objs)
        pos = [o for o in objs if int(stats.loc[o, "liq_bin"]) == 1]
        neg = [o for o in objs if int(stats.loc[o, "liq_bin"]) == 0]
        out: List[str] = []
        while pos or neg:
            if pos:
                out.append(pos.pop())
            if neg:
                out.append(neg.pop())
        return out

    crr = [o for o in uniq if int(stats.loc[o, "has_crr"]) == 1]
    non = [o for o in uniq if int(stats.loc[o, "has_crr"]) == 0]
    assign: Dict[str, int] = {}
    cursor = int(rng.integers(0, n_splits))
    for group in (interleave(crr), interleave(non)):
        for o in group:
            assign[o] = cursor % n_splits
            cursor += 1
    return assign


def make_grouped_cv_folds(
    meta: pd.DataFrame,
    subset_size: int,
    seed: int,
    config: ExperimentConfig,
    n_splits: int = 5,
    val_objects: int = 2,
    n_repeats: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Construct repeated grouped K-fold splits by site.

    Each object is in one test fold per repeat. Allocation considers CRR availability
    and majority label. Validation objects favor class and CRR coverage when available.
    Repeat the scheme with different random seeds.

    :param n_splits: fold count
    :param val_objects: minimum validation-object count, at least 2
    :param n_repeats: cross-validation repeat count
    :return: dictionaries with repeat, fold, benchmark_idx, train_rel, val_rel, test_rel
    """
    benchmark_idx = stratified_subset_indices(meta, subset_size, seed)
    benchmark_meta = meta.iloc[benchmark_idx].reset_index(drop=True)
    rel = np.arange(len(benchmark_idx))
    if "object" not in benchmark_meta.columns:
        raise ValueError("make_grouped_cv_folds requires an 'object' column in meta")
    gcol = _group_col(benchmark_meta) # Group by site_id when available.
    objects = benchmark_meta[gcol].to_numpy()
    uniq = np.array(sorted(pd.unique(objects)))
    n_splits = int(max(2, min(n_splits, len(uniq))))
    stats = _object_level_table(benchmark_meta, uniq, gcol)

    folds: List[Dict[str, np.ndarray]] = []
    for rep in range(int(max(1, n_repeats))):
        rng = np.random.default_rng(seed + 1009 * rep)
        assign = _balanced_object_folds(uniq, stats, n_splits, rng)
        for k in range(n_splits):
            te_obj = [o for o in uniq if assign[o] == k]
            rest = [o for o in uniq if assign[o] != k]
            va_obj = set(_pick_balanced_val(rest, val_objects, stats, rng))
            tr_obj = [o for o in rest if o not in va_obj]
            folds.append({
                "repeat": rep,
                "fold": k,
                "benchmark_idx": benchmark_idx,
                "train_rel": np.sort(rel[np.isin(objects, tr_obj)]),
                "val_rel": np.sort(rel[np.isin(objects, list(va_obj))]),
                "test_rel": np.sort(rel[np.isin(objects, te_obj)]),
            })
    return folds


def make_loo_object_folds(
    meta: pd.DataFrame,
    subset_size: int,
    seed: int,
    config: ExperimentConfig,
    val_objects: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Construct leave-one-site-out splits.

    Each site_id is the sole test site in turn. Select validation objects from the
    remaining sites with CRR priority; use the rest for training.
    """
    benchmark_idx = stratified_subset_indices(meta, subset_size, seed)
    benchmark_meta = meta.iloc[benchmark_idx].reset_index(drop=True)
    rel = np.arange(len(benchmark_idx))
    if "object" not in benchmark_meta.columns:
        raise ValueError("make_loo_object_folds requires an 'object' column in meta")
    gcol = _group_col(benchmark_meta) # Leave one site_id out when available.
    objects = benchmark_meta[gcol].to_numpy()
    uniq = np.array(sorted(pd.unique(objects)))
    stats = _object_level_table(benchmark_meta, uniq, gcol)
    rng = np.random.default_rng(seed)
    folds: List[Dict[str, np.ndarray]] = []
    for k, te_o in enumerate(uniq):
        rest = [o for o in uniq if o != te_o]
        va_obj = set(_pick_balanced_val(rest, val_objects, stats, rng))
        tr_obj = [o for o in rest if o not in va_obj]
        folds.append({
            "repeat": 0,
            "fold": k,
            "test_object": te_o,
            "benchmark_idx": benchmark_idx,
            "train_rel": np.sort(rel[np.isin(objects, tr_obj)]),
            "val_rel": np.sort(rel[np.isin(objects, list(va_obj))]),
            "test_rel": np.sort(rel[np.isin(objects, [te_o])]),
        })
    return folds


def prepare_benchmark_dataset(
    population_dict: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
    precomputed_split: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    """Assemble scaled benchmark tensors and their partitions.

    Fit StandardScaler on training static/prefix features and scale sequence inputs
    by channel. Compute log-normalized n_liq_norm and trigger_zone, then assemble
    train/validation/test dictionaries.

    :param population_dict: prepared dataset dictionary
    :param config: normalization and reference-cycle settings
    :param device: target device for mini-batches
    :return: partitions, metadata, fitted scalers and feature names
    """
    # Use the stored split unless grouped splitting is requested.
    # In grouped mode rebuild train/validation/test partitions
    # so no object is shared between training and test.
    if precomputed_split is not None:
        # Explicit fold from make_grouped_cv_folds or make_loo_object_folds.
        bsplit = precomputed_split
    elif getattr(config, "group_split_by_object", False):
        bsplit = make_benchmark_splits(population_dict["meta"],
                                       min(config.benchmark_subset, len(population_dict["meta"])),
                                       config.seed, config)
    else:
        bsplit = population_dict["benchmark"]
    benchmark_idx = bsplit["benchmark_idx"]
    bench_meta = population_dict["meta"].iloc[benchmark_idx].reset_index(drop=True)

    # Required arrays of observed quantities.
    static_raw = population_dict["static_features"][benchmark_idx]
    prefix_raw = population_dict["prefix_summary"][benchmark_idx]
    seq_raw = population_dict["seq_inputs"][benchmark_idx]
    cycles = population_dict["cycles"][benchmark_idx]
    delta_cycles = population_dict["delta_cycles"][benchmark_idx]
    csr = population_dict["csr"][benchmark_idx]
    r_obs = population_dict["r_obs"][benchmark_idx]
    valid_mask = population_dict["valid_mask"][benchmark_idx]
    prefix_mask = population_dict["prefix_mask"][benchmark_idx]
    prefix_obs = population_dict["prefix_obs"][benchmark_idx]
    liq_label = population_dict["liq_label"][benchmark_idx]
    n_liq_true = population_dict["n_liq_true"][benchmark_idx]

    train_rel = bsplit["train_rel"]
    val_rel = bsplit["val_rel"]
    test_rel = bsplit["test_rel"]

    static_scaler = StandardScaler().fit(static_raw[train_rel])
    prefix_scaler = StandardScaler().fit(prefix_raw[train_rel])
    seq_train = seq_raw[train_rel].reshape(-1, seq_raw.shape[-1])
    seq_mean = seq_train.mean(axis=0)
    seq_std = seq_train.std(axis=0) + 1e-6

    static_scaled = static_scaler.transform(static_raw).astype(np.float32)
    prefix_scaled = prefix_scaler.transform(prefix_raw).astype(np.float32)
    seq_scaled = ((seq_raw - seq_mean[None, None, :]) / seq_std[None, None, :]).astype(np.float32)
    # Define events within max_cycle_reference, 3000 cycles.
    # Events after H receive by-H label 0 and censoring at H.
    # Labels and event times therefore use the same observation window.
    liq_label = liq_label.copy().astype(np.float32)
    _late_event = (n_liq_true > config.max_cycle_reference) & (liq_label > 0.5)
    liq_label[_late_event] = 0.0
    # Within H, retain exact event times and censor non-events at the horizon.
    n_liq_target = np.minimum(n_liq_true, config.max_cycle_reference).astype(np.float32)
    n_liq_norm = (np.log1p(n_liq_target) / np.log1p(config.max_cycle_reference)).astype(np.float32)

    # Keep three observation concepts separate:
    # risk_label_observed: whether the binary by-H outcome is known;
    # nliq_censor_valid: whether an exact time or lower bound C is available;
    # regime_*: event/completed/unfinished record status.
    _H = float(config.max_cycle_reference)
    _N0 = float(getattr(config, "prefix_landmark_cycles", 0.0))
    _reached_horizon = (n_liq_true >= _H - 1e-6) # Observation reached H, or an event occurred within H.
    _is_liq = liq_label > 0.5
    risk_label_observed = (_is_liq | ((~_is_liq) & _reached_horizon)).astype(np.float32)
    # A non-event record observed after the landmark supplies right censoring
    # T>C even when C<H; its binary by-H outcome can remain unknown.
    landmark_eligible = (n_liq_true > _N0 + 1e-6) if _N0 > 0 else np.ones_like(_is_liq, bool)
    nliq_censor_valid = landmark_eligible.astype(np.float32)

    # Trigger-suppression states use the observation window:
    # stable reached H without an event; unfinished stopped before H
    # with an unknown by-H outcome, as reflected in the risk mask.
    regime_liq = _is_liq.astype(np.float32)
    regime_stable = ((~_is_liq) & _reached_horizon).astype(np.float32)
    regime_unfinished = ((~_is_liq) & (~_reached_horizon)).astype(np.float32)

    # n_liq_observed is an event-time observation field.
    # Binary-risk consumers use risk_label_observed separately.
    n_liq_observed = nliq_censor_valid.copy()
    continuation_mask = valid_mask * (1.0 - np.minimum(prefix_mask, 1.0))

    benchmark_arrays = {
        "static": static_scaled,
        "static_raw": static_raw.astype(np.float32),
        "prefix_summary": prefix_scaled,
        "prefix_summary_raw": prefix_raw.astype(np.float32),
        "seq_in": seq_scaled,
        "seq_in_raw": seq_raw.astype(np.float32),
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles.astype(np.float32),
        "csr": csr.astype(np.float32),
        "r_obs": r_obs.astype(np.float32),
        "mask": valid_mask.astype(np.float32),
        "prefix_mask": prefix_mask.astype(np.float32),
        "prefix_obs": prefix_obs.astype(np.float32),
        "label": liq_label.astype(np.float32),
        "n_liq_true": n_liq_target.astype(np.float32),
        "n_liq_raw": n_liq_true.astype(np.float32),
        "n_liq_norm": n_liq_norm.astype(np.float32),
        "n_liq_observed": n_liq_observed.astype(np.float32),
        "trajectory_mask": valid_mask.astype(np.float32),
        "continuation_mask": continuation_mask.astype(np.float32),
        "nliq_censor_valid": nliq_censor_valid.astype(np.float32),
        "risk_label_observed": risk_label_observed.astype(np.float32),
        "regime_liq": regime_liq,
        "regime_stable": regime_stable,
        "regime_unfinished": regime_unfinished,
    }

    # Auxiliary targets derived from measured PPR and optional measured CRR;
    # optional latent diagnostics, when present, are retained separately
    # and excluded from laboratory training/evaluation targets.
    optional_fields = {
        # Observed targets.
        "g_obs": "g_obs",
        "risk_proxy": "risk_proxy",
        "r_causal": "r_causal",
        "crr_obs": "crr_obs",
        "crr_obs_mask": "crr_obs_mask",
        # Optional latent diagnostics.
        "r_true": "r_true",
        "z_true": "z_true",
        "g_true": "g_true",
        "uncertainty_proxy": "uncertainty_proxy",
        "crr_mix_true": "crr_mix",
    }
    for split_key, pop_key in optional_fields.items():
        if pop_key in population_dict and population_dict[pop_key] is not None:
            benchmark_arrays[split_key] = population_dict[pop_key][benchmark_idx].astype(np.float32)
    if "g_true" in benchmark_arrays:
        benchmark_arrays["trigger_zone"] = (benchmark_arrays["g_true"] > 0.70).astype(np.float32)

    def make_split(rel_idx: np.ndarray) -> Dict[str, object]:
        """Build one partition from benchmark arrays using relative indices."""
        split = {key: torch.from_numpy(value[rel_idx]) for key, value in benchmark_arrays.items()}
        split["meta"] = bench_meta.iloc[rel_idx].reset_index(drop=True)
        split["indices"] = rel_idx
        return split

    return {
        "train": make_split(train_rel),
        "val": make_split(val_rel),
        "test": make_split(test_rel),
        "meta": bench_meta,
        "scalers": {
            "static": static_scaler,
            "prefix": prefix_scaler,
            "seq_mean": seq_mean.astype(np.float32),
            "seq_std": seq_std.astype(np.float32),
        },
        "feature_names": {
            "static": population_dict["static_feature_names"],
            "prefix": population_dict["prefix_summary_names"],
            "seq": population_dict["seq_feature_names"],
        },
    }


def iterate_minibatches(
    split: Dict[str, object],
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, object]]:
    """Yield mini-batches with tensors on the target device.

    Slice all tensor fields and the corresponding metadata rows together.

    :param split: partition from prepare_benchmark_dataset
    :param batch_size: mini-batch size
    :param device: target tensor device
    :param shuffle: whether to shuffle observations
    :param seed: shuffling seed
    :yield: tensor batch and matching meta rows
    """
    idx = np.arange(split["static"].shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    tensor_keys = [key for key, value in split.items() if isinstance(value, torch.Tensor)]
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        batch = {key: split[key][batch_idx].to(device) for key in tensor_keys}
        batch["meta"] = split["meta"].iloc[batch_idx].reset_index(drop=True)
        yield batch
