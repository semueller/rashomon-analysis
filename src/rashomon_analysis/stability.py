"""Stability of local explanations across repeated explainer runs (seeds) and budgets.

Input: stored explanation variants of one run, e.g. ks_shap with n_samples = 100 ... 8000 and seeds 0, 1, 2, each its
own explanation folder (rashomon_zoo.explain(..., out_name=...)), all for the same models and samples.
Measures per sample, between two runs u, v of the same explainer setting on the same model (all seed pairs):
  topk set agreement   |topk(|u|) ∩ topk(|v|)| / k               (= 1 - feature disagreement of rashomon.py)
  topk rank agreement  1 if the k largest |u| and |v| are the same features in the same order, else 0
  binary flip rate     share of features whose cfire.binarise_threshold entry differs (CFIRE's binarisation:
                       row / max |row|, positive scores >= threshold)
  euclid               ||u - v||_2
  cosine distance      1 - <u, v> / (||u|| ||v||)
Over all S seeds of a budget at once (numeric_noise), per sample, with mean explanation ā and per-feature
standard deviation σ_j over seeds (ddof = 1):
  relative noise            sqrt(sum_j σ_j^2) / ||ā||_2   (RMS deviation of one run from the mean, relative to the
                            explanation's size; Monte Carlo error: expected to fall like n_samples^(-1/2))
  max relative entry noise  max_j σ_j / ||ā||_2
Model separation (separation): explainer noise compared with the differences between two models of the run, on the
samples where both predict the same class:
  separation ratio  sqrt(sum_j σ_j,m^2 + sum_j σ_j,m'^2) / ||ā_m^ref - ā_m'^ref||_2
                    (RMS distance between a run of m and a run of m' that is due to the explainer, over the distance of
                    the models' reference explanations: means over seeds at the reference budget); < 1: model
                    differences exceed explainer noise
  relative model distance   ||ā_m^ref - ā_m'^ref|| / mean(||ā_m^ref||, ||ā_m'^ref||)  (links the two: separation
                            ratio ≈ sqrt(2) * relative noise / relative model distance)
Classifier separation (classifier_separation): held-out accuracy of a logistic regression telling model a's
explanations from model b's, against the same classifier telling two halves of model a's seed runs apart (explainer
noise only); after Watson et al. (2021), who had no explainer-noise baseline.
and, where the explainer stores it (<split>_info.npz), the absolute completeness gap |f(x) - f(b) - sum(u)| per run.
With a reference variant (e.g. an exact or high-budget explanation), every run is also compared with it
(comparison "reference"). Values are averaged over samples per model; rows keep the model, so the budget at which
an explainer settles can be read per model. No convergence threshold is applied here.
"""
from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from rashomon_zoo.store import Run

K_MEASURES = ("topk set agreement", "topk rank agreement")
MEASURES = K_MEASURES + ("binary flip rate", "euclid", "cosine distance")


def _topk_idx(a: np.ndarray, k: int) -> np.ndarray:
    """indices of the k largest |a| per row, in decreasing order (ties: lower feature index first)"""
    return np.argsort(-np.abs(a), axis=-1, kind="stable")[..., :k]


def compare(u: np.ndarray, v: np.ndarray, ks: Sequence[int], threshold: float = 0.01) -> dict:
    """measures between two explanation arrays of shape (n, d), per sample: {measure: (n,)} for the k-free
    measures and {(measure, k): (n,)} for the top-k measures"""
    from cfire.binarise import binarise_threshold
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    out = {}
    for k in ks:
        tu, tv = _topk_idx(u, k), _topk_idx(v, k)
        inter = np.array([len(set(a) & set(b)) for a, b in zip(tu, tv)])
        out[("topk set agreement", k)] = inter / k
        out[("topk rank agreement", k)] = (tu == tv).all(1).astype(float)
    out["binary flip rate"] = (binarise_threshold(u, threshold) != binarise_threshold(v, threshold)).mean(1)
    out["euclid"] = np.linalg.norm(u - v, axis=1)
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cosine distance"] = 1 - (u * v).sum(1) / (nu * nv)
    return out


def numeric_noise(E: np.ndarray) -> dict:
    """E: (S, n, d) explanations of one model and budget from S >= 2 explainer seeds. Returns per sample (n,):
    "relative noise" sqrt(sum_j σ_j^2) / ||ā|| and "max relative entry noise" max_j σ_j / ||ā|| (σ over seeds,
    ddof = 1; ā mean over seeds; NaN where ā = 0)."""
    E = np.asarray(E, dtype=np.float64)
    if E.shape[0] < 2:
        raise ValueError("numeric_noise needs at least two seeds")
    sd = E.std(0, ddof=1)
    size = np.linalg.norm(E.mean(0), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.where(size > 0, size, np.nan)
        return {"relative noise": np.sqrt((sd ** 2).sum(-1)) / size, "max relative entry noise": sd.max(-1) / size}


def _variant(run: Run, name: str, split: str):
    m = run.explainer_manifest(name)
    if m["split"] != split:
        raise ValueError(f"{name}: stored for split {m['split']}, not {split}")
    gap = None
    info = run.dir / "explanations" / name / f"{split}_info.npz"
    if info.exists():
        with np.load(info) as z:
            gap = np.abs(z["gap"]) if "gap" in z.files else None
    return np.asarray(run.explanations(name, split)), m["model_ids"], m["sample_indices"], gap


def budget_study(run: Run, variants: Mapping, *, split: str, ks: Sequence[int], threshold: float = 0.01,
                 reference: str | None = None, models: Sequence[int] | None = None) -> pd.DataFrame:
    """variants: {budget: [explanation names of that budget, one per explainer seed]}. Returns one row per
    (model, budget, comparison, measure, k) with the mean, 10 % and 90 % quantiles over samples (and seed pairs):
    comparison "seeds" (all pairs of runs of the same budget), "reference" (each run vs `reference`) and
    "completeness" (|gap| per run, where stored). models: model ids to include (default: all explained)."""
    loaded = {b: [_variant(run, n, split) for n in names] for b, names in variants.items()}
    first = next(iter(loaded.values()))[0]
    idx, ids = first[2], first[1]
    ref = _variant(run, reference, split) if reference else None
    for b, runs in loaded.items():
        for (_, mids, sidx, _), n in zip(runs, variants[b]):
            if sidx != idx:
                raise ValueError(f"{n}: other samples than {variants[next(iter(variants))][0]}")
            missing = set(models or ids) - set(mids)
            if missing:
                raise ValueError(f"{n}: models {sorted(missing)} not explained")
    if ref is not None and (ref[2] != idx or set(models or ids) - set(ref[1])):
        raise ValueError(f"reference {reference}: other samples or models")
    ids = list(models or ids)
    rows = []

    def add(mid, b, comparison, vals: dict):
        for key, v in vals.items():
            measure, k = key if isinstance(key, tuple) else (key, None)
            v = np.concatenate(v) if isinstance(v, list) else v
            rows.append({"model_id": mid, "budget": b, "comparison": comparison, "measure": measure, "k": k,
                         "mean": float(np.nanmean(v)), "q10": float(np.nanquantile(v, 0.1)),
                         "q90": float(np.nanquantile(v, 0.9)), "n_values": int(np.isfinite(v).sum())})

    for b, runs in loaded.items():
        for mid in ids:
            E = [r[0][r[1].index(mid)] for r in runs]
            if len(E) > 1:
                acc = {}
                for i, j in itertools.combinations(range(len(E)), 2):
                    for key, v in compare(E[i], E[j], ks, threshold).items():
                        acc.setdefault(key, []).append(v)
                add(mid, b, "seeds", acc)
                add(mid, b, "seeds", numeric_noise(np.stack(E)))
            if ref is not None:
                R = ref[0][ref[1].index(mid)]
                acc = {}
                for e in E:
                    for key, v in compare(e, R, ks, threshold).items():
                        acc.setdefault(key, []).append(v)
                add(mid, b, "reference", acc)
            gaps = [r[3][r[1].index(mid)] for r in runs if r[3] is not None]
            if gaps:
                add(mid, b, "completeness", {"abs gap": gaps})
    return pd.DataFrame(rows)


def separation(run: Run, variants: Mapping, *, split: str, models: Sequence[int] | None = None,
               reference_budget=None) -> pd.DataFrame:
    """Explainer noise against model differences, for every pair of models and every budget.

    variants: {budget: [names, one per explainer seed]} as in budget_study. Reference explanations: means over the
    seeds at `reference_budget` (default: the largest budget). Samples: those where both models predict the same
    class (explanations of different classes are not compared). Returns one row per (model_a, model_b, budget,
    measure) with mean, median, 10 % and 90 % quantiles over samples: "separation ratio" and "relative model
    distance" (see module docstring)."""
    loaded = {b: [_variant(run, n, split) for n in names] for b, names in variants.items()}
    first = next(iter(loaded.values()))[0]
    idx, ids = first[2], list(models or first[1])
    for b, runs in loaded.items():
        for (_, mids, sidx, _), n in zip(runs, variants[b]):
            if sidx != idx or set(ids) - set(mids):
                raise ValueError(f"{n}: other samples or models missing")
    ref_b = max(variants) if reference_budget is None else reference_budget
    pred = np.asarray(run.outputs(split))[:, idx].argmax(-1)  # (all models, n)

    def stack(b, mid):
        return np.stack([r[0][r[1].index(mid)] for r in loaded[b]]).astype(np.float64)

    ref = {m: stack(ref_b, m).mean(0) for m in ids}
    noise2 = {(b, m): (stack(b, m).std(0, ddof=1) ** 2).sum(-1) for b in loaded for m in ids}
    rows = []
    for ma, mb in itertools.combinations(ids, 2):
        same = pred[ma] == pred[mb]
        dist = np.linalg.norm(ref[ma] - ref[mb], axis=-1)
        size = (np.linalg.norm(ref[ma], axis=-1) + np.linalg.norm(ref[mb], axis=-1)) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(same, dist / size, np.nan)
            for b in loaded:
                ratio = np.where(same, np.sqrt(noise2[(b, ma)] + noise2[(b, mb)]) / dist, np.nan)
                for measure, v in (("separation ratio", ratio), ("relative model distance", rel)):
                    v = v[np.isfinite(v)]
                    rows.append({"model_a": ma, "model_b": mb, "budget": b, "measure": measure,
                                 "reference_budget": ref_b, "n_values": len(v),
                                 "mean": float(v.mean()) if len(v) else np.nan,
                                 "median": float(np.median(v)) if len(v) else np.nan,
                                 "q10": float(np.quantile(v, 0.1)) if len(v) else np.nan,
                                 "q90": float(np.quantile(v, 0.9)) if len(v) else np.nan})
    return pd.DataFrame(rows)


def classifier_separation(run: Run, variants: Mapping, *, split: str, models: Sequence[int] | None = None,
                          folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Can a classifier tell which model an explanation came from, beyond explainer noise? (after Watson et al.
    2021, with an explainer-noise baseline and held-out evaluation)

    For every budget, the S seed runs per model are split into halves A (first S // 2) and B (next S // 2).
    - "model pair" (a, b): explanations of a (seeds A) vs b (seeds A), on the samples where both predict the same
      class; label = model.
    - "explainer noise" (a): explanations of a, seeds A vs seeds B; label = half. Only explainer noise separates.
    Classifier: standardisation + logistic regression; accuracy on held-out folds (`folds`-fold, grouped by sample:
    all runs of a sample are in the same fold). Returns one row per (comparison, model_a, model_b, budget) with the
    held-out accuracy (0.5: not separable) and the number of samples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    loaded = {b: [_variant(run, n, split) for n in names] for b, names in variants.items()}
    first = next(iter(loaded.values()))[0]
    idx, ids = first[2], list(models or first[1])
    S = min(len(v) for v in loaded.values())
    h = S // 2
    if h < 1:
        raise ValueError("classifier_separation needs at least two seeds per budget")
    pred = np.asarray(run.outputs(split))[:, idx].argmax(-1)

    def runs(b, mid, part):  # (h, n, d)
        sel = loaded[b][:h] if part == "A" else loaded[b][h:2 * h]
        return np.stack([r[0][r[1].index(mid)] for r in sel]).astype(np.float64)

    def accuracy(P, Q, samples):
        P, Q = P[:, samples], Q[:, samples]
        X = np.concatenate([P.reshape(-1, P.shape[-1]), Q.reshape(-1, Q.shape[-1])])
        y = np.r_[np.zeros(P.shape[0] * P.shape[1]), np.ones(Q.shape[0] * Q.shape[1])]
        g = np.r_[np.tile(np.arange(P.shape[1]), P.shape[0]), np.tile(np.arange(Q.shape[1]), Q.shape[0])]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        k = min(folds, len(samples))
        return float(cross_val_score(clf, X, y, groups=g, cv=GroupKFold(k), scoring="accuracy").mean())

    rows = []
    all_samples = np.arange(len(idx))
    for b in loaded:
        for m in ids:
            rows.append({"comparison": "explainer noise", "model_a": m, "model_b": m, "budget": b,
                         "accuracy": accuracy(runs(b, m, "A"), runs(b, m, "B"), all_samples),
                         "n_samples": len(all_samples), "runs_per_model": h})
        for ma, mb in itertools.combinations(ids, 2):
            same = np.flatnonzero(pred[ma] == pred[mb])
            rows.append({"comparison": "model pair", "model_a": ma, "model_b": mb, "budget": b,
                         "accuracy": accuracy(runs(b, ma, "A"), runs(b, mb, "A"), same) if len(same) >= 2 else np.nan,
                         "n_samples": len(same), "runs_per_model": h})
    return pd.DataFrame(rows).assign(classifier="standardise + logistic regression", folds=folds,
                                     evaluation="held-out folds grouped by sample")


def required_budget(table: pd.DataFrame, target: float, *, measure: str = "relative noise",
                    comparison: str = "seeds") -> pd.DataFrame:
    """Budget each model needs to reach `measure` <= target, from a budget_study table.

    Per (explainer, model_id): log(measure) is interpolated linearly in log(budget) between the measured budgets;
    outside the measured range it is extrapolated with slope -1/2 (Monte Carlo noise falls like 1/sqrt(n)) from
    the nearest measured budget, and the row is flagged `extrapolated`. Returns explainer, model_id,
    required_budget, extrapolated."""
    t = table[(table.measure == measure) & (table.comparison == comparison)]
    if "explainer" not in t:
        t = t.assign(explainer="")
    rows = []
    for (e, mid), g in t.groupby(["explainer", "model_id"]):
        g = g.sort_values("budget")
        n, v = g.budget.to_numpy(float), g["mean"].to_numpy(float)
        below = np.flatnonzero(v <= target)
        if len(below) == 0:  # beyond the largest budget
            need, extra = n[-1] * (v[-1] / target) ** 2, True
        elif below[0] == 0:  # already below at the smallest budget
            need, extra = n[0] * (v[0] / target) ** 2, True
        else:
            i = below[0]
            slope = np.log(v[i] / v[i - 1]) / np.log(n[i] / n[i - 1])
            need, extra = n[i - 1] * (target / v[i - 1]) ** (1 / slope), False
        rows.append({"explainer": e, "model_id": mid, "required_budget": float(need), "extrapolated": bool(extra),
                     "target": target, "measure": measure})
    return pd.DataFrame(rows)


def converged_share(required: pd.DataFrame, budgets: Sequence[float]) -> pd.DataFrame:
    """share of models (per explainer) whose required budget is <= each budget (the empirical CDF of
    required_budget), for plotting 'share of the Rashomon set converged' against the budget"""
    rows = []
    for e, g in required.groupby("explainer"):
        need = g.required_budget.to_numpy()
        for b in budgets:
            rows.append({"explainer": e, "budget": float(b), "share": float((need <= b).mean()), "n_models": len(need)})
    return pd.DataFrame(rows)
