"""Rashomon analyses on stored runs: explanation diversity across models and method disagreement within a model.

Measures and procedure follow Müller et al. (2023, ECML PKDD):
- measures (per sample, then averaged over the comparison subset):
  feature disagreement  1 - |topk(|u|) ∩ topk(|v|)| / k
  sign disagreement     1 - |{i in topk(|u|) ∩ topk(|v|): u_i v_i >= 0}| / k
  euclid                ||u - v||_2
  euclid abs            || |u| - |v| ||_2
- comparison subset: "both_correct" (both models correct; default) or "same_prediction" (both predict the same class)
- Kendall tau between the per-model rank vectors of the method pairs, over all model pairs (p stored)
- distances are not normalised
The evaluation code of the 2023 paper (tag ecml2023) differs in the Euclidean distances, their normalisation and the
Kendall tau, so those values are not directly comparable.
Results and the settings (incl. the Rashomon criterion of the run and the zoo run it read) go to
<results_dir>/<task>/<model_class>/<run>/rashomon/<name>/; the zoo data are only read.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

from rashomon_zoo.store import Run, environment, read_json, write_json  # zoo storage layout = interface

MEASURES = ("feature disagreement", "sign disagreement", "euclid", "euclid abs")
# k for the top-k measures in Müller et al. (2023): AG News 11, Dry Bean 4, Breast Cancer 8, Ionosphere 8
PAPER_R_K = {"agnews": 11, "beans": 4, "breastcancer": 8, "ionosphere": 8}


# ---------------------------------------------------------------------------------------------------------------
def _topk(a: np.ndarray, k: int) -> np.ndarray:
    """indices of the k largest |a| per row (ties: numpy argsort order)"""
    return np.argsort(np.abs(a), axis=1)[:, -k:]


def distance(a: np.ndarray, b: np.ndarray, measure: str, k: int | None = None) -> np.ndarray:
    """per-sample distance between two explanation matrices (n, d)"""
    if measure in ("feature disagreement", "sign disagreement"):
        ta, tb = _topk(a, k), _topk(b, k)
        out = np.empty(len(a))
        for i, (x, y) in enumerate(zip(ta, tb)):
            common = np.intersect1d(x, y)
            if measure == "feature disagreement":
                out[i] = 1 - len(common) / k
            else:
                out[i] = 1 - np.sum(a[i, common] * b[i, common] >= 0) / k
        return out
    if measure == "euclid":
        return np.linalg.norm(a - b, axis=1)
    if measure == "euclid abs":
        return np.linalg.norm(np.abs(a) - np.abs(b), axis=1)
    raise KeyError(measure)


def _pair_value(a, b, measure, k):
    return float(distance(a, b, measure, k).mean())


def _subset(pred_a, pred_b, y, mode):
    if mode == "both_correct":
        return (pred_a == y) & (pred_b == y)
    if mode == "same_prediction":
        return pred_a == pred_b
    raise KeyError(mode)


# ---------------------------------------------------------------------------------------------------------------
def _load(run: Run, explainers, split, models):
    models_table = run.models()
    if models == "rashomon":
        models_table = models_table[models_table["in_rashomon_set"]]
    ids = models_table["model_id"].tolist()
    t = run.load_task()
    y = t.y[split]
    P = np.asarray(run.outputs(split))
    E, idx = {}, None
    for e in explainers:
        explainer_manifest = run.explainer_manifest(e)
        rows = [explainer_manifest["model_ids"].index(i) for i in ids]
        E[e] = np.asarray(run.explanations(e, split))[rows]
        if idx is None:
            idx = explainer_manifest["sample_indices"]
        elif explainer_manifest["sample_indices"] != idx:
            raise ValueError(f"explainers explain different samples: {e}")
    pred = P[np.asarray(ids)][:, idx].argmax(-1)
    return models_table.set_index("model_id").loc[ids], E, pred, y[idx], idx


def diversity(run: Run, explainer: str, measure: str, k: int | None = None, *, split: str,
              subset: str = "both_correct", models: str = "rashomon") -> pd.DataFrame:
    """same explainer, all pairs of models: mean distance over the comparison subset"""
    models_table, E, pred, y, _ = _load(run, [explainer], split, models)
    e, ids, rows = E[explainer], models_table.index.tolist(), []
    for a, b in itertools.combinations(range(len(ids)), 2):
        m = _subset(pred[a], pred[b], y, subset)
        v = _pair_value(e[a][m], e[b][m], measure, k) if m.any() else np.nan
        rows.append({"model_a": ids[a], "model_b": ids[b], "n_common": int(m.sum()), "distance": float(v)})
    return pd.DataFrame(rows).assign(explainer=explainer, measure=measure)


def disagreement(run: Run, explainers: Sequence[str], measure: str, k: int | None = None, *, split: str,
                 subset: str = "both_correct", models: str = "rashomon") -> pd.DataFrame:
    """same model, all pairs of explainers: mean distance over the comparison subset (both_correct: the model is
    correct; same_prediction: all samples); plus the rank of each pair per model (1 = most similar)"""
    models_table, E, pred, y, _ = _load(run, explainers, split, models)
    ids, rows = models_table.index.tolist(), []
    pairs = list(itertools.combinations(explainers, 2))
    for i, mid in enumerate(ids):
        m = (pred[i] == y) if subset == "both_correct" else np.ones(len(y), bool)
        vals = [_pair_value(E[p][i][m], E[q][i][m], measure, k) for p, q in pairs]
        ranks = rankdata(vals, method="average")
        for (p, q), v, r in zip(pairs, vals, ranks):
            rows.append({"model_id": mid, "explainer_1": p, "explainer_2": q, "pair": f"{p}-{q}",
                         "n_common": int(m.sum()), "distance": float(v), "rank": float(r)})
    return pd.DataFrame(rows).assign(measure=measure)


def kendall(dis: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau between the rank vectors of the method pairs for every pair of models"""
    R = dis.pivot(index="model_id", columns="pair", values="rank")
    rows = []
    for a, b in itertools.combinations(R.index, 2):
        tau, p = kendalltau(R.loc[a].to_numpy(), R.loc[b].to_numpy())
        rows.append({"model_a": a, "model_b": b, "tau": float(tau), "p": float(p)})
    return pd.DataFrame(rows).assign(measure=dis["measure"].iloc[0])


# ---------------------------------------------------------------------------------------------------------------
def output_dir(results_dir: Path | str, run: Run, analysis: str, name: str) -> Path:
    return Path(results_dir) / run.task / run.model_class / run.name / analysis / name


def run_all(run: Run, explainers: Sequence[str], k: int, *, results_dir: Path | str, name: str = "default", split: str,
            subset: str = "both_correct", models: str = "rashomon", measures: Sequence[str] = MEASURES) -> Path:
    """diversity + disagreement + Kendall tau for all measures; writes parquet tables and a manifest"""
    out = output_dir(results_dir, run, "rashomon", name)
    out.mkdir(parents=True, exist_ok=True)
    div = pd.concat([diversity(run, e, m, k, split=split, subset=subset, models=models) for e in explainers for m in measures])
    dis = pd.concat([disagreement(run, explainers, m, k, split=split, subset=subset, models=models) for m in measures])
    ken = pd.concat([kendall(dis[dis["measure"] == m]) for m in measures])
    div.to_parquet(out / "diversity.parquet", index=False)
    dis.to_parquet(out / "disagreement.parquet", index=False)
    ken.to_parquet(out / "kendall.parquet", index=False)
    sel = read_json(run.dir / "selection.json") if (run.dir / "selection.json").exists() else None
    write_json(out / "manifest.json", {
        "analysis": "rashomon (diversity, disagreement, Kendall tau)", "zoo_run": str(run.dir),
        "zoo_run_environment": run.manifest().get("environment"), "explainers": list(explainers),
        "explainer_settings": {e: {kk: vv for kk, vv in run.explainer_manifest(e).items() if kk not in ("model_ids", "sample_indices", "seconds_per_model")} for e in explainers}, "k": k, "split": split,
        "comparison_subset": subset, "models": models, "measures": list(measures), "normalisation": None,
        "kendall": "rank vectors of method pairs, all model pairs",
        "rashomon_criterion": sel["criterion"] if sel else None,
        "rashomon_selection": sel["selected_model_ids"] if sel else None, "environment": environment()})
    return out
