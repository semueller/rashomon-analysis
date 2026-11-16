"""CFIRE on a Rashomon set: one rule model per black box and explainer, explainer selection per black box.

Protocol (as in the CFIRE studies): CFIRE is fitted on the samples of the fit split whose explanations are stored,
with the black box's predictions as labels, and the explainer is selected there; rule models are evaluated on the
evaluation split against the black box's predictions. Both splits are named by the caller (the CFIRE paper (2025) under the legacy
splits: fit_split="holdout_1", eval_split="holdout_2"). Only stored outputs and explanations are read (no models).

Writes <results_dir>/<task>/<model_class>/<run>/cfire/<name>/:
  results.parquet   one row per (model, explainer) plus one row per model for the selected explainer
  rules/<model_id>/<explainer>.json   rule models (cfire.DNFClassifier.to_dict)
  manifest.json     settings, cfire version, Rashomon criterion, zoo run
"""
from __future__ import annotations

import json
import time
from importlib import metadata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cfire import CFIRESelect, DNFClassifier, accuracy, coverage, covered_accuracy
from rashomon_zoo.store import Run, environment, read_json, write_json

from .rashomon import output_dir


def _n_literals(dnf) -> int:
    return int(sum(len(rule) for cls in dnf.rules for rule in cls))


def _f1(precision, recall) -> float:
    """F1 of the CFIRE paper (2025): precision = agreement with the black box on covered samples, recall =
    agreement over all samples (rejects count as misses) = precision * coverage"""
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def evaluate(dnf: DNFClassifier, X: np.ndarray, Y: np.ndarray, n_classes: int) -> dict:
    p, r = covered_accuracy(dnf, X, Y), accuracy(dnf, X, Y)
    return {"accuracy": r, "covered_accuracy": p,
            "coverage": coverage(dnf, X), "f1": _f1(p, r),
            "n_rules": int(sum(len(c) for c in dnf.rules)), "n_literals": _n_literals(dnf)}


def run_cfire(run: Run, explainers: Sequence[str], *, results_dir: Path | str, name: str = "default", fit_split: str, eval_split: str,
              models: str = "rashomon", fitness: str = "covered_accuracy",
              cfire_kwargs: dict | None = None) -> Path:
    cfire_kwargs = {"random_state": 0, **(cfire_kwargs or {})}
    out = output_dir(results_dir, run, "cfire", name)
    task = run.load_task()
    models_table = run.models()
    if models == "rashomon":
        models_table = models_table[models_table["in_rashomon_set"]]
    ids = models_table["model_id"].tolist()
    # fit samples: the stored explanations on the fit split (identical for all explainers)
    explainer_manifests = {e: run.explainer_manifest(e) for e in explainers}
    idx = explainer_manifests[explainers[0]]["sample_indices"]
    if any(em["sample_indices"] != idx for em in explainer_manifests.values()) or \
            any(em["split"] != fit_split for em in explainer_manifests.values()):
        raise ValueError("explainers must be stored for the same samples of the fit split")
    X_fit, X_eval = task.X[fit_split][idx], task.X[eval_split]
    P_fit, P_eval = np.asarray(run.outputs(fit_split)), np.asarray(run.outputs(eval_split))
    rows = []
    for mid in ids:
        Y_fit = P_fit[mid][idx].argmax(1)
        Y_eval = P_eval[mid].argmax(1)
        E = {e: np.asarray(run.explanations(e, fit_split))[explainer_manifests[e]["model_ids"].index(mid)] for e in explainers}
        t0 = time.time()
        sel = CFIRESelect(E, fitness=fitness, **cfire_kwargs).fit(X_fit, Y_fit)
        seconds = time.time() - t0
        for e, cf in sel.models_.items():
            rules_dir = out / "rules" / str(mid)
            rules_dir.mkdir(parents=True, exist_ok=True)
            (rules_dir / f"{e}.json").write_text(json.dumps(cf.dnf.to_dict()))
            rows.append({"model_id": mid, "explainer": e, "selected": e == sel.selected_, "fitness": sel.scores_[e],
                         **evaluate(cf.dnf, X_eval, Y_eval, task.n_classes)})
        rows.append({"model_id": mid, "explainer": "CFIRE (selected)", "selected_explainer": sel.selected_,
                     "selected": True, "fitness": sel.scores_[sel.selected_], "fit_seconds_all": seconds,
                     **evaluate(sel.dnf, X_eval, Y_eval, task.n_classes)})
    results = pd.DataFrame(rows)  # one row per (model, explainer) + one per model for the selected explainer
    results.to_parquet(out / "results.parquet", index=False)
    sel_info = read_json(run.dir / "selection.json") if (run.dir / "selection.json").exists() else None
    write_json(out / "manifest.json", {
        "analysis": "cfire (rule model per black box and explainer, explainer selection)",
        "zoo_run": str(run.dir), "zoo_run_environment": run.manifest().get("environment"),
        "explainers": list(explainers), "explainer_settings": {e: em.get("params") for e, em in explainer_manifests.items()},
        "fit_split": fit_split, "fit_sample_indices": idx, "eval_split": eval_split,
        "labels": "black-box predictions (argmax of stored probabilities)", "fitness": fitness,
        "f1": "2PR/(P+R), P = covered_accuracy, R = accuracy (CFIRE paper)",
        "cfire_kwargs": cfire_kwargs, "cfire_version": metadata.version("cfire"), "models": models,
        "rashomon_criterion": sel_info["criterion"] if sel_info else None, "environment": environment()})
    return out


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """median over models per explainer (incl. the selected row) and how often each explainer was selected"""
    cols = ["accuracy", "covered_accuracy", "coverage", "f1", "n_rules", "n_literals"]
    s = results.groupby("explainer")[cols].median()
    picked = results.loc[results["explainer"] == "CFIRE (selected)", "selected_explainer"].value_counts()
    s["times_selected"] = picked.reindex(s.index).fillna(0).astype(int)
    return s
