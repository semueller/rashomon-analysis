"""Minimal working example: zoo (models + explanations) -> Rashomon analysis -> CFIRE with explainer selection.

Two tasks (HELOC, Breast Cancer), 5 XGBoost models and 5 MLPs per task (seeds 0-4). All models of a task get the
same training data; the variation comes from the learner's own seeded randomness: MLP initialisation (same batch
order, same number of steps), XGBoost row subsampling per tree (subsample 0.9).
After training, the run settings are verified (the script stops if one deviates) and the run statistics printed. Explanations on the whole holdout_1 (Rashomon analysis, CFIRE
fit and explainer selection); CFIRE is evaluated on holdout_2. Split sizes and roles: see SPLITS. Existing zoo runs are reused.
Run from the repo root:
  RASHOMON_ZOO_DATASETS=<folder with HELOC/, ...> uv run python examples/mwe.py <out_dir>
Output: <out_dir>/zoo (rashomon-zoo layout), <out_dir>/results (analyses).
"""

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

import rashomon_zoo  # noqa: F401  (OpenMP setting on macOS, before torch / xgboost)
from rashomon_zoo import stages, tasks
from rashomon_zoo.explain import explain
from rashomon_zoo.store import Run
from rashomon_analysis import cfire_eval, rashomon

warnings.filterwarnings("ignore")
base = Path(sys.argv[1] if len(sys.argv) > 1 else "mwe_out")
zoo_dir, results_dir = base / "zoo", base / "results"
TASKS = {"heloc": 8, "breastcancer": 8}          # task -> k for the top-k measures
# task stage: which samples exist. HELOC: holdout_1 500 drawn from the provider test file, holdout_2 300 from the
# provider training data (stratified); Breast Cancer: legacy splits of papers R / C (train 285, holdout_1 114 =
# loader test, holdout_2 170 = loader val).
# Roles as in papers R and C: holdout_1 = Rashomon criterion, explanations, Rashomon analysis, CFIRE fit and explainer
# selection; holdout_2 = CFIRE evaluation.
SPLITS = {"heloc": {"holdout_1": 500, "holdout_2": 300}, "breastcancer": "legacy"}
H1, H2 = "holdout_1", "holdout_2"
SEEDS = range(5)
PARAMS = {"xgb": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "subsample": 0.9},
          "nn": {"hidden": [64, 64], "n_steps": 2000, "batch_size": 64}}   # Rashomon setup: same order, same length
EXPL = {"xgb": ["shap_tree", "ks_shap", "lime_tabular"], "nn": ["ig", "ks_shap", "lime_tabular"]}
EXPL_PARAMS = {"lime_tabular": {"n_samples": 1000}}

EXPECTED = {"xgb": {"variation": "seed", "kind": "xgboost"},
            "nn": {"variation": "seed", "kind": "torch"}}


def verify(r, mc):
    """check the stored settings of a run against what this example intends; stop on any deviation"""
    import json
    run_manifest, models = r.manifest(), r.models()
    checks = {
        "variation": run_manifest["variation"] == EXPECTED[mc]["variation"],
        "model kind": run_manifest["kind"] == EXPECTED[mc]["kind"],
        "params as given": all(run_manifest["params"].get(k) == v for k, v in PARAMS[mc].items()),
        "seeds": run_manifest["seeds"] == list(SEEDS),
        "all models distinct": run_manifest["n_distinct_models"] == len(SEEDS),
        "Rashomon criterion": json.loads((r.dir / "selection.json").read_text())["criterion"]["epsilon"] == 0.05,
    }
    if mc == "nn":
        checks["same number of steps"] = models["steps"].nunique() == 1 and int(models["steps"].iloc[0]) == PARAMS["nn"]["n_steps"]
        checks["fixed-step training"] = run_manifest["params"]["stopping"] == "fixed_steps"
        checks["same batch order"] = run_manifest["params"]["order_seed"] == 0
    if mc == "xgb":
        checks["same training data (variation seed, xgboost row subsampling inside the learner)"] = \
            run_manifest["resample_fraction"] is None and run_manifest["params"].get("subsample") == 0.9
    failed = [k for k, ok in checks.items() if not ok]
    print(f"  verify {r.task} {mc}: " + ("all ok (" + ", ".join(checks) + ")" if not failed else f"FAILED: {failed}"))
    if failed:
        raise SystemExit(f"run settings deviate: {failed}")
    st = json.loads((r.dir / "stats.json").read_text())["all"]
    for split in (H1, H2):
        sp = st[split]
        print(f"  stats {split}: acc {sp['accuracy_mean']:.3f} ± {sp['accuracy_std']:.3f} [{sp['accuracy_min']:.3f}, "
              f"{sp['accuracy_max']:.3f}], prediction disagreement {sp['prediction_disagreement_mean']:.3f} "
              f"(max {sp['prediction_disagreement_max']:.3f}), output JSD {sp['output_jsd_mean']:.3f}; "
              f"distinct models {st['n_distinct_models']}, distinct outputs {st['n_distinct_outputs']}")


def verify_explanations(r, mc):
    from rashomon_zoo.explain import DEFAULTS
    for e in EXPL[mc]:
        for name, split in ((e, H1),):
            explainer_manifest = r.explainer_manifest(name)
            n_expected = len(r.load_task().y[split])
            want = {**DEFAULTS[e], **(EXPL_PARAMS.get(e) or {})}
            ok = explainer_manifest["split"] == split and len(explainer_manifest["sample_indices"]) == n_expected and explainer_manifest["params"] == want \
                and explainer_manifest["model_ids"] == list(range(len(SEEDS)))
            if not ok:
                raise SystemExit(f"explanations {r.task} {mc} {name} deviate: {explainer_manifest['params']} vs {want}")
    print(f"  verify explanations {r.task} {mc}: all ok ({', '.join(EXPL[mc])}; whole {H1})")




def prune_check(r, out_c, cfire_results):
    """sanity check of cfire.prune on the selected rule model of every model: "safe" must keep the predictions on the
    fit split; "best" (pruning paper, 5 % tolerance) shrinks the model. Rule counts and agreement on holdout_2."""
    import json
    import numpy as np
    from cfire import DNFClassifier, accuracy, prune
    t = r.load_task()
    P1, P2 = np.asarray(r.outputs(H1)), np.asarray(r.outputs(H2))
    rows = []
    for _, sel in cfire_results[cfire_results.explainer == "CFIRE (selected)"].iterrows():
        mid = int(sel.model_id)
        dnf = DNFClassifier.from_dict(json.loads((out_c / "rules" / str(mid) / f"{sel.selected_explainer}.json").read_text()))
        X1, Y1, X2, Y2 = t.X[H1], P1[mid].argmax(1), t.X[H2], P2[mid].argmax(1)
        safe, best = prune(dnf, X1, Y1, mode="safe"), prune(dnf, X1, Y1, mode="best")
        if not np.array_equal(safe.dnf.predict(X1), dnf.predict(X1)):
            raise SystemExit(f"prune safe changed fit-split predictions ({r.task} {r.model_class} model {mid})")
        rows.append({"rules": safe.before["n_rules"], "rules safe": safe.after["n_rules"], "rules best": best.after["n_rules"],
                     f"acc {H2}": accuracy(dnf, X2, Y2), f"acc {H2} best": accuracy(best.dnf, X2, Y2),
                     "removed rules best, sole cover of fit samples": sum(1 for d in best.removed if d["share_covered_by_kept"] < 1)})
    m = pd.DataFrame(rows).median()
    print("  pruning check (selected rule models, medians; safe keeps fit-split predictions: ok): "
          + ", ".join(f"{k} {v:.3g}" for k, v in m.items()))


BOUNDED = ["feature disagreement", "sign disagreement"]  # in [0, 1]; the Euclidean measures depend on each method's scale


def print_rashomon(div, dis, ken, k):
    """the two comparisons of the Rashomon analysis, kept apart:
    diversity (Rashomon effect): different models, same method -> median over model pairs
    disagreement problem: same model, different methods -> median over models"""
    name = lambda e: e[:-5] if e.endswith("_test") else e  # noqa: E731
    print(f"  diversity, Rashomon effect (different models, same method; median over model pairs, top-{k}):")
    medians = div[div.measure.isin(BOUNDED)].groupby(["explainer", "measure"])["distance"].median().unstack()[BOUNDED]
    for e, row in medians.iterrows():
        print(f"    {name(e):14s} " + ", ".join(f"{m} {row[m]:.2f}" for m in BOUNDED))
    print(f"  disagreement problem (same model, different methods; median over models, top-{k}):")
    pairs = dis[dis.measure.isin(BOUNDED)].copy()
    pairs["pair"] = pairs.explainer_1.map(name) + " vs " + pairs.explainer_2.map(name)
    medians = pairs.groupby(["pair", "measure"])["distance"].median().unstack()[BOUNDED]
    for pr, row in medians.iterrows():
        print(f"    {pr:28s} " + ", ".join(f"{m} {row[m]:.2f}" for m in BOUNDED))
    kf = ken[ken.measure == "feature disagreement"]
    print(f"    same ranking of the method pairs across models (Kendall tau, feature disagreement): mean "
          f"{kf.tau.mean():.2f} over {len(kf)} model pairs ({dis.pair.nunique()} method pairs per model)")


t0 = time.time()
summary_rows = []
for task, k in TASKS.items():
    if not (zoo_dir / task / "task.json").exists():
        tasks.make_task(zoo_dir, task, splits=SPLITS[task])
    for mc in ("nn", "xgb"):  # model classes
        # 1. zoo: train, select, explain
        r = Run(zoo_dir, task, mc, "mwe")
        if not (r.dir / "models.parquet").exists():
            stages.train(zoo_dir, task, mc, "mwe", seeds=SEEDS, params=PARAMS[mc])
        stages.select(r)
        verify(r, mc)
        models = r.models()
        print(f"[{time.time() - t0:5.0f}s] {task} {mc}: {H1} acc {models[H1 + '_acc'].mean():.3f} ± {models[H1 + '_acc'].std():.3f}, "
              f"Rashomon set {int(models.in_rashomon_set.sum())}/{len(models)}")
        for e in EXPL[mc]:
            if not (r.dir / "explanations" / e).exists():
                explain(r, e, split=H1, models="rashomon", params=EXPL_PARAMS.get(e))
        verify_explanations(r, mc)
        # 2. Rashomon analysis (holdout_1, as paper R)
        out_r = rashomon.run_all(r, EXPL[mc], k=k, results_dir=results_dir, name="mwe", split=H1)
        div = pd.read_parquet(out_r / "diversity.parquet")
        dis = pd.read_parquet(out_r / "disagreement.parquet")
        ken = pd.read_parquet(out_r / "kendall.parquet")
        # 3. CFIRE with explainer selection (fit and select on holdout_1, evaluate on holdout_2, as paper C)
        out_c = cfire_eval.run_cfire(r, EXPL[mc], results_dir=results_dir, name="mwe", fit_split=H1, eval_split=H2)
        cfire_results = pd.read_parquet(out_c / "results.parquet")
        cfire_summary = cfire_eval.summary(cfire_results)
        print_rashomon(div, dis, ken, k)
        print(f"  CFIRE (medians over models, agreement with the black box on {H2}):")
        print("    " + cfire_summary[["accuracy", "covered_accuracy", "coverage", "f1", "n_rules", "times_selected"]].round(3)
              .to_string().replace("\n", "\n    "))
        prune_check(r, out_c, cfire_results)  # 4. pruning (cfire.prune) on the selected rule models
        for e, row in cfire_summary.iterrows():
            summary_rows.append({"task": task, "model_class": mc, "explainer": e, **row.to_dict()})
pd.DataFrame(summary_rows).to_csv(results_dir / "mwe_summary.csv", index=False)
print(f"\ntotal {time.time() - t0:.0f}s; zoo in {zoo_dir}, results in {results_dir} (summary: mwe_summary.csv)")
