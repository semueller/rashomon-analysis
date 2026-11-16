import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import rashomon_zoo  # noqa: F401
from rashomon_zoo import stages
from rashomon_zoo.explain import explain
from rashomon_zoo.store import Task, save_task
from rashomon_analysis import cfire_eval


def test_run_cfire_with_selection(tmp_path):
    X, y = make_classification(n_samples=600, n_features=5, n_informative=3, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = {"train": slice(0, 300), "holdout_2": slice(300, 450), "holdout_1": slice(450, 600)}
    save_task(tmp_path, Task("toy", {k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()}, 2, {}))
    r = stages.train(tmp_path, "toy", "xgb", "t", seeds=range(3), params={"n_estimators": 20})
    stages.select(r, {"epsilon": 1.0})
    for e in ("shap_tree", "ks_captum"):
        explain(r, e, split="holdout_1")
    out = cfire_eval.run_cfire(r, ["shap_tree", "ks_captum"], results_dir=tmp_path / "results", fit_split="holdout_1",
                               eval_split="holdout_2")
    res = pd.read_parquet(out / "results.parquet")
    assert len(res) == 3 * 3  # 3 models x (2 explainers + selected)
    sel = res[res.explainer == "CFIRE (selected)"]
    assert sel.selected_explainer.isin(["shap_tree", "ks_captum"]).all()
    rules = json.loads((out / "rules" / "0" / "shap_tree.json").read_text())
    assert rules["format"].startswith("cfire.DNFClassifier")
    man = json.loads((out / "manifest.json").read_text())
    assert man["fitness"] == "covered_accuracy" and man["rashomon_criterion"]["epsilon"] == 1.0
    assert not (r.dir / "analyses").exists()


def test_f1_is_the_cfire_paper_definition():
    """F1 = 2PR/(P+R) with P = precision on covered samples, R = accuracy incl. rejects = P * coverage"""
    from rashomon_analysis.cfire_eval import evaluate

    class DNF:  # minimal rule model: predicts -1 (reject) on the last 4 of 10 samples
        rules = [[[0]], [[1]]]
        def predict(self, X):
            return np.array([0, 0, 1, 1, 0, 1, -1, -1, -1, -1])

    Y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    m = evaluate(DNF(), np.zeros((10, 2)), Y, 2)
    p, c = 5 / 6, 0.6
    assert np.isclose(m["covered_accuracy"], p) and np.isclose(m["coverage"], c)
    assert np.isclose(m["accuracy"], p * c) and np.isclose(m["f1"], p * 2 * c / (1 + c))
