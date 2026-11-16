import sys
from pathlib import Path

import numpy as np

from rashomon_analysis import rashomon as RA


def test_measures_small_examples():
    a = np.array([[1.0, 2.0, 3.0, 4.0]]); b = np.array([[4.0, 3.0, 2.0, 1.0]])
    assert RA.distance(a, b, "feature disagreement", k=2)[0] == 1.0          # top-2 {2,3} vs {0,1}
    assert RA.distance(a, a, "feature disagreement", k=2)[0] == 0.0
    c = np.array([[1.0, 2.0, -3.0, 4.0]])
    assert RA.distance(a, c, "sign disagreement", k=2)[0] == 0.5            # overlap {2,3}, feature 2 flips sign
    assert np.isclose(RA.distance(a, b, "euclid")[0], np.sqrt(20))
    assert RA.distance(a, -a, "euclid abs")[0] == 0.0


def test_run_all_on_pipeline_run(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    import rashomon_zoo  # noqa: F401
    from rashomon_zoo import stages
    from rashomon_zoo.explain import explain
    from rashomon_zoo.store import Task, save_task
    X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = {"train": slice(0, 300), "holdout_2": slice(300, 400), "holdout_1": slice(400, 500)}
    save_task(tmp_path, Task("toy", {k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()}, 2, {}))
    r = stages.train(tmp_path, "toy", "nn", "t", seeds=range(4), params={"hidden": [16], "n_steps": 200})
    stages.select(r, {"epsilon": 1.0})
    for e in ("vg", "ig"):
        explain(r, e, split="holdout_1")
    out = RA.run_all(r, ["vg", "ig"], k=2, results_dir=tmp_path / "results", split="holdout_1")
    assert not (r.dir / "analyses").exists()  # zoo data only read
    assert (out / "kendall.parquet").exists() and (out / "manifest.json").exists()
