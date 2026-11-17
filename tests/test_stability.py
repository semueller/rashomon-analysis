import numpy as np
import pytest

from rashomon_analysis import stability as ST


def test_compare_small_examples():
    u = np.array([[4.0, 3.0, 2.0, 1.0]])
    v = np.array([[3.0, 4.0, 1.0, 2.0]])
    m = ST.compare(u, v, ks=[1, 2])
    assert m[("topk set agreement", 2)][0] == 1.0 and m[("topk rank agreement", 2)][0] == 0.0
    assert m[("topk set agreement", 1)][0] == 0.0
    assert m["binary flip rate"][0] == 0.0          # all positive and >= 1 % of the maximum
    w = np.array([[4.0, -3.0, 2.0, 1.0]])
    assert ST.compare(u, w, ks=[2])["binary flip rate"][0] == 0.25  # feature 1 turns negative -> not relevant
    assert ST.compare(u, u, ks=[2])["cosine distance"][0] == pytest.approx(0.0)


def _run(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    import rashomon_zoo  # noqa: F401
    from rashomon_zoo import stages
    from rashomon_zoo.store import Task, save_task
    X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=0)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = {"train": slice(0, 300), "holdout_2": slice(300, 400), "holdout_1": slice(400, 500)}
    save_task(tmp_path, Task("toy", {k: X[v] for k, v in idx.items()}, {k: y[v] for k, v in idx.items()}, 2, {}))
    r = stages.train(tmp_path, "toy", "rf", "t", seeds=range(3), params={"n_estimators": 20})
    stages.select(r, {"epsilon": 1.0})
    return r


def test_budget_study_on_pipeline_run(tmp_path):
    from rashomon_zoo.explain import explain
    r = _run(tmp_path)
    variants = {}
    for n in (20, 200):
        variants[n] = []
        for s in (0, 1):
            name = f"ks__n{n}__s{s}"
            explain(r, "ks_shap", split="holdout_1", params={"n_samples": n, "seed": s}, models=[0, 1],
                    samples=30, out_name=name)
            variants[n].append(name)
    explain(r, "shap_tree", split="holdout_1", params={"baseline": "train_mean"}, models=[0, 1], samples=30,
            out_name="exact")
    df = ST.budget_study(r, variants, split="holdout_1", ks=[1, 2, 3], reference="exact")
    assert set(df["comparison"]) == {"seeds", "reference", "completeness"}
    assert set(df["model_id"]) == {0, 1}
    ref = df[(df.comparison == "reference") & (df.measure == "euclid")].groupby("budget")["mean"].mean()
    assert ref[200] < ref[20]  # more coalitions: closer to the exact values
    gap = df[df.comparison == "completeness"]["mean"]
    assert (gap < 1e-4).all()  # ks_shap enforces efficiency
    with pytest.raises(ValueError):
        ST.budget_study(r, variants, split="holdout_1", ks=[2], models=[2])  # model 2 not explained


def test_rashomon_on_explained_subset(tmp_path):
    from rashomon_zoo.explain import explain
    from rashomon_analysis import rashomon as RA
    r = _run(tmp_path)
    for e in ("ks_shap", "shap_tree"):
        explain(r, e, split="holdout_1", models=[0, 1], samples=40)
    with pytest.raises(ValueError, match="not explained"):
        RA.run_all(r, ["ks_shap", "shap_tree"], k=2, results_dir=tmp_path / "res", split="holdout_1")
    out = RA.run_all(r, ["ks_shap", "shap_tree"], k=2, results_dir=tmp_path / "res", split="holdout_1", models=[0, 1])
    assert (out / "diversity.parquet").exists()


def test_numeric_noise_small_example():
    # 2 seeds, 1 sample: mean (3, 4) (norm 5), deviations ±1 in feature 0 -> σ_0 = sqrt(2), σ_1 = 0
    E = np.array([[[4.0, 4.0]], [[2.0, 4.0]]])
    m = ST.numeric_noise(E)
    assert m["relative noise"][0] == pytest.approx(np.sqrt(2) / 5)
    assert m["max relative entry noise"][0] == pytest.approx(np.sqrt(2) / 5)
    with pytest.raises(ValueError):
        ST.numeric_noise(E[:1])


def test_noise_falls_with_budget_and_separation(tmp_path):
    from rashomon_zoo.explain import explain
    r = _run(tmp_path)
    variants = {}
    for n in (30, 300):
        variants[n] = []
        for s in (0, 1, 2):
            name = f"ks__n{n}__s{s}"
            explain(r, "ks_shap", split="holdout_1", params={"n_samples": n, "seed": s}, models=[0, 1, 2],
                    samples=30, out_name=name)
            variants[n].append(name)
    df = ST.budget_study(r, variants, split="holdout_1", ks=[2])
    noise = df[(df.measure == "relative noise")].groupby("budget")["mean"].mean()
    assert noise[300] < noise[30]
    sep = ST.separation(r, variants, split="holdout_1")
    assert set(sep["measure"]) == {"separation ratio", "relative model distance"}
    assert len(sep[(sep.budget == 30) & (sep.measure == "separation ratio")]) == 3  # 3 model pairs
    ratio = sep[sep.measure == "separation ratio"].groupby("budget")["median"].mean()
    assert ratio[300] < ratio[30]


def test_classifier_separation(tmp_path):
    from rashomon_zoo.explain import explain
    r = _run(tmp_path)
    variants = {300: []}
    for s in range(4):
        name = f"ks__n300__s{s}"
        explain(r, "ks_shap", split="holdout_1", params={"n_samples": 300, "seed": s}, models=[0, 1, 2],
                samples=40, out_name=name)
        variants[300].append(name)
    df = ST.classifier_separation(r, variants, split="holdout_1", folds=4)
    noise = df[df.comparison == "explainer noise"]
    pairs = df[df.comparison == "model pair"]
    assert len(noise) == 3 and len(pairs) == 3 and (df.runs_per_model == 2).all()
    assert noise.accuracy.between(0.3, 0.7).all()  # halves of the same model's runs: about chance
    assert pairs.accuracy.notna().all()


def test_required_budget_and_share():
    import pandas as pd
    # model 0: noise 0.2 / sqrt(n / 100) exactly; model 1: twice as noisy
    rows = []
    for mid, c in ((0, 0.2), (1, 0.4)):
        for n in (100, 400, 1600):
            rows.append({"explainer": "e", "model_id": mid, "budget": n, "comparison": "seeds",
                         "measure": "relative noise", "mean": c / np.sqrt(n / 100)})
    t = pd.DataFrame(rows)
    req = ST.required_budget(t, 0.1).set_index("model_id")
    assert req.loc[0, "required_budget"] == pytest.approx(400) and not req.loc[0, "extrapolated"]
    assert req.loc[1, "required_budget"] == pytest.approx(1600)
    req2 = ST.required_budget(t, 0.025).set_index("model_id")  # beyond 1600: extrapolated with slope -1/2
    assert req2.loc[1, "required_budget"] == pytest.approx(25600) and req2.loc[1, "extrapolated"]
    share = ST.converged_share(ST.required_budget(t, 0.1), [300, 400, 2000]).set_index("budget")["share"]
    assert list(share) == [0.0, 0.5, 1.0]
