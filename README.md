# rashomon-analysis

Analyses of Rashomon sets produced by [rashomon-zoo](../rashomon-zoo): it reads the stored models' outputs and explanations (the zoo's storage layout is the interface) and writes its results to its own folder (`<results_dir>/<task>/<model_class>/<run>/<analysis>/<name>/`); the zoo data are only read.

- `rashomon_analysis.rashomon`: explanation diversity, method disagreement, Kendall tau.
- `rashomon_analysis.cfire_eval`: CFIRE rule models per model and explainer (precision, coverage, F1, size) and explainer selection.
- `rashomon_analysis.stability`: stability of explanations across explainer seeds and budgets, per model: numeric noise over seeds (relative to the explanation's size), pairwise measures (top-k set and rank agreement for any k, flip rate of CFIRE's binarisation, Euclidean and cosine distance), completeness gap, optionally against a reference explanation; `separation`: explainer noise against the differences between models of the run (distance ratio); `classifier_separation`: held-out accuracy of a classifier telling two models' explanations apart, against an explainer-noise baseline. No convergence threshold built in.


`rashomon_analysis.py`: explanation diversity (same explainer, pairs of models), method disagreement (same model,
pairs of explainers, with per-model ranks) and Kendall tau between the per-model rank vectors.

```python
from rashomon_analysis import rashomon as RA
out = RA.run_all(run, ["ig", "ks_shap", "lime_tabular"], k=8, results_dir="results", split="holdout_1")
```
Comparison subset: `"both_correct"` (default) or `"same_prediction"`. The manifest records the settings and the
run's Rashomon criterion. Feature and sign disagreement, comparison subsets and ranks were checked against the evaluation
code of Müller et al. (2023) on its stored data; the Euclidean distances and the Kendall tau are computed differently
from that code (see the module docstring).

## Development

The dependencies `rashomon-zoo` and `cfire` are installed from GitHub (`[tool.uv.sources]` in `pyproject.toml`).
To work on all three repositories at once, with the others checked out next to this one:

```bash
uv run --with-editable ../rashomon-zoo --with-editable ../CFIRE python examples/mwe.py <out_dir>
```
