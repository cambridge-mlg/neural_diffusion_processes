import pandas as pd
import pathlib
import itertools
import numpy as np

_HERE = pathlib.Path(__file__).parent
_LOG_DIR = 'logs'

DATASETS = [
    "se",
    "matern",
    "weaklyperiodic",
    "noisymixture",
]

TASKS = [
    "interpolation",
    "extrapolation",
    "generalization",
]

MODELS = [
    "gp",
    "gpfull"
]

METRICS = [
    "log_prob"
]

STATS = [
    "mean"
]

STATS_FN = {
    "mean": np.mean
}

data = []
err_fn = lambda v: 1.96 * np.std(v) / np.sqrt(len(v))

for dataset, model, task, metric, stat in itertools.product(DATASETS, MODELS, TASKS, METRICS, STATS):
    df = pd.read_csv(str(_HERE / _LOG_DIR / dataset / model / f"{task}.csv"))
    values = df[metric].values
    data.append({
        "dataset": dataset,
        "model": model,
        "task": task,
        "metric": metric,
        "stat": stat,
        "value": STATS_FN[stat](values),
        "error": err_fn(values)
    })

df = pd.DataFrame(data)
print(df)



