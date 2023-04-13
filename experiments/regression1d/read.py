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
    data.extend([{
        "dataset": dataset,
        "model": model,
        "task": task,
        "metric": metric,
        "value": v,
        # "metric": metric,
        # "stat": stat,
        # metric: -1.0 * STATS_FN[stat](values),
        # "error": err_fn(values)
    } for v in values])

df = pd.DataFrame(data)
print(df)

import matplotlib.pyplot as plt

num_datasets = len(DATASETS)
num_tasks = len(TASKS)
fig, axes = plt.subplots(num_datasets, num_tasks, sharex=True)
for i, j in itertools.product(range(num_datasets), range(num_tasks)):
    dataset = DATASETS[i]
    task = TASKS[j]
    ax = axes[i, j]
    tmp = df[(df.dataset == dataset) & (df.task == task)][["value", "model"]]
    tmp.boxplot(column="value", by="model", ax=ax)
    if j == 0:
        ax.set_ylabel(dataset)
    if i < num_datasets - 1:
        ax.set_xlabel('')
    if i == 0:
        ax.set_title(task)
    else:
        ax.set_title('')


fig.suptitle(metric)
plt.show()

# pd.plotting
# df.boxplot(column=["dataset", "task"], by="task")
# plt.show()
# %%
# import pandas as pd


# %%
