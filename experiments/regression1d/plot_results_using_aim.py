from pathlib import Path
import pandas as pd
import numpy as np
import aim

import seaborn as sns
import matplotlib.pyplot as plt


EXPERIMENT_ID = "regression1d-2"

_HERE = Path(__file__).parent

repo = aim.sdk.repo.Repo(str(_HERE))


def get_last_non_nan_value_as_dict(df: pd.DataFrame):
    """
    Returns a dictionary where each entry is the last non-nan value
    of the corresponding column in the dataframe.
    """
    last_non_nan_values = {}
    for column in df.columns:
        if len(df[column].dropna()) == 0:
            continue
        last_non_nan_value = df[column].dropna().iloc[-1]
        last_non_nan_values[column] = last_non_nan_value
    return last_non_nan_values
    


def read_data() -> pd.DataFrame:
    data = []

    for run in repo.iter_runs():
        if run.experiment != EXPERIMENT_ID:
            continue
        metrics = {}
        for metric_info in run.iter_metrics_info():
            name, context = metric_info[0], metric_info[1]
            if name.startswith("__"):
                # do not include internal metrics
                continue
            metric = run.get_metric(name, context)
            last_values_for_metric = get_last_non_nan_value_as_dict(metric.dataframe())
            metrics[name] = last_values_for_metric["value"]
        
        data.append({**run["hparams"], **metrics})

    df = pd.DataFrame(data)
    # filter columns out that have only one value
    df = df.loc[:, df.nunique() != 1]
    return df

if __name__ == "__main__":
    # df = read_data()
    df = pd.read_csv("tmp.csv", index_col=0)
    datasets = [
        "se", "matern", "weaklyperiodic", "sawtooth", "mixture",
    ]
    assert set(datasets) == set(df["data.dataset"].unique())
    tasks = ["interpolation", "generalization", "extrapolation"]
    limiting_kernels = df["sde.limiting_kernel"].unique()
    translation_invariances = [True, False]

    print(datasets)
    print(tasks)
    print(limiting_kernels)
    width = .3

    fig, axes = plt.subplots(len(tasks), len(datasets), figsize=(2 * len(datasets), 2 * len(tasks)))

    for i, task in enumerate(tasks):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]

            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                ax.set_ylabel(task)

            for ti in translation_invariances:
                df_subset = df[(df["data.dataset"] == dataset) & (df["network.translation_invariant"] == ti)]
                kernels, y, err = df_subset["sde.limiting_kernel"].values, df_subset[f"{task}_loglik_mean"].values, df_subset[f"{task}_loglik_err"].values
                x = np.arange(len(kernels)) - width/2. + width * int(ti)
                ax.bar(x, y, yerr=err, width=width)
            # x = range(len(limiting_kernels))
            # ax.bar(range(len(limiting_kernels)), df_subset["test_mse"])
            sns.despine(ax=ax)
            if i < len(tasks) - 1:
                ax.set_xticklabels([])
            else:
                labels = [kernel.replace("_", " ") for kernel in limiting_kernels]
                # x_values = np.arange(len(limiting_kernels)) - width/2. + width
                ax.set_xticks(range(len(limiting_kernels)))
                # set labels and rotate 45 degrees
                ax.set_xticklabels(labels, rotation=45, ha="right")
            
plt.tight_layout()
plt.savefig("tmp.png")