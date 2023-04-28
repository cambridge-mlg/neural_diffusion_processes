from pathlib import Path
import pandas as pd
import numpy as np
import aim

import seaborn as sns
import matplotlib.pyplot as plt


EXPERIMENT_ID = "regression1d-Apr26"

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
    return df

if __name__ == "__main__":
    # df = read_data()
    # df.to_csv(f"results_{EXPERIMENT_ID}.csv")
    df = pd.read_csv(f"results_{EXPERIMENT_ID}.csv", index_col=0)
    df = df.loc[:, df.nunique() != 1]
    datasets = [
        "se", "matern", "weaklyperiodic", "sawtooth", "mixture",
    ]
    assert set(datasets) == set(df["data.dataset"].unique())
    tasks = ["interpolation", "generalization"]
    limiting_kernels = df["sde.limiting_kernel"].unique()
    limiting_kernels = [
        "white", "noisy-se", "noisy-matern52"
    ]
    translation_invariances = [True, False]
    y_lims = {
        "se": [-0.1, 0.8],
        "matern": [-0.2, 0.3],
        "weaklyperiodic": [-1.0, 0.0],
        "sawtooth": [-0.1, 0.0],
        "mixture": [-0.2, 0.6],
    }
    gp_baseline = {
        "interpolation": {
            "se": {'value': 0.70, 'error': 4.8e-3},
            "matern" : {'value': 0.31, 'error': 4.8e-3},
            "weaklyperiodic" : {'value': -0.32, 'error': 4.3e-3},
        },
        "generalization": {
            "se": {'value': 0.70, 'error': 4.8e-3},
            "matern" : {'value': 0.31, 'error': 4.8e-3},
            "weaklyperiodic" : {'value': -0.32, 'error': 4.3e-3},
        }
    }

    print(datasets)
    print(tasks)
    print(limiting_kernels)
    sns.set_style("whitegrid")
    sns.despine(left=True)

    width = .3
    xlim = (-width - .1, len(limiting_kernels) - 1 + width + .1)

    fig, axes = plt.subplots(len(tasks), len(datasets), figsize=(2 * len(datasets), 2 * len(tasks)))

    for i, task in enumerate(tasks):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            ax.set_xlim(-width - .1, len(limiting_kernels) - 1 + width + .1)
            ax.set_xlim(xlim[0], xlim[1])

            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                ax.set_ylabel(task)

            handles = []
            for ti in translation_invariances:
                df_subset = df[(df["data.dataset"] == dataset) & (df["network.translation_invariant"] == ti)]
                df_subset = df_subset.iloc[[df_subset["sde.limiting_kernel"].values.tolist().index(k) for k in limiting_kernels]]
                kernels, y, err = df_subset["sde.limiting_kernel"], df_subset[f"{task}_loglik_mean"], df_subset[f"{task}_loglik_err"]
                if ti:
                    print(dataset, task, '\t\t:' f"{y.values[0]:.2f}", f"({err.values[0]:.3f})")
                outside_range = y.abs() > 2.0
                y[outside_range] = -2.0
                err[outside_range] = 0.0
                x = np.arange(len(kernels)) - width/2. + width * int(ti)
                ax.bar(
                    x[outside_range],
                    y.values[outside_range],
                    yerr=err.values[outside_range],
                    width=width, label=f"translation invariant={ti}", color="C0" if ti else "C1", alpha=0.2)
                b = ax.bar(
                    x[~outside_range],
                    y.values[~outside_range],
                    yerr=err.values[~outside_range],
                    width=width, label=f"translation invariant={ti}", color="C0" if ti else "C1")
                handles.append(b)
            
            if gp_baseline[task].get(dataset) is not None:
                v = gp_baseline[task].get(dataset)["value"]
                e = gp_baseline[task].get(dataset)["error"]
                handle_gp, = ax.plot([xlim[0], xlim[1]], [v, v], color="black", linestyle="--", lw=1, label="GP baseline")
                ax.fill_between([xlim[0], xlim[1]], [v - e, v - e], [v + e, v + e], color="black", alpha=0.2)
            
            
            # x = range(len(limiting_kernels))
            # ax.bar(range(len(limiting_kernels)), df_subset["test_mse"])
            # sns.despine(ax=ax)
            if i < len(tasks) - 1:
                ax.set_xticklabels([])
            else:
                labels = [kernel.replace("_", " ") for kernel in limiting_kernels]
                # x_values = np.arange(len(limiting_kernels)) - width/2. + width
                ax.set_xticks(range(len(limiting_kernels)))
                # set labels and rotate 45 degrees
                ax.set_xticklabels(labels, rotation=45, ha="right")

print(len(handles))
handles.append(handle_gp)
print(len(handles))
fig.legend(
    handles=handles,
    ncols=3,
    loc='upper center',
    # bbox_to_anchor=(-1.5, -0.03),
    frameon=False,
    borderaxespad=0)
# fig.subplots_adjust(top=.85, hspace=0.125 / 4 * 2, wspace=.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.align_ylabels()
plt.savefig("regression1d_ablation1.png") #, dpi=600)