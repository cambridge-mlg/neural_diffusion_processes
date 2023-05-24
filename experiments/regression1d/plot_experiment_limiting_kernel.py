from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# sns.set_style("whitegrid")
sns.despine()
RENAMES = {
    "short-noisy-se": "SE $(\ell = 0.1)$",
    "noisy-se": "SE $(\ell = 1.0)$",
    "short-noisy-periodic": "Periodic $(\ell = 0.1)$",
    "noisy-periodic": "Periodic $(\ell = 1.0)$",
    "se": "Squared Exponential $(\ell = 0.25)$",
    "periodic": "Periodic $(\ell = 0.50)$",
    "white": "White",
    'hparams["sde.limiting_kernel"]': "Limiting kernel",
    'hparams["sde.score_parametrization"]': "Score parametrization",
    'interpolation_loglik_mean': "Log-Likelihood",
    'preconditioned_k': r"Precondition by $\mathbf{K}$",
    # 'preconditioned_k2': r"Precondition by $\sigma\mathbf{K}$ (2)",
    'preconditioned_s': r"Precondition by $\mathbf{S}^{\top}$",
    # 'preconditioned_s2': r"Precondition by $\mathbf{L}_{t|0}$ (2)",
    'y0': r"Predict $\mathbf{y}_0$",
    'mu_t': r"Predict $\mathbf{\mu}_{t|0}$",
    'none': "No preconditioning",
    'loss': "Loss",
}


def strip(x):
    x = x.strip("'")
    x = x.strip('"')
    return x


def display_name(x):
    return RENAMES.get(strip(x), strip(x))


def plot_barchart(df, outer_col: str, inner_col: str, col_depth: str, metric: str, error: str, ax, *, selectors={}, outer=None, inner=None, depth=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if outer is None:
        outer = df[outer_col].unique()
    if inner is None:
        inner = df[inner_col].unique()
    if depth is None:
        depth = df[col_depth].unique()

    x_outer = np.arange(len(outer))
    width = .6 / len(inner)

    selectors_flag = np.concatenate(
        [df[k].values.reshape(-1, 1) == v for k, v in selectors.items()],
        axis=1,
    ).all(axis=1, keepdims=True)

    for i, value_outer in enumerate(outer):
        for j, value_inner in enumerate(inner):
            for k, value_depth in enumerate(depth):
                flag = np.concatenate([
                    selectors_flag,
                    (df[col_inner] == value_inner).values.reshape(-1, 1),
                    (df[col_outer] == value_outer).values.reshape(-1, 1),
                    (df[col_depth] == value_depth).values.reshape(-1, 1),
                ], axis=1).all(axis=1)
                try:
                    print(value_outer, value_inner, value_depth, flag.sum())
                    v = df[flag][metric].values[0]
                    e = df[flag][error].values[0] if error is not None else 0.
                    v, e = float(v), float(e)
                except:
                    print("====> No value for", value_outer, value_inner, value_depth)
                    v, e = np.nan, 0
                if abs(e) > 10:
                    e = 0
                x_pos = x_outer[i] - len(inner)/2 * width + (j) * width + width/2 + width/4 * (k - 1)
                ax.set_ylim(-0.5, 1.0)
                b = ax.errorbar(
                    x_pos,
                    max(v, -0.5),
                    yerr=e,
                    fmt="o" if value_depth else "s",
                    lw=2,
                    # width=.5 * width,
                    color="C{}".format(j),
                    alpha=.8 if value_depth else 1.,
                )

    ax.set_xticks([])
    ax.set_xlabel(display_name(col_outer))
    # ax.set_ylabel(display_name(metric))
    labels = list(map(display_name, outer))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return ax


def plot_errorbar(df, outer_col: str, inner_col: str, col_marker: str, metric: str, error: str, *, selectors={}, outer=None, inner=None, marker=None, axes=None):
    if outer is None:
        outer = df[outer_col].unique()
    if inner is None:
        inner = df[inner_col].unique()
    if marker is None:
        marker = df[col_marker].unique()

    if axes is None:
        fig, axes = plt.subplots(1, len(outer), figsize=(5 * len(outer), 5))

    selectors_flag = np.concatenate(
        [df[k].values.reshape(-1, 1) == v for k, v in selectors.items()],
        axis=1,
    ).all(axis=1, keepdims=True)

    for i, value_outer in enumerate(outer):
        for j, value_inner in enumerate(inner):
            for k, value_marker in enumerate(marker):
                flag = np.concatenate([
                    selectors_flag,
                    (df[col_inner] == value_inner).values.reshape(-1, 1),
                    (df[col_outer] == value_outer).values.reshape(-1, 1),
                    (df[col_marker] == value_marker).values.reshape(-1, 1),
                ], axis=1).all(axis=1)
                try:
                    print(value_outer, value_inner, value_marker, flag.sum())
                    v = df[flag][metric].values[0]
                    e = df[flag][error].values[0] if error is not None else 0.
                    v, e = float(v), float(e)
                except:
                    print("====> No value for", value_outer, value_inner, value_marker)
                    v, e = np.nan, 0
                if abs(e) > 10:
                    e = 0
                # x_pos = len(inner)/2 * width + (j) * width + width/2 + width/4 * (k - 1)
                x_pos = len(marker) * j + k
                axes[i].set_ylim(-0.5, 1.0)
                b = axes[i].errorbar(
                    x_pos,
                    max(v, -0.5),
                    yerr=e,
                    fmt="o" if value_marker else "s",
                    lw=2,
                    # width=.5 * width,
                    color="C{}".format(j),
                    alpha=.8 if value_marker else 1.,
                )
                axes[i].set_ylabel(display_name(value_outer))

    # ax.set_xticks(range(len(outer)))
    # ax.set_xlabel(display_name(col_outer))
    # # ax.set_ylabel(display_name(metric))
    # labels = list(map(display_name, outer))
    # ax.set_xticklabels(labels, rotation=45, ha="right")
    return axes

    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '8', '<', '>']

def plot_metric_with_errorbar(
    df,
    metric_col,
    error_col,
    groupby_col=None,
    groupby_row=None,
    groupby_color=None,
    groupby_marker=None,
    col_values=None,
    row_values=None,
    color_values=None,
    marker_values=None,
    reduce_func=None,
    ylim=None,
):
    def get_unique_values(df, col_names, specified_values):
        if specified_values is not None:
            return [(x,) for x in specified_values]
        else:
            return [tuple(x) for x in df[col_names].drop_duplicates().values]

    if groupby_col is not None:
        col_values = get_unique_values(df, groupby_col, col_values)
    if groupby_row is not None:
        row_values = get_unique_values(df, groupby_row, row_values)
    if groupby_color is not None:
        color_values = get_unique_values(df, groupby_color, color_values)
    if groupby_marker is not None:
        marker_values = get_unique_values(df, groupby_marker, marker_values)

    nrows = len(row_values) if row_values else 1
    ncols = len(col_values) if col_values else 1

    # print debug information
    print("col_values:", col_values)
    print("row_values:", row_values)
    print("color_values:", color_values)
    print("marker_values:", marker_values)

    from tueplots import bundles, figsizes
    plt.rcParams.update(bundles.neurips2023(nrows=nrows, ncols=ncols))
    print(nrows, ncols)
    # def neurips2023(*, usetex=True, rel_width=1.0, nrows=1, ncols=1, family="serif"):
    figsize = figsizes.neurips2023(nrows=nrows, ncols=ncols, height_to_width_ratio=.66)['figure.figsize']
    print(figsize)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)


    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx, col_idx]

            for color_idx, color_val in enumerate(color_values):
                for marker_idx, marker_val in enumerate(marker_values):
                    filtered_df = df
                    if groupby_col is not None:
                        for col_name, col_v in zip(groupby_col, col_val):
                            filtered_df = filtered_df[filtered_df[col_name] == col_v]
                    if groupby_row is not None:
                        for row_name, row_v in zip(groupby_row, row_val):
                            filtered_df = filtered_df[filtered_df[row_name] == row_v]
                    if groupby_color is not None:
                        for color_name, color_v in zip(groupby_color, color_val):
                            filtered_df = filtered_df[filtered_df[color_name] == color_v]
                    if groupby_marker is not None:
                        for marker_name, marker_v in zip(groupby_marker, marker_val):
                            filtered_df = filtered_df[filtered_df[marker_name] == marker_v]

                    if not filtered_df.empty:
                        marker = marker_list[marker_idx % len(marker_list)]
                        # if color_val == ("white",) and marker_val == (False,):
                        #     print(filtered_df.iloc[filtered_df[metric_col].argmax()])

                        if reduce_func == "max":
                            idx = np.argmax(filtered_df[metric_col].values)
                            v = filtered_df[metric_col].values[idx]
                            e = filtered_df[error_col].values[idx]
                        elif reduce_func == "mean":
                            v = filtered_df[metric_col].values.mean()
                            e = filtered_df[metric_col].values.std()
                        elif reduce_func == "all":
                            v = filtered_df[metric_col].values
                            e = np.nan
                        else:
                            v = filtered_df[metric_col].values[0]
                            e = filtered_df[error_col].values[0]

                        x = len(marker_values) * color_idx + marker_idx * np.ones_like(v)
                        arrow_props = {
                            'arrowstyle': '->',  # Arrow style with a smaller head
                            'mutation_scale': 5,  # Control the size of the arrowhead
                            'lw': 1.5,  # Line width
                            'color': f'C{color_idx}',  # Arrow color
                            'alpha': .5,  # Transparency
                        }
                        if ylim is not None and v < ylim[0]:
                            ax.annotate(
                                "", xy=(x, ylim[0]), xytext=(x, ylim[0] + .2 *(ylim[1] - ylim[0])),
                                arrowprops=arrow_props, alpha=.1)
                        elif ylim is not None and v > ylim[1]:
                            ax.annotate(
                                "", xy=(x, ylim[1]), xytext=(x, ylim[1] - .2 *(ylim[1] - ylim[0])),
                                arrowprops=arrow_props)

                        ax.errorbar(x, v, yerr=e, markersize=4, marker=marker, color=f"C{color_idx}", alpha=1.0, linestyle="none")
                        ax.set_ylim(*ylim)
                        
                    col_title = ', '.join([f"{name}: {value}" for name, value in zip(groupby_col, col_val)]) if groupby_col else ''
                    row_title = ', '.join([f"{name}: {value}" for name, value in zip(groupby_row, row_val)]) if groupby_row else ''
                    if col_idx == 0:
                        ax.set_ylabel(row_title)
                    if row_idx == nrows - 1:
                        ax.set_xlabel(col_title)

                    ax.set_xticks(range(len(marker_values) * len(color_values)))
                    
    return fig, axes    


if __name__ == "__main__":
    HERE = Path(__file__).parent
    df = pd.read_csv(str(HERE / "experiments_May11_se_per.csv"))

    for col in df.columns:
        if col.startswith("hparams"):
            try:
                df[col] = df[col].apply(strip)
            except:
                print("Could not strip", col)

    for col in ["interpolation_loglik_mean", "interpolation_loglik_err", "interpolation_loglik_std"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    dataset = "periodic"
    col_values=["preconditioned_k", "preconditioned_s", "none", "y0"]
    color_values=["white", f"noisy-{dataset}", f"short-noisy-{dataset}"]
    marker_values=[True, False]
    fig, axes = plot_metric_with_errorbar(
        df[df['hparams["data.dataset"]'] == dataset],
        metric_col="interpolation_loglik_mean",
        error_col="interpolation_loglik_err",
        groupby_col=['hparams["sde.score_parametrization"]'],
        groupby_row=['hparams["data.dataset"]'],
        groupby_color=['hparams["sde.limiting_kernel"]'],
        groupby_marker=['hparams["sde.exact_score"]'],
        col_values=col_values,
        color_values=color_values,
        marker_values=marker_values,
        reduce_func="max" if dataset == "se" else None,
        ylim = (0.3, 0.9) if dataset == "se" else (-.1, 1.5),
    )
    axes = np.ravel(axes)
    axes[0].set_ylabel("TLL")
    for ax, c in zip(axes, col_values):
        ax.set_xlabel(display_name(c))

    fig = plt.figure()
    legend_entries = [
        {"color": f"C{i}", "alpha": 1, "label": display_name(v)} for i, v in enumerate(color_values)
    ]
    # create handles
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=entry["color"], alpha=entry["alpha"])
        for entry in legend_entries
    ]
    handles.extend([
        plt.Line2D([0], [0], marker=marker_list[i], color="w", markerfacecolor="k")
        for i, _ in enumerate(marker_values)
    ])
    # Create the legend labels
    labels = [entry["label"] for entry in legend_entries]
    labels.extend([
        display_name("exact score"),
        display_name("approximate score"),
    ])
    fig.legend(handles, labels, loc="upper center", ncol=5, facecolor='white', framealpha=0)
    # plt.tight_layout(rect=(0, 0, 1, .92))
    # plt.savefig(f"limiting_kernel_{dataset}.pdf")
    # plt.savefig(f"limiting_kernel_{dataset}.png")
    plt.show()
