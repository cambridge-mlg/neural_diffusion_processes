#%%
import numpy as np
import pandas as pd


RENAMES = {
    "se": "Squared Exponential",
    "sawtooth": "Sawtooth",
    "matern": r"Mat\'ern-$\frac52$",
    "weaklyperiodic": "Weakly Periodic",
    "mixture": "Mixture",
    "GP": "GP (optimum)",
    "NDP": "$\mathcal{T}-$sNDP (ours)",
    "generalization": "Generalisation",
    "interpolation": "Interpolation",
}

def display_name(x):
    return RENAMES.get(x, x)


DATASETS = [
    "se",
    "matern",
    "weaklyperiodic",
    "sawtooth",
    "mixture",
]

TASKS = [
    "interpolation",
    "generalization",
]

MODELS = [
    "GP",
    "DiagGP",
    "GNP",
    "ConvCNP",
    "ConvNP",
    "ANP",
    "NDP",
]

## interpolation
results_interpolation = {
    'GP': [
        {'value': 0.70, 'error': 4.8e-3},
        {'value': 0.31, 'error': 4.8e-3},
        {'value': -0.32, 'error': 4.3e-3},
        {'value': None, 'error': None},
        {'value': None, 'error': None},
    ],
    'DiagGP': [
        {'value': -0.81, 'error': 0.01},
        {'value': -0.93, 'error': 0.01},
        {'value': -1.18, 'error': 7.0e-3},
        {'value': None, 'error': None},
        {'value': None, 'error': None},
    ],
    'GNP': [
        {'value': 0.70, 'error': 5.0e-3},
        {'value': 0.30, 'error': 5.0e-3},
        {'value': -0.47, 'error': 5.0e-3},
        {'value': 0.42, 'error': 0.01},
        {'value': 0.10, 'error': 0.02},
    ],
    'ConvCNP': [
        {'value': -0.80, 'error': 0.01},
        {'value': -0.95, 'error': 0.01},
        {'value': -1.20, 'error': 7.0e-3},
        {'value': 0.55, 'error': 0.02},
        {'value': -0.93, 'error': 0.02},
    ],
    'ConvNP': [
        {'value': -0.46, 'error': 0.01},
        {'value': -0.67, 'error': 9.0e-3},
        {'value': -1.02, 'error': 6.0e-3},
        {'value': 1.20, 'error': 7.0e-3},
        {'value': -0.50, 'error': 0.02},
    ],
    'ANP': [
        {'value': -0.61, 'error': 0.01},
        {'value': -0.75, 'error': 0.01},
        {'value': -1.19, 'error': 5.0e-3},
        {'value': 0.34, 'error': 7.0e-3},
        {'value': -0.69, 'error': 0.02},
    ],
    'NDP': [
        {'value': 0.69, 'error': 0.02},
        {'value': 0.32, 'error': 0.02},
        {'value': -0.51, 'error': 0.027},
        {'value': 3.39, 'error': 0.041},
        {'value': 0.41, 'error': 0.018},
    ]
}

## Generalization
results_generalization = {
    "GP": [
        {"value": 0.70, "error": 4.8e-3},
        {"value": 0.31, "error": 4.8e-3},
        {"value": -0.32, "error": 4.3e-3},
        {"value": None, "error": None},
        {"value": None, "error": None}
    ],
    "DiagGP": [
        {"value": -0.81, "error": 0.01},
        {"value": -0.93, "error": 0.01},
        {"value": -1.18, "error": 7.0e-3},
        {"value": None, "error": None},
        {"value": None, "error": None}
    ],
    "GNP": [
        {"value": 0.69, "error": 5.0e-3},
        {"value": 0.30, "error": 5.0e-3},
        {"value": -0.47, "error": 5.0e-3},
        {"value": 0.42, "error": 0.01},
        {"value": 0.10, "error": 0.02}
    ],
    "ConvCNP": [
        {"value": -0.81, "error": 0.01},
        {"value": -0.95, "error": 0.01},
        {"value": -1.20, "error": 7.0e-3},
        {"value": 0.53, "error": 0.02},
        {"value": -0.96, "error": 0.02}
    ],
    "ConvNP": [
        {"value": -0.46, "error": 0.01},
        {"value": -0.67, "error": 9.0e-3},
        {"value": -1.02, "error": 6.0e-3},
        {"value": 1.19, "error": 7.0e-3},
        {"value": -0.53, "error": 0.02}
    ],
    "ANP": [
        {"value": -1.42, "error": 6.0e-3},
        {"value": -1.34, "error": 6.0e-3},
        {"value": -1.33, "error": 4.0e-3},
        {"value": -0.17, "error": 2.0e-3},
        {"value": -1.24, "error": 0.01}
    ],
    'NDP': [
        {'value': 0.71, 'error': 0.02},
        {'value': 0.31, 'error': 0.02},
        {'value': -0.52, 'error': 0.028},
        {'value': 3.39, 'error': 0.026},
        {'value': 0.39, 'error': 0.023},
    ]
}

#%%
all_results = {
    "interpolation": results_interpolation,
    "generalization": results_generalization,
}

data = []
for i, task in enumerate(TASKS):
    for j, dataset in enumerate(DATASETS):
        for model in MODELS:
            row = {
                "task": task,
                "dataset": dataset,
                "model": model,
                **all_results[task][model][j],
            }
            data.append(row)

#%%

df = pd.DataFrame(data)
df = df.groupby(["task", "dataset", "model"]).agg({"value": ["first"], "error": ["first"]})
df = df.unstack(1).reset_index()
df.columns = df.columns.droplevel(1)
df
#%%


# for k, v in all_results.items():
# check that all results contain the same models
# assert set(v.keys()) == set(all_results["interpolation"].keys())


# models = list(all_results["interpolation"].keys())
# for task, results in all_results.items():
#     for model, values in results.items():
#         assert len(values) == len(DATASETS)


def format_number(value, error, *, bold=False, possibly_negative=True):
    if value is None or np.isnan(value):
        return "n/a"
    if np.abs(value) > 10:
        return "F"
    if value >= 0 and possibly_negative:
        sign_spacer = "\\hphantom{-}"
    else:
        sign_spacer = ""
    if bold:
        bold_start, bold_end = "\\mathbf{", "}"
    else:
        bold_start, bold_end = "", ""
    if error is None:
        return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\scriptstyle X }}$"
    else:
        return f"${sign_spacer}{bold_start}{value:.2f}{bold_end} {{ \\pm \\scriptstyle {error:.2f} }}$"
    

def in_statistically_n_best(value, std_error, data_list, std_error_list):
    idx_best = np.nanargmax(data_list)
    v0, e0 = data_list[idx_best], std_error_list[idx_best]
    return abs(value - v0) < np.sqrt(std_error**2 + e0**2)

# \begin{table}[h]
# \small
# \centering
table = r"""
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{1}{c}{\scshape SE} & \multicolumn{1}{c}{\scshape Mat\'ern--$\frac52$} & \multicolumn{1}{c}{\scshape Weakly Per.} & \multicolumn{1}{c}{\scshape Sawtooth} & \multicolumn{1}{c}{\scshape Mixture}\\
"""

task_title = "\\midrule\\multicolumn{{6}}{{l}}{{\\textsc{{{task}}}}} \\\\[0.5em]"

table_end = r"""
\bottomrule
\end{tabular}
"""
# \end{table}

lines = []

for task in TASKS:
    lines.append(task_title.format(task=display_name(task)))
    for model in MODELS:
        line = r"\scshape " + display_name(model)
        for dataset in DATASETS:
            value = df.loc[(df["task"] == task) & (df["model"] == model), ("value", dataset)].values[0]
            error = df.loc[(df["task"] == task) & (df["model"] == model), ("error", dataset)].values[0]
            all_values = df.loc[(df["task"] == task) & (df["model"] != "GP"), ("value", dataset)].values
            all_errors = df.loc[(df["task"] == task) & (df["model"] != "GP"), ("error", dataset)].values
            possibly_negative = any(all_values < 0)
            bold = (model != "GP") and in_statistically_n_best(value, error, all_values, all_errors)
            line += "& " + format_number(value, error, possibly_negative=True, bold=bold)
        line += r"\\"
        lines.append(line)

print(table)
lines = "\n".join(lines)
print(lines)
print(table_end)
