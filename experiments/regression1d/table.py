import numpy as np

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
    "extrapolation",
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
    'GP (diag.)': [
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
    "GP (diag.)": [
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
    ]
}


results_extrapolation = {
    'GP': [
        {'value': 0.44, 'error': 2.9e-3},
        {'value': 0.09, 'error': 3.1e-3},
        {'value': -0.52, 'error': 3.4e-3},
        {'value': None, 'error': 'n/a'},
        {'value': None, 'error': 'n/a'}
    ],
    'GP (diag.)': [
        {'value': -1.40, 'error': 6.7e-3},
        {'value': -1.41, 'error': 6.6e-3},
        {'value': -1.41, 'error': 5.6e-3},
        {'value': None, 'error': 'n/a'},
        {'value': None, 'error': 'n/a'}
    ],
    'GNP': [
        {'value': 0.44, 'error': 3.0e-3},
        {'value': 0.08, 'error': 3.0e-3},
        {'value': -0.62, 'error': 4.0e-3},
        {'value': 0.04, 'error': 9.0e-3},
        {'value': -0.07, 'error': 0.01}
    ],
    'ConvCNP': [
        {'value': -1.41, 'error': 7.0e-3},
        {'value': -1.42, 'error': 6.0e-3},
        {'value': -1.41, 'error': 6.0e-3},
        {'value': 0.06, 'error': 8.0e-3},
        {'value': -1.36, 'error': 0.02}
    ],
    'ConvNP': [
        {'value': -1.11, 'error': 5.0e-3},
        {'value': -1.12, 'error': 5.0e-3},
        {'value': -1.23, 'error': 4.0e-3},
        {'value': 0.88, 'error': 9.0e-3},
        {'value': -0.93, 'error': 0.01}
    ],
    'ANP': [
        {'value': -1.31, 'error': 5.0e-3},
        {'value': -1.28, 'error': 5.0e-3},
        {'value': -1.32, 'error': 5.0e-3},
        {'value': -0.17, 'error': 1.0e-3},
        {'value': -1.11, 'error': 0.01}
    ]
}
all_results = {
    "interpolation": results_interpolation,
    "generalization": results_generalization,
    "extrapolation": results_extrapolation,
}

for k, v in all_results.items():
    # check that all results contain the same models
    assert set(v.keys()) == set(all_results["interpolation"].keys())


models = list(all_results["interpolation"].keys())
for task, results in all_results.items():
    for model, values in results.items():
        assert len(values) == len(DATASETS)


def format_number(value, error, *, bold=False, possibly_negative=True):
    if value is None:
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
    

def parse_results(results):
    lines = []
    for model, values in results.items():
        line = r"\scshape " + model
        for entry in values:
            line += "& " + format_number(entry["value"], entry["error"], possibly_negative=True)
        line += r"\\"
        lines.append(line)
    return lines


table = r"""
\begin{table}[h]
\small
\centering
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{1}{c}{\scshape SE} & \multicolumn{1}{c}{\scshape Mat\'ern--$\frac52$} & \multicolumn{1}{c}{\scshape Weakly Per.} & \multicolumn{1}{c}{\scshape Sawtooth} & \multicolumn{1}{c}{\scshape Mixture}\\
"""

task_title = "\\midrule\\multicolumn{{6}}{{l}}{{\\textsc{{{task}}}}} \\\\[0.5em]"

table_end = r"""
\bottomrule
\end{tabular}
\end{table}
"""

lines = []
for task, results in all_results.items():
    lines.append(task_title.format(task=task))
    lines.extend(parse_results(results))


print(table)
lines = "\n".join(lines)
print(lines)
# for l in lines:
#     print(l)
print(table_end)


