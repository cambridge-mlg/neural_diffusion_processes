from pathlib import Path
import pandas as pd
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
print(df)

# filter columns out that have only one value
df = df.loc[:, df.nunique() != 1]
