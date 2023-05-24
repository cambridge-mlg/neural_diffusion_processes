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


exp = "Apr16_181930_se_10841"
filename = f"{exp}/metrics.csv"
path = _HERE / _LOG_DIR / filename

df = pd.read_csv(path, index_col=0)
df = df.groupby("step").first().reset_index()


def get_metrics(task):
    c1 = f'{task}_loglik_mean'
    c2 = f'{task}_loglik_err'
    print(f"{df[c1].iloc[-1]} ({df[c2].iloc[-1]})")


for task in TASKS:
    print(task)
    get_metrics(task)

