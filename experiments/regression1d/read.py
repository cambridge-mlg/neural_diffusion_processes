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

mat = "Apr16_171012_7063364153930192415"
weakp = "Apr16_171404_1789127319741651408"
filename = f"{weakp}/metrics.csv"
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

