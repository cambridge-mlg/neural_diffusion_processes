# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# %%
def process_file(basin):
    raw_data = pd.read_csv(f"/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/{basin}.csv")
    raw_data = raw_data.drop(0)

    data = raw_data[
        [
        "SID",
        "NAME",
        "ISO_TIME",
        "NATURE",
        "LAT",
        "LON",
        "TRACK_TYPE",
        "USA_SSHS", # For category
        "USA_WIND"
        ]
    ]
    # data=data.dropna()
    data = data.astype({
        "SID": str,
        "NAME": str,
        # "ISO_TIME": np.float32,
        "NATURE": str,
        "LAT": np.float32,
        "LON": np.float32,
        "TRACK_TYPE":str,
        "USA_SSHS": int, 
    })
    data.loc[:, "LAT"] = data.loc[:, "LAT"].astype(np.float32)
    data.loc[:, "LON"] = data.loc[:, "LON"].astype(np.float32)
    data = data[data['TRACK_TYPE'] == 'main'].reset_index(drop=True) # Filter only reanalysed storm data
    data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])
    storm_starts = data.groupby(by='SID').min()["ISO_TIME"]
    data["STORM_TIME"] = data.apply(lambda row: row["ISO_TIME"] - storm_starts[row["SID"]], axis=1)

    idx = pd.IndexSlice

    pivot_data = data.pivot(
        index=[
            "SID",
            "NAME",
        ],
        columns="STORM_TIME",
        values=[
            "LAT",
            "LON",
            "USA_SSHS", # For category
            "USA_WIND",
            "NATURE",
        ]
    )

    keep_times = pivot_data["LAT"].columns[(pivot_data["LAT"].columns.values.astype(int) % 10800000000000) == 0]
    pivot_data = pivot_data.loc[:, idx[:, keep_times]] # Makes strom time points contiguous. Maybe want to move to the 3 hourly thing
  
    pivot_data.to_csv(f"/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/{basin}_processed.csv")
# %%
process_file('wp')
# %%
for basin in [
    'all',
    'ep',
    # 'na',
    'ni',
    'sa',
    'si',
    'sp',
    'wp',
]:
    process_file(basin)