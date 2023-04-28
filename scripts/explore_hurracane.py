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
# raw_data = pd.read_csv("/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/2020-23.csv")
raw_data = pd.read_csv("/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/all.csv")
raw_data = raw_data.drop(0)
raw_data
# %%
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
for col in data.columns:
    print(col, data[col].dtype)
data
# %%
initial_points = data[
    data.apply(lambda row: row["STORM_TIME"].asm8 == 0, axis=1)
].reset_index()
# %%
m = Basemap()
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
m.scatter(
    *m(initial_points["LON"], initial_points['LAT']), s=1
)   
# %%
sid = data["SID"].unique()[1]

m = Basemap()
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
for sid in data["SID"].unique()[:100]:
    track = data[data["SID"] == sid]
    m.plot(
        track["LON"], track['LAT']
    )
# %%
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
time_index = pivot_data.columns.levels[1]
prop_na = pivot_data.apply(lambda col: np.sum(col.isna())/len(col))
# keep_times = prop_na["LON"][prop_na["LON"] < 0.98].index
keep_times = pivot_data["LAT"].columns[(pivot_data["LAT"].columns.values.astype(int) % 10800000000000) == 0]
pivot_data = pivot_data.loc[:, idx[:, keep_times]] # Makes strom time points contiguous. Maybe want to move to the 3 hourly thing
plt.imshow(pivot_data.isnull().values, aspect=0.03)
# %%
plt.hist((~pivot_data['LAT'].isnull().values).sum(axis=1)*3/24)
plt.xlabel("Storm length (days)")
# %%
m = Basemap()
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
for idx, row in pivot_data.iterrows():
    m.plot(
        row["LON"], row['LAT'], linewidth=.3
    )
plt.savefig("hurricane_tracks.png", dpi=300, bbox_inches='tight')
# %%
pivot_data.to_csv("/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/all_processes.csv")
# %%
dataset = pd.read_csv("/data/ziz/not-backed-up/mhutchin/score-sde-sp/data/storm/all_processes.csv", header=[0,1])
lats = dataset["LAT"].values.astype(np.float32)
lons = dataset["LON"].values.astype(np.float32)
storm_length = (~np.isnan(lats)).sum(axis=1)
# pre_pad = lats.shape[1] // 2
# nan_array = np.empty((lats.shape[0], pre_pad))
# nan_array[:] = np.nan
# lats = np.concatenate(
#     [nan_array, lats], axis=1
# )
# lons = np.concatenate(
#     [nan_array, lons], axis=1
# )
data_mask = ~np.isnan(lats)
lonlats = np.stack([lons, lats], axis=-1)
# %%
plt.hist(storm_length)

# %%
min_trajectory_length = 10
max_trajectory_legnth = 50

index = np.arange(len(storm_length))
trajectory_length = 25
batch_size = 32

storms_selected = np.random.choice(index[storm_length >= trajectory_length], size=batch_size)
storms_start = np.random.randint(0, storm_length[storms_selected] - trajectory_length + 1)

sample = jax.vmap(lambda storm, start, length: jax.lax.dynamic_slice(
    jnp.array(storm), 
    (start,), 
    (length,),
), in_axes=(0,0,None))(lats[storms_selected], storms_start, trajectory_length)


# %%
lens = jnp.arange(lonlats.shape[1])+1
viable_storms = jax.vmap(lambda len: np.sum(storm_length>=len))(lens)
pick_storm_prob = batch_size / viable_storms
pick_time_prob = jax.vmap(jax.vmap(lambda len, storm_len: jnp.where(len <= storm_len, len /storm_len, 0), in_axes=(0, None)), in_axes=(None, 0))(lens, storm_length)
pick_time_and_storm_prob = pick_time_prob * pick_storm_prob[None, :]
pick_time_and_storm_prob = pick_time_and_storm_prob[:, min_trajectory_length:(max_trajectory_legnth + 1)]
mp_inv = np.linalg.inv(pick_time_and_storm_prob.T @ pick_time_and_storm_prob) @ pick_time_and_storm_prob.T
p_len = mp_inv @  (1/np.sum(~np.isnan(lats)) * np.ones((mp_inv.shape[1], 1)))
p_len = p_len/p_len.sum()
plt.plot(np.arange(min_trajectory_length, max_trajectory_legnth + 1), p_len)
# %%
# copy_data = lats[:100]
copy_data = np.arange(1000).astype(np.float32).reshape((10,100))
data_index = np.arange(np.sum(~np.isnan(copy_data)))
fake_data = np.empty_like(copy_data)
fake_data[:] = np.nan
fake_data[~np.isnan(copy_data)] = data_index
fake_data = fake_data[:, :np.sum(~np.isnan(fake_data), axis=1).max()]


def sample_batch(data, p_len, lens, data_len, batch_size):
    len = np.random.choice(lens, p=p_len)
    index = np.arange(data.shape[0])
    data_selected = np.random.choice(index[data_len >= len], size=batch_size)
    data_start = np.random.randint(0, data_len[data_selected] - len+1)
    # data_start = np.random.choice()
    sample = jax.vmap(lambda data, start, length: jax.lax.dynamic_slice(
        jnp.array(data), 
        (start,), 
        (length,),
    ), in_axes=(0,0,None))(data[data_selected], data_start, len)
    return sample

np.isnan(sample_batch(
    fake_data,
    np.ones(fake_data.shape[1]) / fake_data.shape[1],
    jnp.arange(fake_data.shape[1])+1,
    np.sum(~np.isnan(fake_data), axis=1),
    32
)).sum()

# %%

counts = jnp.zeros_like(data_index)

for i in range(1000):
    sample = sample_batch(
        fake_data,
        np.ones(fake_data.shape[1]) / fake_data.shape[1],
        jnp.arange(fake_data.shape[1])+1,
        np.sum(~np.isnan(fake_data), axis=1),
        32
    )

    counts += jax.vmap(lambda sample, data_index: np.sum(sample == data_index), in_axes=(None, 0))(sample, data_index)

# plt.plot(counts)

fake_counts = np.empty_like(fake_data)
fake_counts[:] = np.nan
fake_counts[~np.isnan(fake_data)] = counts

plt.imshow(fake_counts, interpolation=None)
# %%
choice = np.array([
    [1,0,0,0,0],
    [1,1,0,0,0],
    [1,1,1,0,0],
    [0,1,1,1,0],
    [0,0,1,1,1],
    [0,0,0,1,1],
    [0,0,0,0,1]]
, dtype=np.float32)

mp_inv = np.linalg.inv(choice.T @ choice) @ choice.T
p = mp_inv @ ((1/5) * jnp.ones((7,)))
# %%
