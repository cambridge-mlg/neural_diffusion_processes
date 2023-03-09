# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# %%
from neural_diffusion_processes.data import ERA5Dataset
ds = ERA5Dataset(10, "/scratch/ebm32/score-sde/data/era5/us", "ERA5_US")
# %%
y, x = next(ds)


# %%
i = 5
plt.scatter(
    x[i, :, 0],
    x[i, :, 1],
    c=y[i, :, 4],
)
plt.quiver(
    x[i, :, 0],
    x[i, :, 1],
    y[i, :, 2],
    y[i, :, 3],
)
plt.colorbar()
plt.gca().set_aspect('equal')
# %%