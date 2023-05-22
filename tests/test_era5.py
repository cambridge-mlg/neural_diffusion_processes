# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import setGPU
# os.environ["GEOMSTATS_BACKEND"] = "jax"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# %%
from neural_diffusion_processes.data import ERA5Dataset, ERA5Dataloader
x, y = ERA5Dataset(None, "/scratch/ebm32/score-sde/data/era5/us", "ERA5_US")
print(x.shape)
print(y.shape)

# x = x[:, ::2, :]
# y = y[:, ::2, :]

i = 500
plt.scatter(
    x[i, :, 0],
    x[i, :, 1],
    c=y[i, :, 0],
)
# plt.show()
plt.quiver(
    x[i, :, 0],
    x[i, :, 1],
    y[i, :, 2],
    y[i, :, 3],
    # scale = 10.

)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.show()

# %%
dl = ERA5Dataloader(10, "/scratch/ebm32/score-sde/data/era5/us", "ERA5_US")
y, x = next(dl)


i = 1
plt.scatter(
    x[i, :, 0],
    x[i, :, 1],
    c=y[i, :, 4],
)
# plt.show()
plt.quiver(
    x[i, :, 0],
    x[i, :, 1],
    y[i, :, 2],
    y[i, :, 3],
    # scale = 10.

)
plt.colorbar()
plt.gca().set_aspect('equal')
plt.show()
# %%