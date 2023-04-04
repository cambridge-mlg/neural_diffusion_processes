#%%
import jax.numpy as jnp
from neural_diffusion_processes.kernels import WhiteKernel
from neural_diffusion_processes.utils import Array, Scalar

from check_shapes import check_shapes

# %%
k = WhiteKernel(2.0)
k(jnp.linspace(-1, 1, 3)[:, None])
k(jnp.linspace(-1, 1, 3)[:, None])

# %%
k(jnp.linspace(-1, 1, 3)[:, None])

# %%
