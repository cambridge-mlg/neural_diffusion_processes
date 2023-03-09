import os

import jax
import jax.numpy as jnp

from einops import rearrange


class ERA5Dataset:
    def __init__(self, batch_size, data_dir, file_ending, rng=jax.random.PRNGKey(0)):
        self.batch_size = batch_size

        self.data = rearrange(
            jnp.load(os.path.join(data_dir, f"data_{file_ending}.npy")),
            "n h w d -> n (h w) d",
        )
        self.time = jnp.load(
            os.path.join(data_dir, f"time_{file_ending}.npy"), allow_pickle=True
        )
        self.latlon = rearrange(
            jnp.load(os.path.join(data_dir, f"latlon_{file_ending}.npy")),
            "h w d -> (h w) d",
        )

        self.rng = rng

    def __len__(self):
        return self.data.shape[0]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        indices = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self.data[indices], jnp.repeat(
            self.latlon[None, ...], axis=0, repeats=self.batch_size
        )
