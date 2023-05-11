import os

import jax
import jax.numpy as jnp

from einops import rearrange


class ERA5Dataloader:
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

        print("data", self.data.shape)
        self.rng = rng

    def __len__(self):
        return self.data.shape[0]

    def __next__(self):
        self.rng, next_rng = jax.random.split(self.rng)
        indices = jax.random.choice(next_rng, len(self), shape=(self.batch_size,))

        return self.data[indices], jnp.repeat(
            self.latlon[None, ...], axis=0, repeats=self.batch_size
        )


def ERA5Dataset(key, data_dir, file_ending, dataset="train", **kwargs):
    data = rearrange(
        jnp.load(os.path.join(data_dir, f"data_{file_ending}.npy")),
        "n h w d -> n (h w) d",
    )
    # time = jnp.load(
    #     os.path.join(data_dir, f"time_{file_ending}.npy"), allow_pickle=True
    # )
    latlon = rearrange(
        jnp.load(os.path.join(data_dir, f"latlon_{file_ending}.npy")),
        "h w d -> (h w) d",
    )

    y = data[..., 2:]
    x = jnp.repeat(latlon[None, ...], data.shape[0], 0)
    y = jax.nn.normalize(y, (0, 1))
    x = jax.nn.normalize(x, (0, 1))
    if dataset == "train":
        x = x[:3500]
        y = y[:3500]
    else:
        x = x[3500:]
        y = y[3500:]

    print("x", x.shape)
    print("y", y.shape)
    return x, y
