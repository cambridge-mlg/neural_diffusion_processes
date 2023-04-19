from __future__ import annotations
from typing import Tuple, Iterator, Optional, Mapping
import dataclasses

import jax
import jax.numpy as jnp
from simple_pytree import Pytree
from check_shapes import check_shapes, check_shape
from jaxtyping import Array

@dataclasses.dataclass
class DataBatch(Pytree):
    xs: Array
    ys: Array
    xc: Array | None = None
    yc: Array | None = None

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def num_points(self) -> int:
        return self.xs.shape[1]

    # @check_shapes()
    # def __post_init__(self) -> None:
    #     check_shape(self.xs, "[batch, num_points, input_dim]")
    #     check_shape(self.ys, "[batch, num_points, output_dim]")


@check_shapes(
    "data[0]: [len_data, num_points, input_dim]",
    "data[1]: [len_data, num_points, output_dim]",
)
def dataloader(
    data: Tuple[Array, Array], batch_size: int, *, key, run_forever=True, n_points=-1
) -> Iterator[DataBatch]:
    """Yields minibatches of size `batch_size` from the data."""
    x, y = data
    n_points = n_points if n_points > 0 else x.shape[1]
    # x = x.astype(jnp.float32)
    # y = y.astype(jnp.float32)
    dataset_size = len(x)
    indices_batch = jnp.arange(dataset_size)
    indices_points = jnp.arange(x.shape[1])
    while True:
        perm = jax.random.permutation(key, indices_batch)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            (key,) = jax.random.split(key, 1)
            points_perm = jax.random.permutation(key, indices_points)[:n_points]
            yield DataBatch(
                xs=jnp.take(x[batch_perm], axis=1, indices=points_perm),
                ys=jnp.take(y[batch_perm], axis=1, indices=points_perm),
                )
            start = end
            end = start + batch_size

        if not run_forever:
            break


def split_dataset_in_context_and_target(data: DataBatch, key, min_context, max_context) -> DataBatch:
    if key is None:
        key = jax.random.PRNGKey(0)

    key1, key2 = jax.random.split(key)
    x, y = data.xs, data.ys
    indices = jnp.arange(data.num_points)
    num_context = jax.random.randint(key1, (), minval=min_context, maxval=max_context)
    num_target = data.num_points - num_context
    perm = jax.random.permutation(key2, indices)
    # print("split_dataset_in_context_and_target")
    # print("x.shape", x.shape)
    # print("num_target", num_target, "num_target", num_target)
    # print(jnp.take(x, axis=1, indices=perm[:num_target]).shape, jnp.take(x, axis=1, indices=perm[-num_context:]).shape)
    # print(jnp.take(x, axis=1, indices=perm[num_context:]).shape, jnp.take(x, axis=1, indices=perm[:num_context]).shape)
    return DataBatch(
        # xs=jnp.take(x, axis=1, indices=perm[:num_target]),
        # ys=jnp.take(y, axis=1, indices=perm[:num_target]),
        # xc=jnp.take(x, axis=1, indices=perm[-num_context:]),
        # yc=jnp.take(y, axis=1, indices=perm[-num_context:]),
        xs=jnp.take(x, axis=1, indices=perm[num_context:]),
        ys=jnp.take(y, axis=1, indices=perm[num_context:]),
        xc=jnp.take(x, axis=1, indices=perm[:num_context]),
        yc=jnp.take(y, axis=1, indices=perm[:num_context]),
    )
