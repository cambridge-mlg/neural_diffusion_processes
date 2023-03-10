from __future__ import annotations
from typing import Tuple, Iterator, Optional, Mapping
import dataclasses

import jax
import jax.numpy as jnp
from check_shapes import check_shapes, check_shape
from jaxtyping import Array


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DataBatch:
    xs: Array
    ys: Array
    xc: Array | None = None
    yc: Array | None = None

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def num_points(self) -> int:
        return self.xs.shape[1]

    @check_shapes()
    def __post_init__(self) -> None:
        check_shape(self.xs, "[batch, num_points, input_dim]")
        check_shape(self.ys, "[batch, num_points, output_dim]")

    def tree_flatten(self):
        children = (self.xs, self.ys)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


@check_shapes(
    "data[0]: [len_data, num_points, input_dim]",
    "data[1]: [len_data, num_points, output_dim]",
)
def dataloader(
    data: Tuple[Array, Array], batch_size: int, *, key, run_forever=True
) -> Iterator[DataBatch]:
    """Yields minibatches of size `batch_size` from the data."""
    x, y = data
    dataset_size = len(x)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield DataBatch(xs=x[batch_perm], ys=y[batch_perm])
            start = end
            end = start + batch_size

        if not run_forever:
            break


def split_dataset_in_context_and_target(data: DataBatch, key) -> DataBatch:
    if key is None:
        key = jax.random.PRNGKey(0)

    key1, key2 = jax.random.split(key)
    x, y = data.xs, data.ys
    indices = jnp.arange(data.num_points)
    num_context = jax.random.randint(key1, (), minval=4, maxval=20)
    num_target = data.num_points - num_context
    perm = jax.random.permutation(key2, indices)
    return DataBatch(
        xs=jnp.take(x, axis=1, indices=perm[:num_target]),
        ys=jnp.take(y, axis=1, indices=perm[:num_target]),
        xc=jnp.take(x, axis=1, indices=perm[-num_context:]),
        yc=jnp.take(y, axis=1, indices=perm[-num_context:]),
    )
