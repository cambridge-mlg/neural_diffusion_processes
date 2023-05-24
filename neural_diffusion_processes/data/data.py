from __future__ import annotations
from typing import Tuple, Iterator, Optional, Mapping, Union, Sequence
import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
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
    data: Tuple[Array, Array],
    batch_size: int,
    *,
    key,
    run_forever=True,
    n_points=[-1],
    shuffle_xs=True,
) -> Iterator[DataBatch]:
    """Yields minibatches of size `batch_size` from the data."""
    x, y = data
    n_points = jnp.array(list(n_points))
    dataset_size = len(x)
    indices_batch = jnp.arange(dataset_size)
    indices_points = jnp.arange(x.shape[1])
    if dataset_size >= batch_size:
        while True:
            perm = jr.permutation(key, indices_batch)
            (key,) = jr.split(key, 1)
            start = 0
            end = batch_size
            while end <= dataset_size:
                batch_perm = perm[start:end]
                (key,) = jr.split(key, 1)
                if shuffle_xs:
                    points_perm = jax.random.permutation(key, indices_points)
                else:
                    points_perm = indices_points
                (key,) = jr.split(key, 1)
                n_point = jr.permutation(key, n_points)[0]
                n_point = n_point if n_point > 0 else x.shape[1]
                if n_points > 0:
                    points_perm = points_perm[:n_points]
                yield DataBatch(
                    xs=jnp.take(x[batch_perm], axis=1, indices=points_perm),
                    ys=jnp.take(y[batch_perm], axis=1, indices=points_perm),
                )
                start = end
                end = start + batch_size

            if not run_forever:
                break
    else:
        while True:
            if not run_forever:
                break
            batch_perm = jr.randint(key, (batch_size,), minval=0, maxval=dataset_size)
            (key,) = jr.split(key, 1)
            if shuffle_xs:
                points_perm = jax.random.permutation(key, indices_points)
            else:
                points_perm = indices_points
            (key,) = jr.split(key, 1)
            n_point = jr.permutation(key, n_points)[0]
            n_point = n_point if n_point > 0 else x.shape[1]
            if n_points > 0:
                points_perm = points_perm[:n_points]

            yield DataBatch(
                xs=jnp.take(x[batch_perm], axis=1, indices=points_perm),
                ys=jnp.take(y[batch_perm], axis=1, indices=points_perm),
            )


def split_dataset_in_context_and_target(
    data: DataBatch, key, min_context, max_context
) -> DataBatch:
    if key is None:
        key = jr.PRNGKey(0)

    key1, key2 = jr.split(key)
    x, y = data.xs, data.ys
    indices = jnp.arange(data.num_points)
    num_context = jr.randint(key1, (), minval=min_context, maxval=max_context)
    num_target = data.num_points - num_context
    perm = jr.permutation(key2, indices)
    return DataBatch(
        xs=jnp.take(x, axis=1, indices=perm[num_context:]),
        ys=jnp.take(y, axis=1, indices=perm[num_context:]),
        xc=jnp.take(x, axis=1, indices=perm[:num_context]),
        yc=jnp.take(y, axis=1, indices=perm[:num_context]),
    )


def shuffle_data(key_or_seed: Union[int, jax.random.KeyArray], data):
    if isinstance(key_or_seed, int):
        key = jax.random.PRNGKey(key_or_seed)
    else:
        key = key_or_seed

    perm = jax.random.permutation(key, len(data[0]))

    return (data[0][perm], data[1][perm])


def split_data(data, proportions):
    n = len(data[0])
    lengths = [0]
    for p in proportions[:-1]:
        lengths.append(int(n * p))
    lengths.append(n - sum(lengths))
    split_points = jnp.cumsum(jnp.array(lengths, dtype=int))
    return [
        (data[0][i1:i2], data[1][i1:i2])
        for i1, i2 in zip(split_points[:-1], split_points[1:])
    ]
