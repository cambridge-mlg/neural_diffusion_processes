from __future__ import annotations
from typing import Tuple, Iterator, Optional, Mapping

import dataclasses
import jax
import jax.numpy as jnp
import jaxkern
from check_shapes import check_shapes, check_shape

from jaxtyping import Array

from .misc import sample_mvn


def get_gp_data(key, kernel: jaxkern.base.AbstractKernel, num_samples: int, *, x_range=(-1., 1.), num_points: int = 100, input_dim: int = 1, output_dim:int = 1, params: Optional[Mapping[str, float]] = None):
    """
    Returns tuple of inputs and outputs. The outputs are drawn from a GP prior with a fixed kernel.
    """
    assert input_dim == 1
    assert output_dim == 1

    if params is None:
        params = {
            'lengthscale': 0.2,
            'variance': 1.0,
        }

    def sample_single(key):
        input_key, output_key = jax.random.split(key, 2)
        x = jax.random.uniform(input_key, [num_points, 1], minval=x_range[0], maxval=x_range[1], dtype=jnp.float64)
        x = x.sort(axis=0)
        gram = kernel.gram(params, x).to_dense()
        y = sample_mvn(output_key, jnp.zeros_like(x), gram)
        return x, y
    
    x, y = jax.vmap(sample_single)(jax.random.split(key, num_samples))
    return x, y


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class DataBatch:
    function_inputs: Array
    function_outputs: Array
    context_inputs: Array | None = None
    context_outputs: Array | None = None

    def __len__(self) -> int:
        return len(self.function_inputs)
    
    @property
    def num_points(self) -> int:
        return self.function_inputs.shape[1]

    @check_shapes()
    def __post_init__(self) -> None:
        check_shape(self.function_inputs, "[batch, num_points, input_dim]")
        check_shape(self.function_outputs, "[batch, num_points, output_dim]")

    def tree_flatten(self):
        children = (self.function_inputs, self.function_outputs)
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
def dataloader(data: Tuple[Array, Array], batch_size: int, *, key, run_forever=True) -> Iterator[DataBatch]:
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
            yield DataBatch(function_inputs=x[batch_perm], function_outputs=y[batch_perm])
            start = end
            end = start + batch_size

        if not run_forever:
            break



def split_dataset_in_context_and_target(data: DataBatch, key) -> DataBatch:
    if key is None:
        key = jax.random.PRNGKey(0)

    key1, key2 = jax.random.split(key)
    x, y = data.function_inputs, data.function_outputs
    indices = jnp.arange(data.num_points)
    num_context = jax.random.randint(key1, (), minval=4, maxval=20)
    num_target = data.num_points - num_context
    perm = jax.random.permutation(key2, indices)
    return DataBatch(
        function_inputs=jnp.take(x, axis=1, indices=perm[:num_target]),
        function_outputs=jnp.take(y, axis=1, indices=perm[:num_target]),
        context_inputs=jnp.take(x, axis=1, indices=perm[-num_context:]),
        context_outputs=jnp.take(y, axis=1, indices=perm[-num_context:]),
    )
