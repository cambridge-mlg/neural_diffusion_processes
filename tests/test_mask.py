import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d


@pytest.fixture
def dataset():
    key = jax.random.PRNGKey(0)
    dataset = "se"
    task = "training"
    total_num_samples = int(2**14)
    batch_size = 2
    return regression1d.get_dataset(dataset, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=None)


def test_masking(dataset):

    key = jax.random.PRNGKey(0)

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = ndp.models.attention.BiDimensionalAttentionModel(n_layers=5, hidden_dim=64, num_heads=8, init_zero=False)
        return model(x, y, t, mask)

    batch: regression1d.DataBatch = next(dataset)
    t = 1. * jnp.zeros((batch.ys.shape[0]))
    params = network.init(key, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)

    @jax.jit
    def func(params, x, y, mask):
        print("compiling func:", batch.xs.shape)
        t = 0.5 * jnp.ones((batch.ys.shape[0]))
        o = network.apply(params, t, y, x, mask)
        return jnp.sum(jnp.where(batch.mask[..., None] == 0.0, o, jnp.zeros_like(o)))

    o1 = func(params, batch.xs, batch.ys, batch.mask)

    # corrupt masked values to check that they do *not* affect output
    xs = jnp.where(batch.mask[..., None] == 1.0, -666., batch.xs)
    ys = jnp.where(batch.mask[..., None] == 1.0, -666., batch.ys)
    o2 = func(params, xs, ys, batch.mask)

    # corrupt un-masked values to check that they do affect output
    xs = jnp.where(batch.mask[..., None] == 0.0, -666., batch.xs)
    ys = jnp.where(batch.mask[..., None] == 0.0, -666., batch.ys)
    o3 = func(params, xs, ys, batch.mask)

    assert (o1 - o2)**2 < 1e-9
    assert (o1 - o3)**2 > 1e-1


def test_mask_is_none(dataset):

    key = jax.random.PRNGKey(0)

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = ndp.models.attention.BiDimensionalAttentionModel(n_layers=5, hidden_dim=64, num_heads=8, init_zero=False)
        return model(x, y, t, mask)

    batch: regression1d.DataBatch = next(dataset)
    t = 1. * jnp.zeros((batch.ys.shape[0]))
    params = network.init(key, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)

    @jax.jit
    def func(params, x, y, mask):
        print("compiling func:", batch.xs.shape)
        t = 0.5 * jnp.ones((batch.ys.shape[0]))
        return network.apply(params, t, y, x, mask)

    o1 = func(params, batch.xs, batch.ys, jnp.zeros_like(batch.mask))
    o2 = func(params, batch.xs, batch.ys, None)
    np.testing.assert_array_almost_equal(o1, o2)


if __name__ == "__main__":
    test_mask_is_none()