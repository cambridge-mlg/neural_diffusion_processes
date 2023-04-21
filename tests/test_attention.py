import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

import gpjax
import jaxkern

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d


def get_batch():
    key = jax.random.PRNGKey(0)
    dataset = "se"
    task = "training"
    total_num_samples = int(2**14)
    batch_size = 2
    ds = regression1d.get_dataset(dataset, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=None)
    return next(ds)

    
@pytest.fixture
def network(request):
    init_key = jax.random.PRNGKey(0)

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = ndp.models.attention.BiDimensionalAttentionModel(
            n_layers=5, hidden_dim=64, num_heads=8, init_zero=False,
            translation_invariant=request.param,
        )
        return model(x, y, t, mask)

    batch = get_batch()
    t = 1. * jnp.zeros((batch.ys.shape[0]))
    params = network.init(init_key, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)
    return lambda *args: network.apply(params, *args)


@pytest.mark.parametrize(
        'network',
        [True, pytest.param(False, marks=pytest.mark.xfail)],
        indirect=['network']
)
def test_translation_invariance(network):
    batch = get_batch()
    t = 1. * jnp.zeros((batch.ys.shape[0]))
    out1 = network(t, batch.ys, batch.xs, batch.mask)
    out2 = network(t, batch.ys, batch.xs + 2., batch.mask)
    np.testing.assert_array_almost_equal(out1, out2)
