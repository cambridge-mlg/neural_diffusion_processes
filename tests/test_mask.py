import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

import gpjax
import jaxkern

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


@pytest.fixture
def batch(dataset):
    return next(dataset)

    
@pytest.fixture
def sde():
    beta = ndp.sde.LinearBetaSchedule()
    limiting_kernel = jaxkern.White()
    hyps = {
        "mean_function": {},
        "kernel": limiting_kernel.init_params(None),
    }
    return ndp.sde.SDE(
        limiting_kernel,
        gpjax.mean_functions.Zero(),
        hyps,
        beta,
        is_score_preconditioned=False,
        std_trick=False,
        residual_trick=False,
        exact_score=False,
    )


@pytest.fixture
def network(dataset):
    init_key = jax.random.PRNGKey(0)

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = ndp.models.attention.BiDimensionalAttentionModel(n_layers=5, hidden_dim=64, num_heads=8, init_zero=False)
        return model(x, y, t, mask)

    batch: regression1d.DataBatch = next(dataset)
    t = 1. * jnp.zeros((batch.ys.shape[0]))
    params = network.init(init_key, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)
    return lambda *args: network.apply(params, *args)



def test_masking(batch, network):
    """Corrupts masked values and checks that output of network hasn't been affected."""

    @jax.jit
    def func(x, y, mask):
        print("compiling func:", batch.xs.shape)
        t = 0.5 * jnp.ones((batch.ys.shape[0]))
        o = network(t, y, x, mask)
        return jnp.sum(jnp.where(batch.mask[..., None] == 0.0, o, jnp.zeros_like(o)))

    o1 = func(batch.xs, batch.ys, batch.mask)

    # corrupt masked values to check that they do *not* affect output
    xs = jnp.where(batch.mask[..., None] == 1.0, -666., batch.xs)
    ys = jnp.where(batch.mask[..., None] == 1.0, -666., batch.ys)
    o2 = func(xs, ys, batch.mask)

    # corrupt un-masked values to check that they do affect output
    xs = jnp.where(batch.mask[..., None] == 0.0, -666., batch.xs)
    ys = jnp.where(batch.mask[..., None] == 0.0, -666., batch.ys)
    o3 = func(xs, ys, batch.mask)

    assert (o1 - o2)**2 < 1e-9
    assert (o1 - o3)**2 > 1e-1


def test_mask_is_none(batch, network):
    """Tests that using mask None is equal to a mask with all zeros."""

    @jax.jit
    def func(x, y, mask):
        print("compiling func:", batch.xs.shape)
        t = 0.5 * jnp.ones((batch.ys.shape[0]))
        return network(t, y, x, mask)

    o1 = func(batch.xs, batch.ys, jnp.zeros_like(batch.mask))
    o2 = func(batch.xs, batch.ys, None)
    np.testing.assert_array_almost_equal(o1, o2)


def test_mask_sde(batch, sde: ndp.sde.SDE, network):

    def net(t, yt, x, mask, *, key):
        del key
        return network(t[None], yt[None], x[None], mask[None])[0]

    key = jax.random.PRNGKey(0) 
    t = 0.55 * jnp.ones(())
    o1 = sde.reverse_drift_ode(key, t, batch.ys[0], batch.xs[0], batch.mask[0], net)

    xs = jnp.where(batch.mask[..., None] == 1.0, -666., batch.xs)
    ys = jnp.where(batch.mask[..., None] == 1.0, -666., batch.ys)
    o2 = sde.reverse_drift_ode(key, t, ys[0], xs[0], batch.mask[0], net)

    o1 = jnp.where(batch.mask[0][..., None] == 1., jnp.zeros_like(o1), o1)
    o2 = jnp.where(batch.mask[0][..., None] == 1., jnp.zeros_like(o2), o2)

    np.testing.assert_array_almost_equal(o1, o2)


def test_masked_sde_solve(batch):
    beta = ndp.sde.LinearBetaSchedule()
    limiting_kernel = jaxkern.White()
    hyps = {
        "mean_function": {},
        "kernel": limiting_kernel.init_params(None),
    }
    sde = ndp.sde.SDE(
        limiting_kernel,
        gpjax.mean_functions.Zero(),
        hyps,
        beta,
        is_score_preconditioned=False,
        std_trick=False,
        residual_trick=False,
        exact_score=True,
    )

    m0 = gpjax.mean_functions.Zero()
    k0 = jaxkern.Matern52(active_dims=list(range(1)))
    params0 = {
        "mean_function": {},
        "kernel": k0.init_params(None),
    }
    net = sde.get_exact_score(m0, k0, params0)

    key = jax.random.PRNGKey(0) 

    o1 = ndp.sde.sde_solve(
        sde, net, batch.xs[0], batch.mask[0], key=key
    )

    xs = jnp.where(batch.mask[..., None] == 1.0, -666., batch.xs)
    o2 = ndp.sde.sde_solve(
        sde, net, xs[0], batch.mask[0], key=key
    )

    o1 = jnp.where(batch.mask[0][..., None] == 1., jnp.zeros_like(o1), o1)
    o2 = jnp.where(batch.mask[0][..., None] == 1., jnp.zeros_like(o2), o2)

    np.testing.assert_array_almost_equal(o1, o2)
