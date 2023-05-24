import jax
import pytest

jax.config.update("jax_enable_x64", True)
import haiku as hk
import jax.numpy as jnp
from hydra import compose, initialize
from hydra.utils import call, instantiate

from neural_diffusion_processes.utils.tests import (
    _check_e2_equivariance, _check_permutation_equivariance)

with initialize(config_path="../experiments/steerable_gp/config", version_base="1.3.2"):
    overrides = [
        "data=gpinf",
        "data.n_train=50",
        "data.n_test=50",
        "net=e3nn",
        "net.attention=false",
        "net.n_layers=2",
        # "net.residual=true",
        # "net.batch_norm=true",
        "data.n_points=50",
    ]
    cfg = compose(config_name="main", overrides=overrides)

    @pytest.fixture(name="rng", params=[42])
    def _rng_fixuture(request):
        seed = request.param
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        return next_rng

    @pytest.fixture(name="inputs")
    def _inputs_fixuture(rng):
        rng, next_rng = jax.random.split(rng)

        x = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        rng, next_rng = jax.random.split(rng)
        y = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        rng, next_rng = jax.random.split(rng)

        t = jax.random.uniform(next_rng, (y.shape[0],), minval=0, maxval=1)
        return x, y, t

    @pytest.fixture(name="denoise_model")
    def _denoise_model_fixture(rng, inputs):
        x, y, t = inputs

        def model(x, y, t):
            score = instantiate(cfg.net)
            return score(x, y, t)

        stateful_forward = hk.without_apply_rng(hk.transform_with_state(model))
        params, state = stateful_forward.init(rng, x, y, t)
        return jax.jit(
            lambda x_, y_, t_: stateful_forward.apply(params, state, x_, y_, t_)[0]
        )

    def test_denoise_model_permutation_equivariance(rng, inputs, denoise_model):
        x, y, t = inputs
        _check_permutation_equivariance(
            rng, lambda x_, y_: denoise_model(x_, y_, t), 1, 1, x, y
        )

    def test_denoise_model_isnan(rng, inputs, denoise_model):
        x, y, t = inputs
        out = denoise_model(x, y, t)
        assert not jnp.isnan(out).any()

    def test_denoise_model_non_zero(rng, inputs, denoise_model):
        x, y, t = inputs
        out = denoise_model(x, y, t)
        assert jnp.abs(out.mean()) > 1e-4

    def test_denoise_model_e2_equivariance(rng, inputs, denoise_model):
        _check_e2_equivariance(rng, denoise_model, *inputs)
