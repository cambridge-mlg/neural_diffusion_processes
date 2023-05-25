from typing import Callable, Optional
from functools import partial
import pytest

import jax

from jax import vmap
import jax.random as jr
import jax.numpy as jnp
import haiku as hk
import numpy as np
from einops import rearrange

from hydra import initialize, compose
from hydra.utils import instantiate, call

from neural_diffusion_processes.models.egnn import (
    EGCL,
    EGNNScore,
    get_edges_batch,
    get_senders_and_receivers_fully_connected,
    get_activation,
)
from neural_diffusion_processes.utils.tests import (
    _check_permutation_equivariance,
    _check_e2_equivariance,
)
import neural_diffusion_processes as ndp


with initialize(config_path="../experiments/steerable_gp/config", version_base="1.3.2"):
    overrides = [
        "data=gpinf",
        "data.n_train=50",
        "data.n_test=50",
        "net=egnn2",
        "net.n_layers=2",
        "net.hidden_dim=16",
        # "+net.residual_y=True",
        # "net.k=0",
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
        # data = call(
        #     cfg.data,
        #     key=next_rng,
        #     num_samples=cfg.data.n_train,
        #     dataset="train",
        # )
        # rng, next_rng = jax.random.split(rng)
        # dataloader = ndp.data.dataloader(
        #     data,
        #     batch_size=cfg.optim.batch_size,
        #     key=next_rng,
        #     n_points=cfg.data.n_points,
        # )
        # batch0 = next(dataloader)
        # x, y = batch0.xs, batch0.ys

        x = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        rng, next_rng = jax.random.split(rng)
        y = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        rng, next_rng = jax.random.split(rng)

        t = jax.random.uniform(next_rng, (y.shape[0],), minval=0, maxval=1)
        print("x, y", x.shape, y.shape)
        return x, y, t

    @pytest.fixture(name="denoise_model")
    def _denoise_model_fixture(rng, inputs):
        x, y, t = inputs

        def model(x, y, t):
            score = EGNNScore(
                hidden_dim=16,
                n_layers=2,
                residual_x=True,
                residual_h=True,
                residual_y=True,
                attention=True,
                normalize=True,
                zero_init=False,
                tanh=True,
                k=30,
                # coords_agg="mean",
            )
            return score(x, y, t)

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    @pytest.fixture(name="egnn_layer")
    def _egnn_layer_fixture(rng, inputs):
        x, y, t = inputs

        def model(x, y, t):
            senders, receivers = get_senders_and_receivers_fully_connected(x.shape[1])
            layer = EGCL(
                name="egcl",
                mlp_units=[16, 16],
                n_invariant_feat_hidden=16,
                activation_fn=get_activation("silu"),
                residual_h=True,
                residual_x=True,
                residual_y=True,
                normalize=True,
                norm_constant=1,
                attention=True,
                tanh=True,
                variance_scaling_init=0.001,
                zero_init=False,
                cross_multiplicty_node_feat=False,
                cross_multiplicity_shifts=False,
                norm_wrt_centre_feat=False,
                x_update="None",
            )
            h = hk.Linear(layer.n_invariant_feat_hidden)(y)
            return jax.vmap(
                lambda x, y, h: layer(x[:, None], y[:, None], h, senders, receivers)[1]
            )(x, y, h).reshape(*x.shape)

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    def test_egnn_layer_permutation_equivariance(rng, inputs, egnn_layer):
        x, y, t = inputs
        _check_permutation_equivariance(
            rng, lambda x_, y_: egnn_layer(x_, y_, t), 1, 1, x, y
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
        assert jnp.abs(out.mean()) > 1e-6

    def test_denoise_model_non_zero(rng, inputs, egnn_layer):
        x, y, t = inputs
        out = egnn_layer(x, y, t)
        assert jnp.abs(out.mean()) > 1e-6

    # def test_egnn_layer_e2_equivariance(rng, inputs, egnn_layer):
    #     _check_e2_equivariance(rng, egnn_layer, *inputs)

    def test_denoise_model_e2_equivariance(rng, inputs, denoise_model):
        _check_e2_equivariance(rng, denoise_model, *inputs)
