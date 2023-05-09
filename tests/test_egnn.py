from typing import Callable, Optional
from functools import partial
import pytest

import jax

# jax.config.update("jax_enable_x64", True)1
from jax import vmap
import jax.random as jr
import jax.numpy as jnp
import haiku as hk
import numpy as np
import e3nn_jax as e3nn
from einops import rearrange

from hydra import initialize, compose
from hydra.utils import instantiate, call

from neural_diffusion_processes.models.egnn2 import (
    # EGNN,
    # EFGNN,
    EGCL,
    EGNNScore,
    get_edges_batch,
    get_senders_and_receivers_fully_connected,
    get_activation,
)
from neural_diffusion_processes.utils.tests import (
    _check_permutation_invariance,
    _check_permutation_equivariance,
    _check_e2_equivariance,
    _check_e3_equivariance,
)
import neural_diffusion_processes as ndp

# # Dummy parameters
# batch_size = 8
# n_nodes = 6
# n_feat = 3
# y_dim = 3

# rng = jax.random.PRNGKey(0)
# # Dummy variables h, x and fully connected edges
# rng, next_rng = jax.random.split(rng)
# x = jr.normal(next_rng, (batch_size, n_nodes, n_feat))
# h = x
# y = jr.normal(next_rng, (batch_size, n_nodes, y_dim))
# edges, edge_attr = get_edges_batch(n_nodes, batch_size=1)


# # Initialize EGNN
# @hk.transform_with_state
# def model(h, y, edges, edge_attr):
#     egnn = EGNN(hidden_dim=32)
#     return jax.vmap(lambda h, y: egnn(h, y, edges, edge_attr))(h, y)


# rng, next_rng = jax.random.split(rng)
# params, state = model.init(rng=next_rng, h=h, y=y, edges=edges, edge_attr=edge_attr)
# model = hk.without_apply_rng(model)
# # Run EGNN
# (h, y), _ = model.apply(params, state, h=h, y=y, edges=edges, edge_attr=edge_attr)


# irreps_input = e3nn.Irreps(f"{n_feat}x0e")
# # 1/ check invariance of features
# irreps_output = irreps_input
# f = lambda h, y, t: model.apply(
#     params, state, h=h, y=y, edges=edges, edge_attr=edge_attr
# )[0][0]
# _check_e3_equivariance(next_rng, f, h, y, None, irreps_input, irreps_output)
# # 2/ check equivariance of y
# irreps_output = e3nn.Irreps("1e")
# f = lambda h, y, t: model.apply(
#     params, state, h=h, y=y, edges=edges, edge_attr=edge_attr
# )[0][1]
# _check_e3_equivariance(next_rng, f, h, y, None, irreps_input, irreps_output)


# # Initialize EGNN
# @hk.transform_with_state
# def model(x, y, edges):
#     h = jnp.linalg.norm(x, axis=-1, keepdims=True)
#     egnn = EFGNN(hidden_dim=32)
#     return jax.vmap(lambda h, x, y: egnn(h, x, y, edges)[1])(h, x, y)


# ng, next_rng = jax.random.split(rng)
# params, state = model.init(rng=next_rng, x=x, y=y, edges=edges)
# model = hk.without_apply_rng(model)
# # Run EGNN
# y, _ = model.apply(params, state, x=x, y=y, edges=edges)
# f = lambda x, y, t: model.apply(params, state, x=x, y=y, edges=edges)[0]
# _check_e3_equivariance(next_rng, f, x, y, None)

## Score network


# # Initialize EGNN
# @hk.transform_with_state
# def model(x, y, t):
#     # egnn = EGNNScore(hidden_dim=32)
#     egnn = EGNNScore(
#         hidden_dim=32,
#         n_layers=5,
#         residual_x=True,
#         residual_h=True,
#         residual_y=True,
#         attention=True,
#         normalize=True,
#         # tanh=True,
#         # coords_agg="mean",
#     )
#     # y = rearrange(y, "... (n d) -> ... n d", n=context.shape[-2])
#     res = egnn(x, y, t)
#     # res = rearrange(res, "... n d -> ... (n d)")
#     return res


# batch_size = 8
# n_nodes = 6
# x_dim = 3
# y_dim = 3

# rng = jax.random.PRNGKey(1)
# rng, next_rng = jax.random.split(rng)
# x = jr.normal(next_rng, (batch_size, n_nodes, x_dim))
# rng, next_rng = jax.random.split(rng)
# # y = jr.normal(next_rng, (batch_size, n_nodes * y_dim))
# y = jr.normal(next_rng, (batch_size, n_nodes, y_dim))
# rng, next_rng = jax.random.split(rng)
# t = jax.random.uniform(next_rng, (y.shape[0],), minval=0, maxval=1)

# params, state = model.init(rng=next_rng, x=x, y=y, t=t)
# model = hk.without_apply_rng(model)
# # Run EGNN
# out, _ = model.apply(params, state, x=x, y=y, t=t)

# # check equivariance of y
# f = lambda x, y, t: model.apply(params, state, x=x, y=y, t=t)[0]
# f = jax.jit(f)
# # y = rearrange(y, "... (n d) -> ... n d", n=x.shape[-2])
# _check_e3_equivariance(next_rng, f, x, y, t)


with initialize(config_path="../experiments/steerable_gp/config", version_base="1.3.2"):
    overrides = [
        "data=gpinf",
        "data.num_samples_train=50",
        "data.num_samples_test=50",
        # "net=egnn",
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
        data = call(
            cfg.data,
            key=next_rng,
            num_samples=cfg.data.num_samples_train,
            dataset="train",
        )
        rng, next_rng = jax.random.split(rng)
        dataloader = ndp.data.dataloader(
            data,
            batch_size=cfg.optim.batch_size,
            key=next_rng,
            n_points=cfg.data.n_points,
        )
        batch0 = next(dataloader)
        x, y = batch0.xs, batch0.ys

        # x = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        # rng, next_rng = jax.random.split(rng)
        # y = jax.random.normal(next_rng, (cfg.optim.batch_size, cfg.data.n_points, 2))
        # rng, next_rng = jax.random.split(rng)

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

    # def test_egnn_layer_e3_equivariance(rng, inputs, egnn_layer):
    #     _check_e3_equivariance(rng, egnn_layer, *inputs)

    def test_denoise_model_e2_equivariance(rng, inputs, denoise_model):
        _check_e2_equivariance(rng, denoise_model, *inputs)

    # def test_denoise_model_e3_equivariance(rng, inputs, denoise_model):
    #     _check_e3_equivariance(rng, denoise_model, *inputs)
