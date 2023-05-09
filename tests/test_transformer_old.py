from __future__ import annotations

import math
from typing import Any, Callable, Tuple
import pytest
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import e3nn_jax as e3nn
from einops import rearrange

from hydra import initialize, compose
from hydra.utils import instantiate

from neural_diffusion_processes.utils.tests import (
    _check_permutation_invariance,
    _check_permutation_equivariance,
    _check_e2_equivariance,
    _check_e3_equivariance,
)
from neural_diffusion_processes.models.transformer import (
    Transformer,
    nearest_neighbors_jax,
)
from e3nn_jax.experimental.transformer import _tp_mlp_uvu, FullyConnectedTensorProduct

from jaxtyping import f, jaxtyped
from typeguard import typechecked as typechecker


with initialize(config_path="../config"):
    overrides = [
        "experiment=vecgp",
        "net=transformer",
        "name=test_transformer",
        "batch_size=4",
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
        dataset = instantiate(cfg.dataset, rng=next_rng)
        y, x = next(dataset)
        y = rearrange(y, "... (n d) -> ... n d", n=x.shape[-2])
        rng, step_rng = jax.random.split(rng)
        t = jax.random.uniform(step_rng, (y.shape[0],), minval=0, maxval=1)
        return x, y, t

    @pytest.fixture(name="linear_layer")
    def _linear_layer_fixture(rng, inputs):
        x, y, t = inputs

        def score(x, y, t):
            irreps = e3nn.Irreps("1x1e")
            y = e3nn.IrrepsArray("1x1e", y)
            y = jax.vmap(e3nn.Linear(irreps))(y)
            return y.array

        @jaxtyped
        @typechecker
        def model(x: f["b n d"], y: f["b n d"], t: f["b"]) -> f["b n d"]:
            res = jax.vmap(score)(x, y, t)
            return res

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    @pytest.fixture(name="mlp_model")
    def _mlp_model_fixture(rng, inputs):
        x, y, t = inputs
        config = cfg.net

        mul0 = config["mul0"]
        mul1 = config["mul1"]
        mul2 = config["mul2"]
        irreps_features = e3nn.Irreps(
            f"{mul0}x0e + {mul0}x0o + {mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o"
        ).simplify()
        irreps = e3nn.Irreps("1x1e")

        def act(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            x = e3nn.scalar_activation(
                x, [jax.nn.gelu, jnp.tanh] + [None] * (len(x.irreps) - 2)
            )
            return x
            tp = e3nn.TensorSquare(irreps_features, init=hk.initializers.Constant(0.0))
            y = jax.vmap(tp)(x)
            return x + y

        def score(x, y, t):
            y = e3nn.IrrepsArray("1x1e", y)
            y = jax.vmap(e3nn.Linear(irreps_features))(y)
            y = act(y)
            y = jax.vmap(e3nn.Linear(irreps_features))(y)
            y = act(y)
            y = jax.vmap(e3nn.Linear(irreps))(y)
            return y.array

        @jaxtyped
        @typechecker
        def model(x: f["b n d"], y: f["b n d"], t: f["b"]) -> f["b n d"]:
            res = jax.vmap(score)(x, y, t)
            return res

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    @pytest.fixture(name="sh_layer")
    def _sh_layer_fixture(rng, inputs):
        x, y, t = inputs
        config = cfg.net

        def score(x, y, t):
            pos = x
            node_attr = y

            idx = nearest_neighbors_jax(pos, k=config["k"])
            edge_src = idx.reshape(-1)
            edge_dst = jnp.arange(0, pos.shape[-2], 1)
            edge_dst = jnp.repeat(edge_dst, config["k"], 0).reshape(-1)

            edge_attr = e3nn.spherical_harmonics(
                e3nn.Irreps.spherical_harmonics(config["shlmax"]),
                # e3nn.Irreps.spherical_harmonics(e3nn.Irreps("1x1e")),
                pos[edge_dst] - pos[edge_src],
                normalize=True,
                normalization="component",
            )
            return edge_attr.array

        @jaxtyped
        @typechecker
        def model(x: f["b n d"], y: f["b n d"], t: f["b"]):
            res = jax.vmap(score)(x, y, t)
            return res

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    @pytest.fixture(name="transformer_layer")
    def _transformer_layer_fixture(rng, inputs):
        x, y, t = inputs

        config = cfg.net
        config["num_heads"] = 1

        kw = dict(
            list_neurons=[config["radial_num_neurons"] * config["radial_num_layers"]],
            act=jax.nn.gelu,
            num_heads=config["num_heads"],
        )
        mul0 = config["mul0"]
        mul1 = config["mul1"]
        mul2 = config["mul2"]
        irreps_features = e3nn.Irreps(
            f"{mul0}x0e + {mul0}x0o + {mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o"
        ).simplify()
        irreps_out = e3nn.Irreps("1x1e")

        def act(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            x = e3nn.scalar_activation(
                x, [jax.nn.gelu, jnp.tanh] + [None] * (len(x.irreps) - 2)
            )
            # return x
            # tp = e3nn.TensorSquare(irreps_features, init=hk.initializers.Constant(0.0))
            tp = e3nn.TensorSquare(irreps_features)
            y = jax.vmap(tp)(x)
            return x + y

        def score(x, y, t):
            print("score")
            pos = x
            node_attr = y

            t = jnp.repeat(t.reshape(-1)[None, :], y.shape[-2], -2)
            t = e3nn.IrrepsArray(f"{t.shape[-1]}x0e", t)

            idx = nearest_neighbors_jax(pos, k=config["k"])
            edge_src = idx.reshape(-1)
            edge_dst = jnp.arange(0, pos.shape[-2], 1)
            edge_dst = jnp.repeat(edge_dst, config["k"], 0).reshape(-1)

            edge_attr = e3nn.spherical_harmonics(
                e3nn.Irreps.spherical_harmonics(config["shlmax"]),
                pos[edge_dst] - pos[edge_src],
                normalize=True,
                normalization="component",
            )

            edge_length = jnp.linalg.norm(pos[edge_dst] - pos[edge_src], axis=-1)
            edge_scalars = (
                e3nn.soft_one_hot_linspace(
                    edge_length,
                    start=0.0,
                    end=config["maximum_radius"],
                    number=config["num_basis"],
                    basis=config["radial_basis"],
                    cutoff=False,
                )
                * config["num_basis"] ** 0.5
                * 0.95
            )
            # edge_scalars = jnp.concatenate(
            #     [edge_scalars, node_attr[edge_src], node_attr[edge_dst]],
            #     axis=-1,
            # )
            edge_weight_cutoff = 1.4 * e3nn.sus(
                10 * (1 - edge_length / config["maximum_radius"])
            )
            edge_scalars *= edge_weight_cutoff[:, None]

            # irreps = e3nn.Irreps(f'{config["mul1"]}x1e')
            irreps = irreps_features
            node_attr = e3nn.IrrepsArray("1x1e", node_attr)
            node_attr = e3nn.IrrepsArray.cat([node_attr, t])
            x = jax.vmap(e3nn.Linear(irreps))(node_attr)
            # x = e3nn.IrrepsArray("1x1e", node_attr)

            edge_src_feat = jax.tree_util.tree_map(lambda x: x[edge_src], x)
            edge_dst_feat = jax.tree_util.tree_map(lambda x: x[edge_dst], x)

            tp_kw = dict(list_neurons=kw["list_neurons"], act=kw["act"])
            edge_k = jax.vmap(
                lambda w, x, y: _tp_mlp_uvu(w, x, y, edge_dst_feat.irreps, **tp_kw)
            )(
                edge_scalars, edge_src_feat, edge_attr
            )  # IrrepData[edge, irreps]
            edge_v = jax.vmap(
                lambda w, x, y: _tp_mlp_uvu(w, x, y, irreps_out, **tp_kw)
            )(
                edge_scalars, edge_src_feat, edge_attr
            )  # IrrepData[edge, irreps]
            edge_logit = jax.vmap(FullyConnectedTensorProduct(f"1x0e"))(
                edge_dst_feat, edge_k
            )  # array[edge, head]
            print("edge_k", edge_k.irreps)
            print("edge_v", edge_v.irreps)
            print("edge_dst_feat", edge_dst_feat.irreps)
            print("edge_logit", edge_logit.irreps)
            return edge_k.array

            # x = Transformer(irreps_features, **kw)(
            #     edge_src, edge_dst, edge_scalars, edge_weight_cutoff, edge_attr, x
            # )
            # x = act(x)
            # x = Transformer(irreps_out, **kw)(
            #     edge_src, edge_dst, edge_scalars, edge_weight_cutoff, edge_attr, x
            # )
            # return x.array

        @jaxtyped
        @typechecker
        def model(x: f["b n d"], y: f["b n d"], t: f["b"]):  # -> f["b n d"]:
            t = instantiate(cfg.t_embed)(t.reshape(-1, 1))
            res = jax.vmap(score)(x, y, t)
            return res

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    @pytest.fixture(name="denoise_model")
    def _denoise_model_fixture(rng, inputs):
        x, y, t = inputs

        def model(x: f["b n d"], y: f["b n d"], t: f["b"]) -> f["b n d"]:
            score = instantiate(cfg.generator, cfg.net)
            t = instantiate(cfg.t_embed)(t.reshape(-1, 1))
            # y = rearrange(y, "... (n d) -> ... n d", d=x.shape[-1])
            y = instantiate(cfg.y_embed)(y)
            x = instantiate(cfg.c_embed)(x)
            res = score(x, y, t)
            return res

        init, apply = hk.without_apply_rng(hk.transform(model))
        params = init(rng, x, y, t)
        return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))

    def test_denoise_model_permutation_equivariance(rng, inputs, denoise_model):
        x, y, t = inputs
        _check_permutation_equivariance(
            rng, lambda x_, y_: denoise_model(x_, y_, t), 1, 1, x, y
        )

    def test_denoise_model_e2_equivariance(rng, inputs, denoise_model):
        _check_e2_equivariance(rng, denoise_model, *inputs)

    def test_denoise_model_e3_equivariance(rng, inputs, denoise_model):
        _check_e3_equivariance(rng, denoise_model, *inputs)

    def test_transfomer_layer_e3_equivariance(rng, inputs, transformer_layer):
        # edge_k 16x1e
        # edge_v 16x1e
        # edge_dst_feat 8x1e

        # edge_k 40x0e+40x0o+56x1o+56x1e
        # edge_v 56x1e
        # edge_dst_feat 32x0e+32x0o+8x1e+8x1o

        irreps_output = e3nn.Irreps("40x0e+40x0o+56x1o+56x1e")
        # irreps_output = e3nn.Irreps("1x0e")  # edge_logit
        print("irreps_output", irreps_output)
        _check_e3_equivariance(
            rng, transformer_layer, *inputs, irreps_output=irreps_output
        )

    def test_sh_layer_e3_equivariance(rng, inputs, sh_layer):
        irreps_output = e3nn.Irreps("1x0e+1x1o+1x2e")
        _check_e3_equivariance(rng, sh_layer, *inputs, irreps_output=irreps_output)

    def test_linear_layer_e3_equivariance(rng, inputs, linear_layer):
        _check_e3_equivariance(rng, linear_layer, *inputs)

    def test_mlp_model_e3_equivariance(rng, inputs, mlp_model):
        _check_e3_equivariance(rng, mlp_model, *inputs)
