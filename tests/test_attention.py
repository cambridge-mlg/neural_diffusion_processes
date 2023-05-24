# from __future__ import annotations

# import haiku as hk
# import jax
# import jax.numpy as jnp
# import jax.random as jr
# import numpy as np
# import pytest
# from check_shapes import check_shape, check_shapes

# from neural_diffusion_processes.models.attention import (
#     BiDimensionalAttentionBlock,
#     MultiOutputBiAttentionModel,
#     MultiOutputAttentionModel,
# )

# from neural_diffusion_processes.utils.tests import (
#     _check_permutation_invariance,
#     _check_permutation_equivariance,
# )


# class Consts:
#     num_timesteps = 50
#     batch_size = 32
#     num_points = 101
#     hidden_dim = 16
#     num_heads = 8
#     num_layers = 3


# @pytest.fixture(name="rng", params=[42])
# def _rng_fixture(request):
#     seed = request.param
#     rng = jax.random.PRNGKey(seed)
#     rng, next_rng = jax.random.split(rng)
#     return next_rng


# @pytest.fixture(name="inputs", params=[1, 3])
# def _inputs_fixture(rng, request):
#     x_dim = request.param
#     y_dim = x_dim
#     x = jr.normal(rng, (Consts.batch_size, Consts.num_points, x_dim))
#     y = jr.normal(rng, (Consts.batch_size, Consts.num_points, y_dim))
#     # t = jr.randint(rng, (Consts.batch_size, 1), minval=0, maxval=Consts.num_timesteps)
#     t = jr.uniform(rng, (Consts.batch_size,), minval=0.0, maxval=1.0)
#     return x, y, t


# @pytest.fixture(name="hidden_inputs", params=[3, 5])
# def _hidden_inputs_fixture(rng, request):
#     x_dim = request.param
#     x_embedded = jr.normal(
#         rng, (Consts.batch_size, Consts.num_points, x_dim, Consts.hidden_dim)
#     )
#     t_embedded = jr.normal(rng, (Consts.batch_size, Consts.hidden_dim))
#     return x_embedded, t_embedded


# @pytest.fixture(name="bidimensional_attention_block")
# def _bidim_attn_block_fixture(rng, hidden_inputs):
#     x_emb, t_emb = hidden_inputs

#     init, apply = hk.without_apply_rng(
#         hk.transform(
#             lambda x, t: BiDimensionalAttentionBlock(
#                 Consts.hidden_dim, Consts.num_heads
#             )(x, t)
#         )
#     )
#     params = init(rng, x_emb, t_emb)
#     return jax.jit(lambda x, t: apply(params, x, t))


# @pytest.fixture(name="denoise_model")
# def _denoise_model_fixture(rng, inputs):
#     x, y, t = inputs

#     def model(x, y, t):
#         score = MultiOutputAttentionModel(
#             Consts.num_layers,
#             Consts.hidden_dim,
#             Consts.num_heads,
#         )
#         return score(x, y, t)

#     init, apply = hk.without_apply_rng(hk.transform(model))
#     params = init(rng, x, y, t)
#     return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))


# @pytest.fixture(name="denoise_model2")
# def _denoise_model_fixture(rng, inputs):
#     x, y, t = inputs

#     def model(x, y, t):
#         score = MultiOutputBiAttentionModel(
#             Consts.num_layers,
#             Consts.hidden_dim,
#             Consts.num_heads,
#         )
#         return score(x, y, t)

#     init, apply = hk.without_apply_rng(hk.transform(model))
#     params = init(rng, x, y, t)
#     return jax.jit(lambda x_, y_, t_: apply(params, x_, y_, t_))


# @check_shapes(
#     "hidden_inputs[0]: [batch_size, seq_len, input_dim, hidden_dim]",
#     "hidden_inputs[1]: [batch_size, hidden_dim]",
# )
# def test_attention_block_equivariance_for_input_dimensionality(
#     rng, hidden_inputs, bidimensional_attention_block
# ):
#     x_emb, t_emb = hidden_inputs
#     f = lambda x: bidimensional_attention_block(x, t_emb)
#     _check_permutation_equivariance(rng, f, 2, 2, x_emb)


# @check_shapes(
#     "hidden_inputs[0]: [batch_size, seq_len, input_dim, hidden_dim]",
#     "hidden_inputs[1]: [batch_size, hidden_dim]",
# )
# def test_attention_block_equivariance_for_data_sequence(
#     rng, hidden_inputs, bidimensional_attention_block
# ):
#     x_emb, t_emb = hidden_inputs
#     _check_permutation_equivariance(
#         rng, lambda x_: bidimensional_attention_block(x_, t_emb), 1, 1, x_emb
#     )


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_equivariance_for_data_sequence(rng, inputs, denoise_model):
#     x, y, t = inputs
#     _check_permutation_equivariance(
#         rng, lambda x_, y_: denoise_model(x_, y_, t), 1, 1, x, y
#     )


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_invariance_for_input_dimensionality(rng, inputs, denoise_model):
#     x, y, t = inputs
#     _check_permutation_invariance(rng, lambda x_: denoise_model(x_, y, t), 2, x)


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_equivariance_for_input_dimensionality(
#     rng, inputs, denoise_model
# ):
#     x, y, t = inputs
#     _check_permutation_equivariance(rng, lambda y_: denoise_model(x, y_, t), 2, 2, y)


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_equivariance_for_data_sequence(rng, inputs, denoise_model2):
#     x, y, t = inputs
#     _check_permutation_equivariance(
#         rng, lambda x_, y_: denoise_model2(x_, y_, t), 1, 1, x, y
#     )


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_invariance_for_input_dimensionality(rng, inputs, denoise_model2):
#     x, y, t = inputs
#     _check_permutation_invariance(rng, lambda x_: denoise_model2(x_, y, t), 2, x)


# @check_shapes(
#     "inputs[0]: [batch_size, seq_len, x_dim]",
#     "inputs[1]: [batch_size, seq_len, y_dim]",
#     "inputs[2]: [batch_size]",
# )
# def test_denoise_model_equivariance_for_input_dimensionality(
#     rng, inputs, denoise_model2
# ):
#     x, y, t = inputs
#     _check_permutation_equivariance(rng, lambda y_: denoise_model2(x, y_, t), 2, 2, y)

import gpjax
import haiku as hk
import jax
import jax.numpy as jnp
import jaxkern
import numpy as np
import pytest

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
