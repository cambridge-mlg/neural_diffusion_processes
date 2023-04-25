from typing import Callable, Optional
from functools import partial

import jax
from jax import vmap
import jax.random as jr
import jax.numpy as jnp
import haiku as hk

from neural_diffusion_processes.models.egnn import (
    EGNN,
    EFGNN,
    EGNNScore,
    get_edges_batch,
)

# Dummy parameters
batch_size = 8
n_nodes = 6
n_feat = 3
y_dim = 3

rng = jax.random.PRNGKey(0)
# Dummy variables h, x and fully connected edges
rng, next_rng = jax.random.split(rng)
x = jr.normal(next_rng, (batch_size, n_nodes, n_feat))
h = x
y = jr.normal(next_rng, (batch_size, n_nodes, y_dim))
edges, edge_attr = get_edges_batch(n_nodes, batch_size=1)


# Initialize EGNN
@hk.transform_with_state
def model(h, y, edges, edge_attr):
    egnn = EGNN(hidden_dim=32)
    return jax.vmap(lambda h, y: egnn(h, y, edges, edge_attr))(h, y)


rng, next_rng = jax.random.split(rng)
params, state = model.init(rng=next_rng, h=h, y=y, edges=edges, edge_attr=edge_attr)
model = hk.without_apply_rng(model)
# Run EGNN
(h, y), _ = model.apply(params, state, h=h, y=y, edges=edges, edge_attr=edge_attr)

import e3nn_jax as e3nn
from einops import rearrange


def _check_e3_equivariance(
    key: jr.PRNGKey,
    f: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t,
    irreps_input=e3nn.Irreps("1e"),
    irreps_output=e3nn.Irreps("1e"),
):
    """ """
    # print("_check_e3_equivariance")
    y_dim = irreps_input.dim

    key, next_rng = jax.random.split(key)
    R = e3nn.rand_matrix(next_rng, ())
    D_in = irreps_input.D_from_matrix(R)
    D_out = irreps_output.D_from_matrix(R)

    # print("D_in", D_in)
    # print("D_out", D_out)
    x_transformed = x @ D_in.T
    y_transformed = rearrange(
        rearrange(y, "... (n d) -> ... n d", d=y_dim) @ D_out.T, "... n d -> ... (n d)"
    )
    # print("x ", x.shape)
    # print("y ", y.shape)
    out_transformed = f(x, y, t) @ D_out.T
    out = f(x_transformed, y_transformed, t)
    # print("x", x.shape, x[0][:2])
    # print("x_transformed", x_transformed.shape, x_transformed[0][:2])
    # print("y", y.shape, y[0][:2])
    # print("y_transformed", y_transformed.shape, y_transformed[0][:2])
    # print("out_transformed", out_transformed.shape, out_transformed[0][:2])
    # print("out", out.shape, out[0][:2])
    assert jnp.allclose(out_transformed, out, rtol=1e-05, atol=1e-05)


irreps_input = e3nn.Irreps(f"{n_feat}x0e")
# 1/ check invariance of features
irreps_output = irreps_input
f = lambda h, y, t: model.apply(
    params, state, h=h, y=y, edges=edges, edge_attr=edge_attr
)[0][0]
_check_e3_equivariance(next_rng, f, h, y, None, irreps_input, irreps_output)
# 2/ check equivariance of y
irreps_output = e3nn.Irreps("1e")
f = lambda h, y, t: model.apply(
    params, state, h=h, y=y, edges=edges, edge_attr=edge_attr
)[0][1]
_check_e3_equivariance(next_rng, f, h, y, None, irreps_input, irreps_output)


# Initialize EGNN
@hk.transform_with_state
def model(x, y, edges):
    h = jnp.linalg.norm(x, axis=-1, keepdims=True)
    egnn = EFGNN(hidden_dim=32)
    return jax.vmap(lambda h, x, y: egnn(h, x, y, edges)[1])(h, x, y)


ng, next_rng = jax.random.split(rng)
params, state = model.init(rng=next_rng, x=x, y=y, edges=edges)
model = hk.without_apply_rng(model)
# Run EGNN
y, _ = model.apply(params, state, x=x, y=y, edges=edges)
f = lambda x, y, t: model.apply(params, state, x=x, y=y, edges=edges)[0]
_check_e3_equivariance(next_rng, f, x, y, None)

## Score network


# Initialize EGNN
@hk.transform_with_state
def model(x, y, t):
    # egnn = EGNNScore(hidden_dim=32)
    egnn = EGNNScore(
        hidden_dim=32,
        n_layers=5,
        residual=True,
        attention=True,
        normalize=True,
        tanh=True,
        coords_agg="mean",
    )
    # y = rearrange(y, "... (n d) -> ... n d", n=context.shape[-2])
    res = egnn(x, y, t)
    # res = rearrange(res, "... n d -> ... (n d)")
    return res


batch_size = 8
n_nodes = 6
x_dim = 3
y_dim = 3

rng = jax.random.PRNGKey(1)
rng, next_rng = jax.random.split(rng)
x = jr.normal(next_rng, (batch_size, n_nodes, x_dim))
rng, next_rng = jax.random.split(rng)
# y = jr.normal(next_rng, (batch_size, n_nodes * y_dim))
y = jr.normal(next_rng, (batch_size, n_nodes, y_dim))
rng, next_rng = jax.random.split(rng)
t = jax.random.uniform(next_rng, (y.shape[0],), minval=0, maxval=1)
t = t.reshape(-1, 1)

params, state = model.init(rng=next_rng, x=x, y=y, t=t)
model = hk.without_apply_rng(model)
# Run EGNN
out, _ = model.apply(params, state, x=x, y=y, t=t)

# check equivariance of y
f = lambda x, y, t: model.apply(params, state, x=x, y=y, t=t)[0]
f = jax.jit(f)
# y = rearrange(y, "... (n d) -> ... n d", n=x.shape[-2])
_check_e3_equivariance(next_rng, f, x, y, t)
