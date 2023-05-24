import math
from typing import Tuple, Mapping, Iterator, List
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import socket
import logging
from collections import defaultdict
from functools import partial
import tqdm
import yaml
import dataclasses


import haiku as hk
import jax
from jax import jit, vmap
import jax.numpy as jnp
import optax
import numpy as np
from jaxtyping import Array
from simple_pytree import Pytree

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from hydra import initialize, compose
from jax.config import config as jax_config
from check_shapes import check_shapes, check_shape

import neural_diffusion_processes as ndp
from neural_diffusion_processes.ml_tools.state import TrainingState
from neural_diffusion_processes.kernels import SumKernel, WhiteVec

initialize(config_path="../experiments/steerable_gp/config", version_base="1.3.2")

cfg = compose(
    # config_path="../config",
    config_name="main",
    overrides=[
        "lr_schedule.warmup_steps=50",
        "optim.num_steps=300",
        "optim.batch_size=5",
        "optim.learning_rate=1e-3",
        # "optim.learning_rate=5e-4",
        "data=gpinf",
        "data.n_samples_ain=1",
        # "net=mlp",
        # "net=egnn",
        "net=egnn2",
        "net.n_layers=5",
        # "net.hidden_dim=16",
        "net.hidden_dim=64",
        "net.num_heads=2",
        # "+net.residual_y=True",
        # "net.k=648",
        # "net.k=50",
        "net.k=100",
        # "net.k=0",
    ],
    # overrides=[
    #     "lr_schedule.warmup_steps=50",
    #     "optim.num_steps=300",
    #     # "optim.batch_size=5",
    #     "optim.learning_rate=1e-3",
    #     # "optim.learning_rate=2e-3",
    #     "data=gpinf",
    #     "data.n_train=1",
    #     # "net=mlp",
    #     "net=mattn",
    #     # "net.n_layers=5",
    #     # "net.hidden_dim=16",
    # ],
)


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


key = jax.random.PRNGKey(cfg.seed)
key_iter = _get_key_iter(key)


####### init relevant diffusion classes
limiting_kernel = instantiate(cfg.kernel.cls)
kernel_params = limiting_kernel.init_params(key)
kernel_params.update(
    OmegaConf.to_container(cfg.kernel.params, resolve=True)
)  # NOTE: breaks RFF?
if not isinstance(limiting_kernel, WhiteVec):
    limiting_kernel = SumKernel([limiting_kernel, WhiteVec(2)])
    kernel_params = [kernel_params, {"variance": cfg.kernel.noise}]
limiting_mean_fn = instantiate(cfg.sde.limiting_mean_fn)
limiting_params = {
    "kernel": kernel_params,
    "mean_function": limiting_mean_fn.init_params(key),
}
sde = instantiate(
    cfg.sde,
    limiting_kernel=limiting_kernel,
    limiting_params=limiting_params,
    beta_schedule=cfg.beta_schedule,
)


@dataclasses.dataclass
class DataBatch(Pytree):
    x: Array
    y: Array
    yt: Array
    t: Array

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def num_points(self) -> int:
        return self.xs.shape[1]


@check_shapes(
    "data[0]: [len_data, num_points, input_dim]",
    "data[1]: [len_data, num_points, output_dim]",
)
def get_dataloader(
    data: Tuple[Array, Array, Array],
    batch_size: int,
    *,
    key,
    run_forever=True,
    n_points=-1,
) -> Iterator[DataBatch]:
    """Yields minibatches of size `batch_size` from the data."""
    x, y, yt, t = data
    n_points = n_points if n_points > 0 else x.shape[1]
    # x = x.astype(jnp.float32)
    # y = y.astype(jnp.float32)
    dataset_size = len(x)
    indices_batch = jnp.arange(dataset_size)
    indices_points = jnp.arange(x.shape[1])
    while True:
        perm = jax.random.permutation(key, indices_batch)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            # (key,) = jax.random.split(key, 1)
            # points_perm = jax.random.permutation(key, indices_points)[:n_points]
            # yield DataBatch(
            #     x=jnp.take(x[batch_perm], axis=1, indices=points_perm),
            #     y=jnp.take(y[batch_perm], axis=1, indices=points_perm),
            #     yt=jnp.take(yt[batch_perm], axis=1, indices=points_perm),
            # )
            yield DataBatch(
                x=x[batch_perm], y=y[batch_perm], yt=yt[batch_perm], t=t[batch_perm]
            )
            start = end
            end = start + batch_size

        if not run_forever:
            break


data = call(
    cfg.data,
    key=jax.random.PRNGKey(cfg.data.seed),
    # num_samples=40,
    num_samples=1000,
    dataset="train",
)


x, y = data
print("x", x.shape)
# t = jnp.array(0.3)
t = jax.random.uniform(key, (x.shape[0],), minval=5e-4, maxval=0.3)
# t = 0.3 * jnp.ones_like(t)
# K = 10
# subkeys = jax.random.split(key, num=K)
yt = jax.vmap(lambda t, x, y: sde.sample_marginal(key, t, x, y))(t, x, y)
data = x, y, yt, t

dataloader = get_dataloader(
    data, batch_size=cfg.optim.batch_size, key=next(key_iter), run_forever=True
)


@hk.without_apply_rng
@hk.transform
def network(t, y, x):
    print(y.shape, x.shape)
    model = instantiate(cfg.net)
    # model = ndp.models.MLPScore(hidden_dim=128)
    print("model", model)
    return model(t=t, x=x, y=y)


@jit
def net(params, t, y, x, *, key):
    return network.apply(params, t=t[None], y=y[None], x=x[None])[0]


def loss_fn(params, batch, key):
    network_ = partial(net, params)
    out = jax.vmap(lambda t, x, y: network_(t=t, x=x, y=y, key=key))(
        batch.t, batch.x, batch.yt
    )
    return jnp.square(out - batch.y).sum(axis=-1).mean()
    # out = jax.vmap(lambda x, y: network_(x=x, y=y, key=key))(batch.x, batch.y)
    # return jnp.square(out - batch.yt).sum(axis=-1).mean()


learning_rate_schedule = instantiate(cfg.lr_schedule)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)


@jit
def init(t, y, x, key) -> TrainingState:
    key, init_rng = jax.random.split(key)
    initial_params = network.init(init_rng, t=t, y=y, x=x)
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )


@jit
def update_step(
    state: TrainingState, batch: DataBatch
) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    new_state = TrainingState(
        params=new_params,
        params_ema=new_params,
        opt_state=new_opt_state,
        key=new_key,
        step=state.step + 1,
    )
    metrics = {"loss": loss_value, "step": state.step}
    return new_state, metrics


state = init(t[0][None], y[0][None], x[0][None], jax.random.PRNGKey(cfg.seed))
nb_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
print(f"Number of parameters: {nb_params}")

miniters = 1
progress_bar = tqdm.tqdm(
    list(range(1, cfg.optim.num_steps + 1)),
)
for step, batch, key in zip(progress_bar, dataloader, key_iter):
    state, metrics = update_step(state, batch)
    # state, metrics = update_step(state, DataBatch(x, y, yt))
    metrics["lr"] = learning_rate_schedule(step)

    if step == 1 or step % miniters == 0:
        progress_bar.set_description(f"loss {metrics['loss']:.2f}", refresh=False)
