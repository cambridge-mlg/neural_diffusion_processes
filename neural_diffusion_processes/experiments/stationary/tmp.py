#%%
from typing import Callable, Tuple, Iterator, Any

from abc import abstractmethod
from functools import partial
import dataclasses

import pickle
import tqdm
import optax
import diffrax as dfx
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from check_shapes import check_shape, check_shapes
from jaxtyping import Array
import jaxkern

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import get_gp_data, dataloader, DataBatch


from jax.config import config


config.update("jax_enable_x64", True)
# %%
LOGS_DIR = f"logs"
DATETIME_STR = datetime.datetime.now().strftime("%b%d_%H%M%S")
TRAIN = False
# %%


@dataclasses.dataclass(frozen=True)
class Config:
    seed = 42
    batch_size = 16
    num_steps = 100_000

    kernel: str

    num_bidim_attention_layers = 2
    hidden_dim = 16
    num_heads = 4


mean_function = lambda x: jnp.zeros_like(x[..., :1])


key = jax.random.PRNGKey(Config.seed)
# k0 = ndp.kernels.SquaredExpontialKernel(lengthscale=.2)
k0 = jaxkern.RBF(active_dims=list(range(1)))

key, dkey, lkey = jax.random.split(key, 3)
data = get_gp_data(dkey, k0, 1_000)

dataset = dataloader(data, Config.batch_size, key=lkey)
batch = next(dataset)
plt.plot(batch.function_inputs[..., 0].T, batch.function_outputs[..., 0].T, ".")
plt.show()


k1 = jaxkern.White()
beta_schedule = ndp.sde.LinearBetaSchedule()

sde = ndp.sde.SDE(limiting_kernel=k1, beta_schedule=beta_schedule)

# %%


@hk.transform
def precond_score_network(x, y, t):
    model = ndp.models.attention.BiDimensionalAttentionModel(
        num_bidim_attention_layers=Config.num_bidim_attention_layers,
        hidden_dim=Config.hidden_dim,
        num_heads=Config.num_heads,
    )
    return model(x, y, t)


key, model_init_key = jax.random.split(key)
params = precond_score_network.init(
    rng=model_init_key,
    x=batch.function_inputs,
    y=batch.function_outputs,
    t=jnp.ones([len(batch.function_inputs)]),
)

loss(sde, precond_score_network.apply, params, batch, key)

# %%


def fit(key, model, params: hk.Params, optimizer: optax.GradientTransformation) -> hk.Params:
    metrics = []
    opt_state = optimizer.init(params)
    loss_fixed_fn = jax.jit(lambda params: loss(sde, model, params, batch, key))

    @jax.jit
    def step(params, opt_state, batch, key):
        loss_value, grads = jax.value_and_grad(partial(loss, sde, model))(params, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    progress_bar = tqdm.tqdm(list(range(1, Config.num_steps + 1)), miniters=1)

    for i in progress_bar:
        batch = next(dataset)
        key, subkey = jax.random.split(key)
        params, opt_state, loss_value = step(params, opt_state, batch, subkey)
        if i % 100 == 0:
            fixed_loss_value = loss_fixed_fn(params)
            progress_bar.set_description(f"loss {fixed_loss_value:.2f}")
            metrics.append({"loss": loss_value, "fixed_loss": fixed_loss_value})

        track_loss(i, loss_value)

    return params, metrics


track_loss = ndp.training_utils.tensorboard_loss_tracker(
    log_directory=f"{LOGS_DIR}/tensorboards/{DATETIME_STR}"
)

learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-4,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10_000,
    end_value=1e-4,
)
optimizer = optax.chain(
    # Clip the gradient by the global norm.
    optax.clip_by_global_norm(1.0),
    # Use the updates from adam.
    optax.scale_by_adam(),
    # Use the learning rate from the scheduler.
    optax.scale_by_schedule(learning_rate_schedule),
    # optax.apply_updates is additive so negate gradients
    optax.scale(-1.0),
)


opt_params, metrics = fit(key, precond_score_network.apply, params, optimizer)
df = pd.DataFrame(metrics)
# %%
path = f"{LOGS_DIR}/checkpoints/{DATETIME_STR}"
ndp.training_utils.save_checkpoint(None, opt_params, path, step_index=Config.num_steps)

# %%
plt.plot(df["loss"])
plt.plot(df["fixed_loss"])
plt.ylim(0, 100)
# %%

if TRAIN:
    eval_params = opt_params
else:
    path = f"{LOGS_DIR}/checkpoints/Feb04_182429/variables/step_100000.pickle"
    with open(path, "rb") as file:
        eval_params = pickle.load(file)
# %%

network = lambda t, yt, x: precond_score_network.apply(
    eval_params, key, x[None], yt[None], t[None]
)[0]


@check_shapes(
    "t: []",
    "yt: [N, 1]",
    "x: [N, 1]",
    "return: [N, 1]",
)
def reverse_drift_ode(t, yt, x):
    # covariance doesn't depend on y0
    _, cov = sde.pt(t, jnp.ones_like(yt) * jnp.nan, x, full_cov=False)
    weight = (1.0 - jnp.exp(-0.5 * sde.beta_schedule.B(t))) ** -1
    score = -weight * cov * network(t, yt, x)
    return sde.drift(t, yt, x) - 0.5 * sde.beta_schedule(t) * score  # [N, 1]


@check_shapes(
    "t: []",
    "yt: [N, 1]",
    "x: [N, 1]",
    "return: [N, 1]",
)
def reverse_drift_sde(t, yt, x):
    weight = (1.0 - jnp.exp(-0.5 * sde.beta_schedule.B(t))) ** -1
    return sde.drift(t, yt, x) + weight * sde.beta_schedule(t) * network(t, yt, x)  # [N, 1]


def reverse_solve(key, x, yT, prob_flow: bool = True):
    t0, t1 = beta_schedule.t0, beta_schedule.t1
    ts = t1 + (t0 - t1) * (jnp.exp(jnp.linspace(t0, t1, 9)) - jnp.exp(t0)) / (
        jnp.exp(t1) - jnp.exp(t0)
    )
    ts = jnp.linspace(t0, t1, 9)[::-1]
    saveat = dfx.SaveAt(ts=ts)
    # reverse time, solve from t1 to t0
    if prob_flow:
        terms = dfx.ODETerm(reverse_drift_ode)
    else:
        drift = dfx.ODETerm(reverse_drift_sde)
        shape = jax.ShapeDtypeStruct(yT.shape, yT.dtype)
        bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=1e-3 / 2.0, shape=shape, key=key)
        terms = dfx.MultiTerm(drift, dfx.ControlTerm(sde.diffusion, bm))

    return dfx.diffeqsolve(
        terms,
        solver=dfx.Euler(),
        t0=t1,
        t1=t0,
        dt0=-1e-3 / 2.0,
        y0=yT,
        saveat=saveat,
        args=x,
        adjoint=dfx.NoAdjoint(),
    )


key, *subkeys = jax.random.split(key, 3)
x_test = jnp.linspace(-1, 1, 100)[:, None]
yT = jax.random.normal(subkeys[0], x_test.shape)
out = reverse_solve(subkeys[1], x_test, yT)
plt.plot(x_test, out.ys[-1])
# %%
plt.plot(x_test, out.ys[..., 0].T)

# %%


def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp
