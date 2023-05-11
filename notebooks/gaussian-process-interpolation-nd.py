# %%
%load_ext autoreload
%autoreload 2

# %%
from __future__ import annotations
from functools import partial
import os
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np
from einops import rearrange
import gpjax

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.sde import (
    SumKernel,
)
from neural_diffusion_processes.utils import flatten, unflatten
from neural_diffusion_processes.data import radial_grid_2d
from neural_diffusion_processes.utils.vis import (
    plot_scalar_field,
    plot_vector_field,
    plot_covariances,
)
from neural_diffusion_processes.kernels import sample_prior_gp, prior_gp, posterior_gp
from neural_diffusion_processes.sde import log_prob
from neural_diffusion_processes.config import get_config

JITTER = get_config().jitter

# %%
from jax.config import config

config.update("jax_enable_x64", True)

norm = matplotlib.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)

from jaxtyping import Array

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

output_dim = 2
# beta_schedule = ndp.sde.LinearBetaSchedule(t0=1e-5, beta0=1e-4, beta1=15.0)
beta_schedule = ndp.sde.LinearBetaSchedule(t0=1e-5, beta0=1e-4, beta1=10.0)
x = radial_grid_2d(10, 30)

# k0 = ndp.kernels.RBFVec(output_dim)
# k0 = ndp.kernels.RBFCurlFree()
k0 = ndp.kernels.RBFDivFree()
# k0_params = k0.init_params(None)
# k0_variance = 10
k0_variance = 1.0
lengthscale = 2.23606797749979
# lengthscale = 1
k0_params = {"variance": k0_variance, "lengthscale": lengthscale}
k0 = SumKernel([k0, ndp.kernels.WhiteVec(output_dim)])
k0_params = [k0_params, {"variance": 0.02}]

k1 = ndp.kernels.RBFVec(output_dim)
# k1 = ndp.kernels.WhiteVec(output_dim)
# k1_params = {"variance": k0_variance, "lengthscale": 1.}
k1_params = {"variance": 1.0, "lengthscale": 1.0}
k1 = SumKernel([k1, ndp.kernels.WhiteVec(output_dim)])
k1_params = [k1_params, {"variance": 0.1}]
# k1_params = [k1_params, {"variance": 0.02}]

mean_function = gpjax.Zero(output_dim)

limiting_params = {
    "kernel": k1_params,
    "mean_function": mean_function.init_params(key),
}
data_params = {
    "kernel": k0_params,
    "mean_function": mean_function.init_params(key),
}

# kxx = k0.gram(k0_params, x).to_dense()
# kxx = rearrange(kxx, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=output_dim, p2=output_dim)
# kxx = rearrange(kxx, 'n1 n2 p1 p2 -> (p1 n1) (p2 n2)')
# plt.matshow(kxx)

# kxx = k1.gram(limiting_params["kernel"], x).to_dense()
# kxx = rearrange(kxx, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=output_dim, p2=output_dim)
# kxx = rearrange(kxx, 'n1 n2 p1 p2 -> (p1 n1) (p2 n2)')
# plt.matshow(kxx)

sde = ndp.sde.SDE(
    k1,
    mean_function,
    limiting_params,
    beta_schedule,
    # is_score_preconditioned=False,
    is_score_preconditioned=True,
    std_trick=False,
    residual_trick=False,
)
network = sde.get_exact_score(mean_function, k0, data_params)

# import jmp
# import haiku as hk
# from neural_diffusion_processes.models.attention import MultiOutputAttentionModel
# policy = jmp.get_policy('params=float32,compute=float32,output=float32')
# @hk.without_apply_rng
# @hk.transform
# def network_def(t, y, x):
#     t, y, x = policy.cast_to_compute((t, y, x))
#     model = MultiOutputAttentionModel(n_layers=3, hidden_dim=64, num_heads=4, sparse=False)
#     return model(x, y, t)

# dist = prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.)(x)
# y0 = unflatten(dist.sample(seed=subkey, sample_shape=(8)), 2)
# key, subkey = jax.random.split(key)
# params = policy.cast_to_param((network_def.init(subkey, t= 1. * jnp.zeros((y0.shape[0])), y=y0, x=jnp.repeat(x[None, ...], 8, 0))))
# @jax.jit
# def network(t, yt, x, *, key):
#     #NOTE: Network awkwardly requires a batch dimension for the inputs
#     # return network_def.apply(params, t[None], yt[None], x[None])[0]
#     return network_def.apply(params, t[None], yt[None], x[None])[0]


plot_vf = partial(plot_vector_field, scale=50 * math.sqrt(k0_variance), width=0.005)
plot_cov = partial(plot_covariances, scale=0.3 / math.sqrt(k0_variance), zorder=-1)


def get_timesteps(t0, t1, num_ts):
    ts = jnp.exp(jnp.linspace(t0, t1, num_ts))
    ts = t0 + (t1 - t0) * (ts - ts[0]) / (ts[-1] - ts[0])
    return ts


def plot(ys, ts=None):
    plot_num_timesteps = ys.shape[1]
    fig, axes = plt.subplots(
        plot_num_timesteps,
        2,
        sharex=True,
        sharey=True,
        figsize=(8 * 2, 8 * plot_num_timesteps),
    )
    # axes = axes if isinstance(type(axes[0]), np.ndarray) else axes[None, :]
    if plot_num_timesteps == 1:
        axes = axes[None, :]
    fig.subplots_adjust(wspace=0, hspace=0.0)

    for j in range(plot_num_timesteps):
        for i in range(2):
            if x.shape[-1] == 1:
                for o in range(output_dim):
                    axes[j, i].plot(x, ys[i, j, :, o], "-", ms=2)
            elif x.shape[-1] == 2:
                plot_vf(x, ys[i, j], axes[j, i])
            if ts is not None:
                axes[j, 0].set_ylabel(f"t = {ts[j]:.2f}")
    plt.tight_layout()
    plt.show()


# %%
# Forward process

seed = 1
key = jax.random.PRNGKey(seed)

# Solves forward SDE for multiple initia states using vmap.
num_samples = 2
key, subkey = jax.random.split(key)
y0s = sample_prior_gp(
    key,
    mean_function,
    k0,
    {"kernel": k0_params, "mean_function": {}},
    x,
    num_samples=num_samples,
    obs_noise=0.0,
)
print(y0s.shape)
subkeys = jax.random.split(key, num=num_samples)
ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1, num_ts=5)

solve = jax.jit(
    lambda y, key: ndp.sde.sde_solve(
        sde,
        network,
        x,
        y=y,
        key=key,
        ts=ts,
        prob_flow=True,
        atol=None,
        rtol=None,
        num_steps=100,
        forward=True,
    )
)
ys = jax.vmap(solve)(y0s, subkeys)
# solve = lambda y, key: ndp.sde.sde_solve(sde, network, x, y=y, key=key, ts=ts, prob_flow=False, atol=None, rtol=None, num_steps=100, forward=True)
# ys = solve(y0s[0], subkeys[0])

plot(ys, ts)

# %%
# Backward

# ts = get_timesteps(sde.beta_schedule.t1, 1e-3, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0 + 1e-8, num_ts=5)

reverse_solve = lambda key, y: ndp.sde.sde_solve(
    sde,
    network,
    x,
    key=key,
    y=y,
    ts=ts,
    prob_flow=True,
    # solver=dfx.Heun(), rtol=1e-3, atol=1e-3, num_steps=100)
    solver=dfx.Euler(),
    rtol=None,
    atol=None,
    num_steps=100,
)

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 1 + num_samples)
# rev_out = reverse_solve(key, ys[0, -1])
yT = sample_prior_gp(
    key,
    mean_function,
    k1,
    {"kernel": k1_params, "mean_function": {}},
    x,
    num_samples=num_samples,
)
# yT = ys[:, -1]

rev_out = jax.vmap(reverse_solve)(np.stack(subkeys), yT)
# print(rev_out.shape)
key, *subkeys = jax.random.split(key, 1 + num_samples)
rev_out2 = jax.vmap(reverse_solve)(np.stack(subkeys), yT)

plot(rev_out, ts)
# plot(rev_out2, ts)

# %%
# Likelihood evaluation of data proces

key = jax.random.PRNGKey(1)
num_samples = 2
subkeys = jax.random.split(key, num=num_samples)

# dist = prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.02)(x)
dist_T = prior_gp(
    mean_function, k1, {"kernel": k1_params, "mean_function": {}}, obs_noise=0.0
)(x)
yT = dist_T.sample(seed=subkey, sample_shape=(num_samples))

# ts = get_timesteps(sde.beta_schedule.t0+1e-6, sde.beta_schedule.t1, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0 + 1e-6, num_ts=5)
# ts = None
num_steps = 100
# solver = dfx.Tsit5()
solver = dfx.Euler()
# solver = dfx.Dopri5()
# rtol: float = 1e-3
# atol: float = 1e-4
rtol = atol = None
hutchinson_type = "None"

log_prior, delta_logp, nfe, ys = jax.vmap(
    lambda y, key: log_prob(
        sde,
        network,
        x,
        y,
        key=key,
        num_steps=num_steps,
        solver=solver,
        rtol=rtol,
        atol=atol,
        hutchinson_type=hutchinson_type,
        ts=ts,
        forward=False,
        return_all=True,
    )
)(unflatten(yT, output_dim), subkeys)

plot(ys)

# %%
# Likelihood evaluation of model

num_samples = 2
key, subkey = jax.random.split(key)
dist = prior_gp(
    mean_function, k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.0
)(x)
y0s = dist.sample(seed=subkey, sample_shape=(num_samples))
n_test = y0s.shape[-2]
# print("mean var", (jnp.std(y0s, 0) ** 2).mean())
true_logp = jax.vmap(dist.log_prob)(y0s).squeeze()
print("true_logp", true_logp.shape, true_logp / n_test)

# log_prior = jax.vmap(dist_T.log_prob)(y0s).squeeze()
# print("log_prior", log_prior.shape, log_prior)

y0s = unflatten(y0s, output_dim)

num_steps = 100
solver = dfx.Tsit5()
# solver = dfx.Euler()
rtol: float = 1e-6
atol: float = 1e-6
# rtol: float = 1e-3
# atol: float = 1e-3
# rtol = atol = None
print(solver, rtol, atol)

key = jax.random.PRNGKey(4)
subkeys = jax.random.split(key, num=num_samples)
# hutchinson_type="Gaussian"
# hutchinson_type="Rademacher"
hutchinson_samples = 16
hutchinson_type = "None"

ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1 - 1e-3, num_ts=5)
# ts = None

log_prior, delta_logp, nfe, ys = jax.vmap(
    jax.jit(
        lambda y, key: log_prob(
            sde,
            network,
            x,
            y,
            key=key,
            num_steps=num_steps,
            solver=solver,
            rtol=rtol,
            atol=atol,
            hutchinson_type=hutchinson_type,
            hutchinson_samples=hutchinson_samples,
            ts=ts,
            forward=True,
            return_all=True,
        )
    )
)(y0s, subkeys)
plot(ys, ts)

log_prior = log_prior[:, -1]
delta_logp = delta_logp[:, -1]

model_logp = log_prior + delta_logp
print("log_prior ode", log_prior.shape, log_prior / n_test)
print("delta_logp", delta_logp.shape, delta_logp / n_test)
print("model_logp", model_logp.shape, model_logp / n_test)
print("true_logp", true_logp.shape, true_logp / n_test)
print("nfe", nfe)
print("norm diff", jnp.linalg.norm(model_logp - true_logp) / num_samples)

# %%
# Solve for conditional samples

key = jax.random.PRNGKey(1)

num_context = 25
x = radial_grid_2d(10, 30)
num_points = x.shape[-2]
indices = jnp.arange(num_points)
# num_context = jax.random.randint(key1, (), minval=min_context, maxval=max_context)
num_target = num_points - num_context
perm = jax.random.permutation(key, indices)
x_test = jnp.take(x, axis=-2, indices=perm[:num_target])
x_known = jnp.take(x, axis=-2, indices=perm[-num_context:])


idx = jax.random.permutation(key, jnp.arange(len(x_test)))
x_known = x_test[idx[:num_context]] + 1e-5

num_samples = 100

key, subkey = jax.random.split(key)
y_known = unflatten(
    prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(
        x_known
    ).sample(seed=key, sample_shape=()),
    output_dim,
)

key, subkey = jax.random.split(key)
data_posterior_gp = posterior_gp(
    mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x_known, y_known
)(x_test)
y_test = unflatten(
    data_posterior_gp.sample(seed=subkey, sample_shape=(num_samples)), output_dim
)

# # Plotting empirical mean and cov vs true values (both from GP)
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8 * 2, 8 * 2))
# fig.subplots_adjust(wspace=0, hspace=0.0)

# mean = jnp.mean(y_test, 0)
# cov = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(y_test)

# y_dim = 2
# true_mean = unflatten(data_posterior_gp.mean(), y_dim)
# ktt = data_posterior_gp.covariance()
# ktt = rearrange(ktt, "(n1 p1) (n2 p2) -> n1 n2 p1 p2", p1=y_dim, p2=y_dim)
# true_cov = ktt[jnp.diag_indices(ktt.shape[0])]

# for ax, (mean, cov) in zip(axes.T, [(mean, cov), (true_mean, true_cov)]):
#     plot_vf(x_test, mean, ax[0])
#     plot_vf(x_known, y_known, ax[0], color="r")

#     plot_vf(x_test, mean, ax[1])
#     plot_vf(x_known, y_known, ax[1], color="r", zorder=2)
#     plot_cov(x_test, cov, ax=ax[1], zorder=1)

# %%
num_steps = 50
# num_inner_steps = 50
num_inner_steps = 50
print(f"num_steps={num_steps} and num_inner_steps={num_inner_steps}")
import time

start = time.time()
# conditional_sample = jax.jit(lambda  x, y, x_eval, key: ndp.sde.conditional_sample2(sde, None, x, y, x_eval, key=key, num_steps=num_steps, num_inner_steps=num_inner_steps, langevin_kernel=True, alpha=3.))
conditional_sample = jax.jit(
    lambda x, y, x_eval, key: ndp.sde.conditional_sample_independant_context_noise(
        sde,
        network,
        x,
        y,
        x_eval,
        key=key,
        num_steps=num_steps,
        num_inner_steps=num_inner_steps,
        langevin_kernel=False,
        # langevin_kernel=True,
        tau=0.5,
        psi=1.5,
        lambda0=1.5,
        # tau=1.0,
        # psi=1.0,
        # lambda0=1.0,
        # prob_flow=True,
        prob_flow=False,
    )
)

samples = jax.jit(
    jax.vmap(lambda key: conditional_sample(x_known, y_known, x_test, key=key))
)(jax.random.split(key, num_samples))
end = time.time()
print(f"time={end - start:.2f}")
mse_mean_pred = jnp.sum((samples.mean(0) - y_test.mean(0)) ** 2, -1).mean(0)
print(f"mean mse={mse_mean_pred:.2f}")
sample_cov = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(samples)
true_cov = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(y_test)
mse_cov_pred = jnp.sum(
    (sample_cov - true_cov).reshape(true_cov.shape[0], -1) ** 2, -1
).mean(0)
print(f"cov mse={mse_cov_pred:.2f}")

# Plotting conditional vs true posterior gP
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4 * 2, 4 * 2))
fig.subplots_adjust(wspace=0, hspace=0.0)

for ax, ys in zip(axes.T, [samples, y_test]):
    samples_mean = jnp.mean(ys, 0)
    plot_vf(x_test, samples_mean, ax[0])
    plot_vf(x_known, y_known, ax[0], color="r")

    plot_vf(x_test, samples_mean, ax[1])
    plot_vf(x_known, y_known, ax[1], color="r")
    covariances = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
    plot_cov(x_test, covariances, ax=ax[1])

plt.tight_layout()
plt.show()
# plt.savefig('conditional_ndp.png', dpi=300, facecolor='white', edgecolor='none')

# %%
# Equivariant posterior

key, subkey = jax.random.split(key)
# theta = jax.random.uniform(subkey) * jnp.pi
theta = np.deg2rad(90)
print(f"theta = {theta*360/2/jnp.pi:.2f} degrees")
R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
I = jnp.eye(2)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
for k, rot in enumerate([I, R]):
    # posterior = gp_posterior(mean_function, kernel, x_context @ rot.T, y_context @ rot.T, x)
    ys = jax.vmap(
        lambda key: conditional_sample(
            x_known @ rot.T, y_known @ rot.T, x_test, key=key
        )
    )(jax.random.split(key, num_samples))
    y = jnp.mean(ys, axis=0)
    # plot_vf(x_test, y, axes[k])
    # plot_vf(x_known @ rot.T, y_known @ rot.T, axes[k],  color="r")
    plot_vf(x_test @ rot, y @ rot, axes[k])
    plot_vf(x_known, y_known, axes[k], color="r")
    covariances = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
    plot_cov(x, covariances, ax=axes[k])
plt.savefig("conditional_ndp_rot.png", dpi=300, facecolor="white", edgecolor="none")

# %%
# Conditional likelihood evaluation of model
key = jax.random.PRNGKey(0)

batch_size = 8
num_context = 30

# x = jnp.repeat(radial_grid_2d(10, 30)[None], batch_size, 0)
x = radial_grid_2d(10, 30)
y = unflatten(
    prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x).sample(
        seed=key, sample_shape=(batch_size)
    ),
    output_dim,
)
num_points = x.shape[-2]
indices = jnp.arange(num_points)
# num_context = jax.random.randint(key1, (), minval=min_context, maxval=max_context)
num_target = num_points - num_context
perm = jax.random.permutation(key, indices)
x = jnp.repeat(x[None], batch_size, 0)
x_test = jnp.take(x, axis=-2, indices=perm[:num_target])
x_known = jnp.take(x, axis=-2, indices=perm[-num_context:])
y_test = jnp.take(y, axis=-2, indices=perm[:num_target])
y_known = jnp.take(y, axis=-2, indices=perm[-num_context:])

print("x_context", x_known.shape)
print("y_context", y_known.shape)
print("x_target", x_test.shape)
print("y_target", y_test.shape)

# num_samples = 10
# key, subkey = jax.random.split(key)
# key, subkey = jax.random.split(key)
# data_posterior_gp = posterior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x_known, y_known)(x_test)
# y_test = unflatten(data_posterior_gp.sample(seed=subkey, sample_shape=(num_samples)), output_dim)

n_test = y_test.shape[-2]

# true_logp = jax.vmap(data_posterior_gp.log_prob)(flatten(y_test)).squeeze()[:2]
true_logp = jax.vmap(
    lambda x_known, y_known, x_test, y_test: posterior_gp(
        mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x_known, y_known
    )(x_test).log_prob(flatten(y_test))
)(x_known, y_known, x_test, y_test).squeeze()
print("true_logp", true_logp.shape, true_logp.mean() / n_test)

# %%
num_steps = 100
solver = dfx.Tsit5()
# solver = dfx.Euler()
rtol: float = 1e-6
atol: float = 1e-6
# rtol: float = 1e-3
# atol: float = 1e-3
# rtol = atol = None
print(solver, rtol, atol)

key = jax.random.PRNGKey(4)
subkeys = jax.random.split(key, num=batch_size)
# k = 8
k = batch_size
# hutchinson_type="Gaussian"
hutchinson_type = "None"

ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1 - 1e-3, num_ts=5)


model_logp2, nfe = jax.vmap(
    jax.jit(
        lambda xc, yc, xs, ys, key: log_prob(
            sde,
            network,
            xs,
            ys,
            x_known=xc,
            y_known=yc,
            key=key,
            num_steps=num_steps,
            solver=solver,
            rtol=rtol,
            atol=atol,
            hutchinson_type=hutchinson_type,
            ts=ts,
            forward=True,
        )
    )
)(x_known[:k], y_known[:k], x_test[:k], y_test[:k], subkeys[:k])

# x = jnp.concatenate([x_test, x_known], axis=-2)
# y = jnp.concatenate([y_test, jnp.repeat(y_known[None], y_test.shape[0], 0)], axis=-2)
joint_logp, nfe = jax.vmap(
    jax.jit(
        lambda x, y, key: log_prob(
            sde,
            network,
            x,
            y,
            key=key,
            num_steps=num_steps,
            solver=solver,
            rtol=rtol,
            atol=atol,
            hutchinson_type=hutchinson_type,
            ts=ts,
            forward=True,
        )
    )
)(x[:k], y[:k], subkeys[:k])

context_logp, nfe = jax.vmap(
    jax.jit(
        lambda x, y, key: log_prob(
            sde,
            network,
            x,
            y,
            key=key,
            num_steps=num_steps,
            solver=solver,
            rtol=rtol,
            atol=atol,
            hutchinson_type=hutchinson_type,
            ts=ts,
            forward=True,
        )
    )
)(x_known[:k], y_known[:k], subkeys[:k])
model_logp = joint_logp - context_logp
# plot(ys, ts)

# log_prior = log_prior[:, -1]
# delta_logp = delta_logp[:, -1]
model_logp = model_logp[:, -1]
model_logp2 = model_logp2[:, -1]

# model_logp = log_prior + delta_logp
# print("log_prior ode", log_prior.shape, log_prior)
# print("delta_logp", delta_logp.shape, delta_logp)
print("model_logp (diff)", model_logp.shape, model_logp / n_test)
print("model_logp (direct)", model_logp2.shape, model_logp2 / n_test)
print("true_logp", true_logp.shape, true_logp[:k] / n_test)
print("nfe", nfe)
print("norm diff", jnp.linalg.norm(model_logp - true_logp[:k]) / batch_size)
# %%
