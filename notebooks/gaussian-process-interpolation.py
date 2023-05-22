# %%
from __future__ import annotations
from functools import partial
import os
import math
# import setGPU

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np
from einops import rearrange
import gpjax
import haiku as hk
import jmp

from tensorflow_probability.substrates import jax as tfp

# tfd = tfp.distributions

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
from neural_diffusion_processes.models.attention import MultiOutputAttentionModel

JITTER = get_config().jitter

# %%
from jax.config import config

config.update("jax_enable_x64", True)

norm = mpl.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)

from jaxtyping import Array

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

output_dim = 1
beta_schedule = ndp.sde.LinearBetaSchedule(t0=1e-5, beta0=1e-4, beta1=15.0)
# beta_schedule = ndp.sde.LinearBetaSchedule(t0=1e-5, beta0=1e-4, beta1=10.0)
x = jnp.linspace(-2.2, 2.2, 200)[..., None]

k0 = ndp.kernels.RBFVec(output_dim)
# k0 = ndp.kernels.RBFCurlFree()
# k0 = ndp.kernels.RBFDivFree()
# k0_params = k0.init_params(None)
# k0_variance = 10
k0_variance = 1.0
lengthscale = 1.0
# lengthscale = 1
k0_params = {"variance": k0_variance, "lengthscale": lengthscale}
k0 = SumKernel([k0, ndp.kernels.WhiteVec(output_dim)])
k0_params = [k0_params, {"variance": 0.0}]

k1 = ndp.kernels.RBFVec(output_dim)
# k1 = ndp.kernels.WhiteVec(output_dim)
# k1_params = {"variance": k0_variance, "lengthscale": 1.}
# k1_params = {"variance": 1.0, "lengthscale": 1.0}
k1_params = {"variance": 1.0, "lengthscale": 0.2}
# k1_params = {"variance": 0.2, "lengthscale": 1.0}
# k1 = SumKernel([k1, ndp.kernels.WhiteVec(output_dim)])
# k1_params = [k1_params, {"variance": 0.1}]
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
    # residual_trick=True,
)
network = sde.get_exact_score(mean_function, k0, data_params)

plot_vf = partial(plot_vector_field, scale=50 * math.sqrt(k0_variance), width=0.005)
plot_cov = partial(plot_covariances, scale=0.3 / math.sqrt(k0_variance), zorder=-1)


def get_timesteps(t0, t1, num_ts):
    ts = jnp.exp(jnp.linspace(t0, t1, num_ts))
    ts = t0 + (t1 - t0) * (ts - ts[0]) / (ts[-1] - ts[0])
    return ts


font = {'family': 'serif',
        'serif': ['Times New Roman'],
        # 'weight' : 'bold',
        'size': 8.
        }
mpl.rc('font', **font)
mpl.rc('text', usetex='true')
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
pw = 5.50107
lw = pw / 2
dir_path = "../doc/equiv_sp/figs/"

def plot(ys, ts=None, N=10):
    plot_num_timesteps = ys.shape[1]
    fig, axes = plt.subplots(
        1,
        plot_num_timesteps,
        sharex=True,
        sharey=True,
        figsize=(pw, pw * 0.15),
    )
    # axes = axes if isinstance(type(axes[0]), np.ndarray) else axes[None, :]
    if plot_num_timesteps == 1:
        axes = axes[None, :]
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)
    # fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    for j in range(plot_num_timesteps):
        for i in range(N):
            for o in range(output_dim):
                axes[j].plot(x, ys[i, j, :, o], lw=0.5, color="C0", alpha=0.4)
        # if ts is not None:
        #     # axes[j].set_ylabel(f"t = {ts[j]:.2f}")
        #     axes[j].set_title(f"t = {ts[j]:.2f}", fontsize=8.)
    for ax in axes[1:]:
        ax.set_yticks([])
        ax.set_xticks([])
    # fig.tight_layout(w_pad=0.)
    fig.tight_layout(pad=0, w_pad=0.1, rect=(0,0.0,0.99,0.94))
    fig.show()
    return fig

# fig = plot(ys, ts, N=20)
# fig.savefig(os.path.join(dir_path, '1d/noising.pdf'))

# %%
# Forward process

seed = 4
key = jax.random.PRNGKey(seed)

# Solves forward SDE for multiple initia states using vmap.
num_samples = 100
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
# ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1, num_ts=4)
ts = jnp.array([sde.beta_schedule.t0, 0.1, 0.5, sde.beta_schedule.t1])

solve = jax.jit(
    lambda y, key: ndp.sde.sde_solve(
        sde,
        network,
        x,
        y=y,
        key=key,
        ts=ts,
        # prob_flow=True,
        prob_flow=False,
        atol=None,
        rtol=None,
        num_steps=100,
        forward=True,
    )
)
ys = jax.vmap(solve)(y0s, subkeys)
# solve = lambda y, key: ndp.sde.sde_solve(sde, network, x, y=y, key=key, ts=ts, prob_flow=False, atol=None, rtol=None, num_steps=100, forward=True)
# ys = solve(y0s[0], subkeys[0])

fig = plot(ys, ts, N=20)
fig.savefig(os.path.join(dir_path, '1d/noising.pdf'))


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
# Solve for conditional samples

key = jax.random.PRNGKey(1)

num_context = 25
# x = radial_grid_2d(10, 30)
x = jnp.linspace(-2.2, 2.2, 200)[..., None]
num_points = x.shape[-2]
indices = jnp.arange(num_points)
# num_context = jax.random.randint(key1, (), minval=min_context, maxval=max_context)
num_target = num_points - num_context
perm = jax.random.permutation(key, indices)
# x_test = jnp.take(x, axis=-2, indices=perm[:num_target])
x_test = x
# x_known = jnp.take(x, axis=-2, indices=perm[-num_context:])
x_known = jnp.array([-1., 1., 1.5])[..., None]
# idx = jax.random.permutation(key, jnp.arange(len(x_test)))
# x_known = x_test[idx[:num_context]] + 1e-5

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

# %%
num_steps = 50
# num_inner_steps = 50
num_inner_steps = 50
print(f"num_steps={num_steps} and num_inner_steps={num_inner_steps}")

conditional_sample = jax.jit(
    lambda x, y, x_eval, key: ndp.sde.conditional_sample2(
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
        # tau=0.5,
        # psi=1.5,
        # lambda0=1.5,
        tau=1.0,
        psi=1.0,
        lambda0=1.0,
        # prob_flow=True,
        prob_flow=False,
    )
)

def steer(shift):
    return jax.jit(
    jax.vmap(lambda key: conditional_sample(x_known - shift, y_known, x_test - shift, key=key))
)(jax.random.split(key, num_samples))
I, R = jnp.zeros(()), 2 * jnp.ones(())
ys, ys_steered = steer(I), steer(R)

#%%
# Plotting conditional vs true posterior gP
fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(lw, lw/1.63))
for ax, ys, shift in zip(axes, [ys, ys_steered], [I, R]):
    samples_mean = jnp.mean(ys, 0)
    # plot_vf(x_test, samples_mean, ax[0])
    # plot_vf(x_known, y_known, ax[0], color="r")
    # ax.plot(x_test, samples_mean, "-", ms=1, color="C0", alpha=0.2)
    for k in range(num_samples):
        ax.plot(x_test + shift, ys[k], "-", lw=0.5, color="C0", alpha=0.4)
    # ax.plot(x_known + shift, y_known, "ko")
    ax.scatter(x_known + shift, y_known, c="C3", zorder=2, s=5)
    ax.tick_params(axis='x', which='major', pad=1.)
    # ax.xaxis.set_ticks_position('none') 
    # ax.tick_params(direction='in', length=3, width=0.4)
    ax.tick_params(length=1)

    # ax.set_aspect('equal')
    # plot_vf(x_test, samples_mean, ax[1])
    # plot_vf(x_known, y_known, ax[1], color="r")

# axes[0].set_title(r"$f(x)$")
# axes[1].set_title(r"$f(g^{{-1}}x)$")
# axes[0].set_title(r'$f(x^*)|\{{x^c,y^c\}}$')
# axes[1].set_title(r'$f(x^*)|\{{g x^c,\rho(g) y^c\}}$')
axes[0].set_title(r'$f(x^*)|\mathcal{C}$')
axes[1].set_title(r'$f(x^*)|g\cdot\mathcal{C}$')
for ax in axes:
    ax.set_yticks([])
    # ax.set_xticks([])
fig.tight_layout()
fig.show()
fig.tight_layout(pad=0, w_pad=0.1, rect=(0,0.01,0.98,0.95))
# fig.savefig(os.path.join(dir_path, '2d/steerable_demo.pdf'), bbox_inches=mpl.transforms.Bbox([[0,0.14],[lw, lw/2.5 + 0.07]]))
fig.savefig(os.path.join(dir_path, '1d/conditional_ndp_shift.pdf'))


# %%
