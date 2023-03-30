#%%
from __future__ import annotations
from functools import partial
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import seaborn as sns
import matplotlib

import jax
import jax.numpy as jnp
import diffrax as dfx
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import gpjax
from check_shapes import check_shapes

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.sde import LinearBetaSchedule, SDE, LinOpControlTerm
from neural_diffusion_processes.misc import flatten, unflatten

# %%
from jax.config import config
config.update("jax_enable_x64", True)
JITTER = 1e-12

def get_2d_grid(num, min_=-1, max_=1):
    x = jnp.linspace(min_, max_, num)
    x1, x2 = jnp.meshgrid(x, x)
    x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
    return x

# %%

from jaxtyping import Array
key = jax.random.PRNGKey(42)

output_dim = 2
beta_schedule = ndp.sde.LinearBetaSchedule()
x = get_2d_grid(25, -5, 5)

# k0 = ndp.kernels.RBFVec(output_dim)
k0 = ndp.kernels.RBFCurlFree()
k0_params = k0.init_params(None)

k1 = ndp.kernels.WhiteVec(output_dim)
mean_function = gpjax.Zero(output_dim)

limiting_params = {
        "kernel": k1.init_params(key),
        "mean_function": mean_function.init_params(key),
    }


kxx = k1.gram(limiting_params["kernel"], x).to_dense()
kxx = rearrange(kxx, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=output_dim, p2=output_dim) 
kxx = rearrange(kxx, 'n1 n2 p1 p2 -> (p1 n1) (p2 n2)') 
plt.matshow(kxx)
kxx = k0.gram(k0_params, x).to_dense()
kxx = rearrange(kxx, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=output_dim, p2=output_dim) 
kxx = rearrange(kxx, 'n1 n2 p1 p2 -> (p1 n1) (p2 n2)') 
plt.matshow(kxx)

# %%

import jax
import jax.numpy as jnp
import diffrax as dfx
from diffrax import AbstractStepSizeController, PIDController, ConstantStepSize
from diffrax import AbstractSolver, Dopri5, Tsit5
from check_shapes import check_shapes


def get_timesteps(t0, t1, num_ts):
    ts = jnp.exp(jnp.linspace(t0, t1, num_ts))
    ts = t0 + (t1 - t0) * (ts - ts[0]) / (ts[-1] - ts[0])
    return ts


def solve(
    sde: SDE,
    # network,
    x,
    y0,
    # *,
    key,
    num_steps: int = 100,
    # solver: AbstractSolver = dfx.Euler(),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    solver: AbstractSolver = dfx.Heun(),
    # stepsize_controller: AbstractStepSizeController = dfx.PIDController(rtol=1e-3, atol=1e-3),
):
    key, ykey = jax.random.split(key)
    # yT = sde.sample_prior(ykey, x) if yT is None else yT
    y_dim = y0.shape[-1]
    y0 = flatten(y0)

    t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
    dt = (t1 - t0) / num_steps  # TODO: dealing properly with endpoint?
    # ts = jnp.linspace(t0, t1, 9)[::-1]
    # saveat = dfx.SaveAt(ts=ts)

    shape = jax.ShapeDtypeStruct(y0.shape, y0.dtype)
    key, subkey = jax.random.split(key)
    bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    drift_sde = lambda t, yt, arg: flatten(
        # sde.drift(key, t, unflatten(yt, y_dim), arg, network)
        sde.drift(t, unflatten(yt, y_dim), arg)
    )
    diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
    terms = dfx.MultiTerm(
        dfx.ODETerm(drift_sde), LinOpControlTerm(diffusion, bm)
    )
    ts = get_timesteps(min(t0, t1), max(t0, t1), num_ts=5)
    # TODO: adaptive step?
    out = dfx.diffeqsolve(
        terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=x,
        adjoint=dfx.NoAdjoint(),
        stepsize_controller=stepsize_controller,
        saveat=dfx.SaveAt(ts=ts)
    )
    ys = out.ys.squeeze()
    return ts, unflatten(ys, y_dim)


# %%

from neural_diffusion_processes.kernels import sample_prior_gp

sde = ndp.sde.SDE(k1, mean_function, limiting_params, beta_schedule, True, True)

seed = 1
key = jax.random.PRNGKey(seed)

# Solves forward SDE for multiple initia states using vmap.
num_samples = 2
key, subkey = jax.random.split(key)
# y0s= sample_gp(subkey, k0, mean_function, x, num_samples=num_samples)
# y0s = sde.sample_prior(subkey, x)
y0s = sample_prior_gp(key, mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x)
print(y0s.shape)
subkeys = jax.random.split(key, num=num_samples)
out = jax.vmap(solve, in_axes=[None, None, None, 0])(sde, x, y0s, subkeys)
# out = solve(sde, x, y0s, key)
print(out[0].shape)
print(out[1].shape)
#%%

norm = matplotlib.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)

ts, ys = out
plot_num_timesteps = ys.shape[1]
fig, axes = plt.subplots(
    plot_num_timesteps, num_samples, sharex=True, sharey=True, figsize=(8, 16)
)
for j in range(plot_num_timesteps):
    for i in range(num_samples):
        if x.shape[-1] == 1:
            for o in range(output_dim):
                axes[j, i].plot(x, ys[i, j, :, o], '-', ms=2)
        elif x.shape[-1] == 2:
            y_norm = jnp.linalg.norm(ys[i, j], axis=-1)
            axes[j, i].quiver(
                x[:, 0],
                x[:, 1],
                ys[i, j, :, 0],
                ys[i, j, :, 1],
                color=cm(norm(y_norm)),
                scale=50,
                width=0.005,
            )  
        axes[j, 0].set_ylabel(f"t = {ts[0, j]:.2f}")
plt.tight_layout()
plt.show()

# %%
from jaxlinop import identity
solve_lower_triangular = partial(jax.scipy.linalg.solve_triangular, lower=True)  # L⁻¹ x
solve_upper_triangular = partial(jax.scipy.linalg.solve_triangular, lower=False)  # U⁻¹ x


def score(sde, t: Array, yt: Array, x: Array) -> Array:
        from neural_diffusion_processes.kernels import RBFCurlFree
        def pt(t):
            k0 = ndp.kernels.RBFCurlFree()
            return sde.pt(
                t,
                # y0=partial(gpjax.Zero(2), {}),
                y0=jnp.zeros_like(yt),
                k0=k0,
                k0_params=k0.init_params(None),
            )

        # k0 = RBFCurlFree()
        cov_coef = jnp.exp(-sde.beta_schedule.B(t))
        
        
        mu_t, k_t, params = pt(t)
        b = yt# - mu_t({}, x)
        n = b.shape[0] * b.shape[1]
        b = flatten(b)
        
        Sigma_t = k_t.gram(params['kernel'], x)
        JITTER = 1e-5
        Sigma_t = Sigma_t._add_diagonal(identity(n) * JITTER)
        Sigma_inv_b = Sigma_t.solve(b)
        # SigmaT = self.limiting_kernel.gram(self.limiting_params["kernel"], x)
        # out = - (SigmaT @ (SigmaT.T @ Sigma_inv_b))

        # Sigma_t = (1 - cov_coef) * sde.limiting_kernel.gram(sde.limiting_params['kernel'], x).to_dense()
        # Sigma_t += cov_coef * k0.gram(k0_params, x).to_dense() 
        # Lt = jnp.linalg.cholesky(Sigma_t + JITTER * jnp.eye(len(Sigma_t)))
        # Sigma_inv_b = solve_upper_triangular(jnp.transpose(Lt), solve_lower_triangular(Lt, b))

        out = -Sigma_inv_b
        return unflatten(out, yt.shape[-1])

sde.score = lambda key, t, yt, x, network: score(sde, t, yt, x)

#%%

# ts = get_timesteps(sde.beta_schedule.t1, 1e-3, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0+1e-5, num_ts=5)

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 1+num_samples)
rev_out = jax.vmap(lambda key, y: ndp.sde.reverse_solve(sde, None, x, key=key, yT=y, ts=ts))(np.stack(subkeys), ys[:, -1])
# rev_out = reverse_solve(sde, x, key, ys[0, -1])
print(rev_out.shape)
key, *subkeys = jax.random.split(key, 1+num_samples)
rev_out2 = jax.vmap(lambda key, y: ndp.sde.reverse_solve(sde, None, x, key=key, yT=y, ts=ts))(np.stack(subkeys), ys[:, -1])

# %%

fig, axes = plt.subplots(plot_num_timesteps, num_samples, sharex=True, sharey=True, figsize=(8, 16))
for t in range(plot_num_timesteps):
    for i in range(num_samples):
        if x.shape[-1] == 1:
            for o in range(output_dim):
                axes[t, i].plot(x, rev_out[i, t, :, o], '-', ms=2)
                axes[t, i].plot(x, rev_out2[i, t, :, o], ':', ms=2)
            # axes[t, i].set_ylim(-2.5, 2.5)
        elif x.shape[-1] == 2 and output_dim == 2:
            y_norm = jnp.linalg.norm(rev_out[i, t], axis=-1)
            axes[t, i].quiver(
                x[:, 0],
                x[:, 1],
                rev_out[i, t, :, 0],
                rev_out[i, t, :, 1],
                color=cm(norm(y_norm)),
                scale=50,
                width=0.005,
            )  
            # axes[t, i].set_ylim(-1, 1)
            # axes[t, i].set_xlim(-1, 1)

        axes[t, 0].set_ylabel(f"t = {ts[t]:.2f}")
plt.tight_layout()
plt.show()
#%%
if x.shape[-1] == 1:
    x_known = jnp.reshape(jnp.asarray([[-0.2, 0.2, 0.6]]), (-1, 1))
elif x.shape[-1] == 2:
    x_known = jnp.zeros((1, 2)) + 1.e-2

if x.shape[-1] == 1 and output_dim == 1:
    y_known = jnp.reshape(jnp.asarray([[0.0, -1.0, 0.0]]), (len(x_known), output_dim))
elif x.shape[-1] == 1 and output_dim == 2:
    y_known = jnp.reshape(jnp.asarray([[0.0, -1.0, 3.0, .2, 1.1, 0.]]), (len(x_known), output_dim))
elif x.shape[-1] == 2 and output_dim == 2:
    x_known = jnp.array([[0.25, 0.5], [0.5, 0.25], [-0.25, -0.25]]).astype(float) * 5.
    y_known = jnp.array([[1, 1], [1, -2], [-4, 3]]).astype(float)

if x.shape[-1] == 1:
    x_test = jnp.linspace(-1, 1, 101)[:, None]
elif x.shape[-1] == 2:
    x_test = get_2d_grid(25, -5, 5)

key = jax.random.PRNGKey(0)
# num_samples = 100 if x.shape[-1] == 1 else 9
num_samples = 10
samples = jax.vmap(lambda key: ndp.sde.conditional_sample2(sde, None, x_known, y_known, x_test, key=key, num_steps=100, num_inner_steps=5))(jax.random.split(key, num_samples))

# %%
if x.shape[-1] == 1:
    plt.figure()
    for o in range(output_dim):
        plt.plot(x_test, samples[..., o].T, f"C{o}-", alpha=.2)
        plt.plot(x_known, y_known[:, o], f"kx")


elif x.shape[-1] == 2 and output_dim == 2:
    fig, ax = plt.subplots(figsize=(5, 5))
    samples_mean = jnp.mean(samples, 0)
    s_norm = jnp.linalg.norm(samples_mean, axis=-1)
    ax.quiver(
        x_test[:, 0],
        x_test[:, 1],
        samples_mean[:, 0],
        samples_mean[:, 1],
        color=cm(norm(s_norm)),
        scale=50,
        width=0.005,
    )  
    ax.quiver(
        x_known[:, 0],
        x_known[:, 1],
        y_known[:, 0],
        y_known[:, 1],
        color='r',
        scale=50,
        width=0.005,
    )  
    # ax.set_ylim(-1, 1)
    # ax.set_xlim(-1, 1)

plt.tight_layout()
plt.show()
# plt.savefig('conditional_ndp.png', dpi=300, facecolor='white', edgecolor='none')

#%%
key, subkey = jax.random.split(key)
theta = jax.random.uniform(subkey) * jnp.pi
print(f"theta = {theta*360/2/jnp.pi:.2f} degrees")
R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
I = jnp.eye(2)
# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)


for k, rot in enumerate([I, R]):
    # posterior = gp_posterior(mean_function, kernel, x_context @ rot.T, y_context @ rot.T, x)
    samples = jax.vmap(lambda key: ndp.sde.conditional_sample2(sde, None, x_known @ rot.T, y_known @ rot.T, x_test, key=key, num_steps=100, num_inner_steps=5))(jax.random.split(key, num_samples))
    y = jnp.mean(samples, axis=0)
    y_norm = jnp.linalg.norm(y, axis=-1)

    axes[k].quiver(
        x_test[:, 0],
        x_test[:, 1],
        y[:, 0],
        y[:, 1],
        color=cm(norm(y_norm)),
        scale=50,
        width=0.005,
    )

    axes[k].quiver(
        (x_known @ rot.T)[:, 0],
        (x_known @ rot.T)[:, 1],
        (y_known @ rot.T)[:, 0],
        (y_known @ rot.T)[:, 1],
        color="r",
        scale=50,
        width=0.005,
    )
plt.savefig('conditional_ndp_rot.png', dpi=300, facecolor='white', edgecolor='none')
# %%
