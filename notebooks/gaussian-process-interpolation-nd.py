#%%
from __future__ import annotations
from functools import partial
import os
import math
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax as dfx
from diffrax import AbstractStepSizeController, PIDController, ConstantStepSize
from diffrax import AbstractSolver, Dopri5, Tsit5
import numpy as np
from einops import rearrange
import gpjax
from check_shapes import check_shapes

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.sde import LinearBetaSchedule, SDE, LinOpControlTerm, SumKernel
from neural_diffusion_processes.utils.misc import flatten, unflatten
from neural_diffusion_processes.data import radial_grid_2d
from neural_diffusion_processes.utils.vis import plot_scalar_field, plot_vector_field, plot_covariances

# %%
from jax.config import config
config.update("jax_enable_x64", True)
# JITTER = 1e-12
JITTER = 1e-5

norm = matplotlib.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)

# def get_2d_grid(num, min_=-1, max_=1):
#     x = jnp.linspace(min_, max_, num)
#     x1, x2 = jnp.meshgrid(x, x)
#     x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
#     return x

from jaxtyping import Array
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

output_dim = 2
beta_schedule = ndp.sde.LinearBetaSchedule(t0 = 5e-4, beta0 = 1e-4, beta1 = 15.0)
x = radial_grid_2d(10, 30)

# k0 = ndp.kernels.RBFVec(output_dim)
k0 = ndp.kernels.RBFCurlFree()
# k0 = ndp.kernels.RBFDivFree()
# k0_params = k0.init_params(None)
k0_params = {"variance": 10, "lengthscale": 2.23606797749979}

k1 = ndp.kernels.WhiteVec(output_dim)
# # k1 = ndp.kernels.RBFVec(output_dim)
k1_params = {"variance": k0_params["variance"], "lengthscale": 2.}

k1 = SumKernel([ndp.kernels.RBFVec(output_dim), ndp.kernels.WhiteVec(output_dim)])
# k1_params = k1.init_params(key)
k1_params = [{"variance": k0_params["variance"], "lengthscale": 2.}, {"variance": 5.}]
# k1_params = [{"variance": 0., "lengthscale": 2.}, {"variance": 10.}]

mean_function = gpjax.Zero(output_dim)

limiting_params = {
        "kernel": k1_params,
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

sde = ndp.sde.SDE(k1, mean_function, limiting_params, beta_schedule, True, True)

plot_vf = partial(plot_vector_field, scale=50*math.sqrt(k0_params["variance"]), width=0.005)
plot_cov = partial(plot_covariances, scale=0.3/math.sqrt(k0_params["variance"]), zorder=-1)


from jaxlinop import identity
solve_lower_triangular = partial(jax.scipy.linalg.solve_triangular, lower=True)  # L⁻¹ x
solve_upper_triangular = partial(jax.scipy.linalg.solve_triangular, lower=False)  # U⁻¹ x


def score(sde, t: Array, yt: Array, x: Array, kernel) -> Array:
    from neural_diffusion_processes.kernels import RBFCurlFree
    def pt(t):
        k0 = ndp.kernels.RBFCurlFree()
        return sde.pt(
            t,
            # y0=partial(gpjax.Zero(2), {}),
            y0=jnp.zeros_like(yt),
            k0=k0,
            k0_params={"variance": 10, "lengthscale": 2.23606797749979},
            # k0_params=k0_params,
        )

    # k0 = RBFCurlFree()
    cov_coef = jnp.exp(-sde.beta_schedule.B(t))
    
    
    # mu_t, k_t, params = pt(t)
    b = yt# - mu_t({}, x)
    n = b.shape[0] * b.shape[1]
    b = flatten(b)
    
    # Sigma_t = k_t.gram(params['kernel'], x)
    # JITTER = 1e-5
    # Sigma_t = Sigma_t._add_diagonal(identity(n) * JITTER)
    # Sigma_inv_b = Sigma_t.solve(b)

    Sigma_t2 = (1 - cov_coef) * sde.limiting_kernel.gram(sde.limiting_params['kernel'], x).to_dense()
    Sigma_t2 += cov_coef * k0.gram(k0_params, x).to_dense()
    Sigma_t2 += JITTER * jnp.eye(n)
    Lt = jnp.linalg.cholesky(Sigma_t2)
    Sigma_inv_b2 = solve_upper_triangular(jnp.transpose(Lt), solve_lower_triangular(Lt, b))

    SigmaT = sde.limiting_kernel.gram(sde.limiting_params["kernel"], x)
    
    if kernel:
        out = - (SigmaT @ Sigma_inv_b2)
    else:
        out = - Sigma_inv_b2

    return unflatten(out, yt.shape[-1])

sde.score = lambda key, t, yt, x, network, kernel=True: score(sde, t, yt, x, kernel)
# score(sde, 0.5, ys[0, -1], x)


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
    ts=None
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
    saveat = dfx.SaveAt(t1=True) if ts is None else dfx.SaveAt(ts=ts)
    
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
        saveat=saveat
    )
    ys = out.ys.squeeze()
    return ts, unflatten(ys, y_dim)


# %%

from neural_diffusion_processes.kernels import sample_prior_gp

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
ts = get_timesteps(sde.beta_schedule. t0,sde.beta_schedule.t1, num_ts=5)
out = jax.vmap(lambda key: solve(sde, x, y0s, key, ts=ts))(subkeys)
# out = solve(sde, x, y0s, key)
print(out[0].shape)
print(out[1].shape)
#%%
ts, ys = out
plot_num_timesteps = ys.shape[1]
fig, axes = plt.subplots(
    plot_num_timesteps, num_samples, sharex=True, sharey=True, figsize=(8*num_samples, 8*plot_num_timesteps)
)
fig.subplots_adjust(wspace=0, hspace=0.)

for j in range(plot_num_timesteps):
    for i in range(num_samples):
        if x.shape[-1] == 1:
            for o in range(output_dim):
                axes[j, i].plot(x, ys[i, j, :, o], '-', ms=2)
        elif x.shape[-1] == 2:
            plot_vf(x, ys[i, j], axes[j, i])
        axes[j, 0].set_ylabel(f"t = {ts[0, j]:.2f}")
plt.tight_layout()
plt.show()

#%%

# ts = get_timesteps(sde.beta_schedule.t1, 1e-3, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0+1e-8, num_ts=5)

reverse_solve = lambda key, y: ndp.sde.reverse_solve(sde, None, x, key=key, yT=y, ts=ts, solver=dfx.Heun()) #, stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3))

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 1+num_samples)
# rev_out = reverse_solve(key, ys[0, -1])
rev_out = jax.vmap(reverse_solve)(np.stack(subkeys), ys[:, -1])
# print(rev_out.shape)
key, *subkeys = jax.random.split(key, 1+num_samples)
rev_out2 = jax.vmap(reverse_solve)(np.stack(subkeys), ys[:, -1])

# %%

fig, axes = plt.subplots(
    plot_num_timesteps, num_samples, sharex=True, sharey=True, figsize=(8*num_samples, 8*plot_num_timesteps)
)
fig.subplots_adjust(wspace=0, hspace=0.)

for t in range(plot_num_timesteps):
    for i in range(num_samples):
        if x.shape[-1] == 1:
            for o in range(output_dim):
                axes[t, i].plot(x, rev_out[i, t, :, o], '-', ms=2)
                axes[t, i].plot(x, rev_out2[i, t, :, o], ':', ms=2)
        elif x.shape[-1] == 2 and output_dim == 2:
            plot_vf(x, rev_out2[i, t], axes[t, i])

        axes[t, 0].set_ylabel(f"t = {ts[t]:.2f}")
plt.tight_layout()
plt.show()

#%%
# Solve for conditional samples
key = jax.random.PRNGKey(0)

x_test = radial_grid_2d(10, 30)
idx = jax.random.permutation(key, jnp.arange(len(x_test)))
x_known = x_test[idx[:20]] + 1e-2

from neural_diffusion_processes.kernels import prior_gp, posterior_gp
num_samples = 20

# key, subkey = jax.random.split(key)
y_known = unflatten(prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x_known).sample(seed=key, sample_shape=()), output_dim)
# y_known = unflatten(prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x_test).sample(seed=subkey, sample_shape=()), output_dim)

# key, subkey = jax.random.split(key)
data_posterior_gp = posterior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x_known, y_known)(x_test)
y_test = unflatten(data_posterior_gp.sample(seed=subkey, sample_shape=(num_samples)), output_dim)
# y_test = unflatten(data_posterior_gp.mean(), output_dim)[None, ...]

num_steps = 100
num_inner_steps = 10
print(f"num_steps={num_steps} and num_inner_steps={num_inner_steps}")
conditional_sample = jax.jit(lambda  x, y, x_eval, key: ndp.sde.conditional_sample2(sde, None, x, y, x_eval, key=key, num_steps=num_steps, num_inner_steps=num_inner_steps, langevin_kernel=True, alpha=1.))

samples = jax.vmap(lambda key:conditional_sample(x_known, y_known, x_test, key=key))(jax.random.split(key, num_samples))

# %%
# Plotting conditional vs true posterior gP
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8*2, 8*2))
fig.subplots_adjust(wspace=0, hspace=0.)

for ax, ys in zip(axes.T, [samples, y_test]):
    samples_mean = jnp.mean(ys, 0)
    plot_vf(x, samples_mean, ax[0])
    plot_vf(x_known, y_known, ax[0], color='r')

    plot_vf(x, samples_mean, ax[1])
    plot_vf(x_known, y_known, ax[1], color='r')
    covariances = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
    plot_cov(x, covariances, ax=ax[1])

plt.tight_layout()
plt.show()
# plt.savefig('conditional_ndp.png', dpi=300, facecolor='white', edgecolor='none')

# %%

#%%
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
    ys = jax.vmap(lambda key: conditional_sample(x_known @ rot.T, y_known @ rot.T, x_test, key=key))(jax.random.split(key, num_samples))
    y = jnp.mean(ys, axis=0)
    # plot_vf(x_test, y, axes[k])
    # plot_vf(x_known @ rot.T, y_known @ rot.T, axes[k],  color="r")
    plot_vf(x_test @ rot, y @ rot, axes[k])
    plot_vf(x_known, y_known, axes[k],  color="r")
    covariances = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
    plot_cov(x, covariances, ax=axes[k])
plt.savefig('conditional_ndp_rot.png', dpi=300, facecolor='white', edgecolor='none')

# %%
#Likelihood

from neural_diffusion_processes.kernels import prior_gp
# from neural_diffusion_processes.sde import log_prob
from neural_diffusion_processes.sde import get_div_fn, div_noise

@check_shapes("x: [N, x_dim]", "y: [N, y_dim]")#, "return: []")
def log_prob(
    sde: SDE,
    network,
    x,
    y,
    *,
    key,
    num_steps: int = 100,
    # solver: AbstractSolver = dfx.Euler(),
    # solver: AbstractSolver = dfx.Heun(),
    # solver: AbstractSolver = Dopri5(),
    solver: AbstractSolver = Tsit5(),
    rtol: float = 1e-3,
    atol: float = 1e-3,
    # stepsize_controller: AbstractStepSizeController = dfx.PIDController(rtol=1e-3, atol=1e-3)
    hutchinson_type = 'None'
):
    if rtol is None or atol is None:
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller =  dfx.PIDController(rtol=rtol, atol=atol)

    reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
        key, t, yt, arg, network
    )

    div_fn = get_div_fn(reverse_drift_ode, hutchinson_type)
    @jax.jit
    def logp_wrapper(t, carry, static_args):
        yt, _ = carry
        eps, x = static_args

        drift = reverse_drift_ode(t, yt, x)
        logp = div_fn(t, yt, x, eps)
        # drift = jnp.zeros_like(yt)
        # logp = jnp.zeros(())

        return drift, logp

    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    dt = (t1 - t0) / num_steps

    term = dfx.ODETerm(logp_wrapper)
    # #NOTE: should we resample?
    eps = div_noise(key, y.shape, hutchinson_type)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t1,
        t0,
        -dt,
        (y, 0.0),
        (eps, x),
        stepsize_controller=stepsize_controller,
    )
    yT, delta_logp = sol.ys
    nfe = sol.stats['num_steps']
    yT, delta_logp = yT.squeeze(0), delta_logp.squeeze(0)
    logp_prior = sde.log_prob_prior(x, yT)

    return logp_prior, delta_logp, nfe

key = jax.random.PRNGKey(1)

# Solves forward SDE for multiple initia states using vmap.
num_samples = 10
key, subkey = jax.random.split(key)
dist = prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x)
y0s = dist.sample(seed=subkey, sample_shape=(num_samples))
true_logp = jax.vmap(dist.log_prob)(y0s)
print("true_logp", true_logp.shape, true_logp)

num_steps = 100
solver: AbstractSolver = dfx.Tsit5()
# solver: AbstractSolver = dfx.Euler()
# solver: AbstractSolver = dfx.Dopri5()
rtol: float = 1e-3
atol: float = 1e-4
# rtol = atol = None
print(solver, atol)

key = jax.random.PRNGKey(4)
subkeys = jax.random.split(key, num=num_samples)
# ts = get_timesteps(sde.beta_schedule. t0,sde.beta_schedule.t1, num_ts=5)
# hutchinson_type="Gaussian"
hutchinson_type="None"

y0s = unflatten(y0s, output_dim)
log_prior, delta_logp, nfe = jax.vmap(lambda y, key: log_prob(sde, None, x, y, key=key, 
num_steps=num_steps, solver=solver, rtol=rtol, atol=atol, hutchinson_type=hutchinson_type))(y0s, subkeys)
model_logp = log_prior - delta_logp
print("log_prior", log_prior.shape, log_prior)
print("delta_logp", delta_logp.shape, delta_logp)
print("model_logp", model_logp.shape, model_logp)
print("nfe", nfe)
print("norm diff", jnp.linalg.norm(model_logp-true_logp) / num_samples)
# out = jax.vmap(lambda key: solve(sde, x, y0s, key, ts=ts))(subkeys)
# out = solve(sde, x, y0s, key)
# print(out[0].shape)
# print(out[1].shape)