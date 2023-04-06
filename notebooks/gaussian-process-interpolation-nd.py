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
import numpy as np
from einops import rearrange
import gpjax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import neural_diffusion_processes as ndp
from neural_diffusion_processes.sde import LinearBetaSchedule, SDE, LinOpControlTerm, SumKernel
from neural_diffusion_processes.utils import flatten, unflatten, JITTER
from neural_diffusion_processes.data import radial_grid_2d
from neural_diffusion_processes.utils.vis import plot_scalar_field, plot_vector_field, plot_covariances
from neural_diffusion_processes.kernels import sample_prior_gp, prior_gp
from neural_diffusion_processes.sde import log_prob

# %%
from jax.config import config
config.update("jax_enable_x64", True)

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
beta_schedule = ndp.sde.LinearBetaSchedule(t0 = 1e-3, beta0 = 1e-4, beta1 = 4.0)
x = radial_grid_2d(10, 30)

# k0 = ndp.kernels.RBFVec(output_dim)
k0 = ndp.kernels.RBFCurlFree()
# k0 = ndp.kernels.RBFDivFree()
# k0_params = k0.init_params(None)
k0_variance = 10
k0_params = {"variance": k0_variance, "lengthscale": 2.23606797749979}
k0 = SumKernel([k0, ndp.kernels.WhiteVec(output_dim)])
k0_params = [k0_params, {"variance": 0.02}]

k1 = ndp.kernels.RBFVec(output_dim)
# k1 = ndp.kernels.WhiteVec(output_dim)
# k1_params = {"variance": k0_variance, "lengthscale": 2.}
k1_params = {"variance": k0_variance, "lengthscale": 2.}
k1 = SumKernel([k1, ndp.kernels.WhiteVec(output_dim)])
k1_params = [k1_params, {"variance": 1.}]

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

sde = ndp.sde.SDE(k1, mean_function, limiting_params, beta_schedule, True, True, exact=True)

plot_vf = partial(plot_vector_field, scale=50*math.sqrt(k0_variance), width=0.005)
plot_cov = partial(plot_covariances, scale=0.3/math.sqrt(k0_variance), zorder=-1)


from jaxlinop import identity
solve_lower_triangular = partial(jax.scipy.linalg.solve_triangular, lower=True)  # L⁻¹ x
solve_upper_triangular = partial(jax.scipy.linalg.solve_triangular, lower=False)  # U⁻¹ x

def get_timesteps(t0, t1, num_ts):
    ts = jnp.exp(jnp.linspace(t0, t1, num_ts))
    ts = t0 + (t1 - t0) * (ts - ts[0]) / (ts[-1] - ts[0])
    return ts

def plot(ys, ts=None):
    plot_num_timesteps = ys.shape[1]
    fig, axes = plt.subplots(
        plot_num_timesteps, 2, sharex=True, sharey=True, figsize=(8*2, 8*plot_num_timesteps)
    )
    # axes = axes if isinstance(type(axes[0]), np.ndarray) else axes[None, :]
    if plot_num_timesteps == 1:
        axes = axes[None, :]
    fig.subplots_adjust(wspace=0, hspace=0.)

    for j in range(plot_num_timesteps):
        for i in range(2):
            if x.shape[-1] == 1:
                for o in range(output_dim):
                    axes[j, i].plot(x, ys[i, j, :, o], '-', ms=2)
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
y0s = sample_prior_gp(key, mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x, num_samples=num_samples, obs_noise=0.)
print(y0s.shape)
subkeys = jax.random.split(key, num=num_samples)
ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1, num_ts=5)

solve = jax.jit(lambda y, key: ndp.sde.sde_solve(sde, None, x, y=y, key=key, ts=ts, prob_flow=True, atol=None, rtol=None, num_steps=100, forward=True))
ys = jax.vmap(solve)(y0s, subkeys)

plot(ys, ts)

#%%
# Backward

# ts = get_timesteps(sde.beta_schedule.t1, 1e-3, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0+1e-8, num_ts=5)

reverse_solve = lambda key, y: ndp.sde.sde_solve(sde, None, x, key=key, y=y, ts=ts, prob_flow=True,
# solver=dfx.Heun(), rtol=1e-3, atol=1e-3, num_steps=100)
solver=dfx.Euler(), rtol=None, atol=None, num_steps=100)

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 1+num_samples)
# rev_out = reverse_solve(key, ys[0, -1])
yT = sample_prior_gp(key, mean_function, k1, {"kernel": k1_params, "mean_function": {}}, x, num_samples=num_samples)
# yT = ys[:, -1]

rev_out = jax.vmap(reverse_solve)(np.stack(subkeys), yT)
# print(rev_out.shape)
key, *subkeys = jax.random.split(key, 1+num_samples)
rev_out2 = jax.vmap(reverse_solve)(np.stack(subkeys), yT)

plot(rev_out, ts)
# plot(rev_out2, ts)

#%%
# Solve for conditional samples

key = jax.random.PRNGKey(0)

x_test = radial_grid_2d(10, 30)
idx = jax.random.permutation(key, jnp.arange(len(x_test)))
x_known = x_test[idx[:20]] + 1e-5

from neural_diffusion_processes.kernels import prior_gp, posterior_gp
num_samples = 20

# key, subkey = jax.random.split(key)
y_known = unflatten(prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x_known).sample(seed=key, sample_shape=()), output_dim)
# y_known = unflatten(prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}})(x_test).sample(seed=subkey, sample_shape=()), output_dim)

# key, subkey = jax.random.split(key)
data_posterior_gp = posterior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, x_known, y_known)(x_test)
y_test = unflatten(data_posterior_gp.sample(seed=subkey, sample_shape=(num_samples)), output_dim)
# y_test = unflatten(data_posterior_gp.mean(), output_dim)[None, ...]

def conditional_sample2(
    sde: SDE,
    network,
    x_context,
    y_context,
    x_test,
    *,
    key,
    num_steps: int = 100,
    num_inner_steps: int = 5,
    prob_flow: bool = True,
    langevin_kernel = True,
    psi: float = 1.,
    lambda0: float = 1.,
    tau: float = None,
):
    # TODO: Langevin dynamics option

    num_context = len(x_context)
    num_target = len(x_test)
    y_dim = y_context.shape[-1]
    shape_augmented_state = [(num_context + num_target) * y_dim]
    x_augmented = jnp.concatenate([x_context, x_test], axis=0)

    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
    dt = ts[0] - ts[1]
    tau = tau if tau is not None else t1

    solver = dfx.Euler()

    diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
    if not prob_flow:
        # reverse SDE:
        reverse_drift_sde = lambda t, yt, arg: flatten(sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, network))

        shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
        key, subkey = jax.random.split(key)
        bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=key)
        terms_reverse = dfx.MultiTerm(
            dfx.ODETerm(reverse_drift_sde), LinOpControlTerm(diffusion, bm)
        )
    else:
        # reverse ODE:
        reverse_drift_ode = lambda t, yt, arg: flatten(
            sde.reverse_drift_ode(key, t, unflatten(yt, y_dim), arg, network)
            # reverse_drift_ode(sde, key, t, unflatten(yt, y_dim), arg, network)
        )
        terms_reverse = dfx.ODETerm(reverse_drift_ode)

    # langevin dynamics:
    def reverse_drift_langevin(t, yt, x) -> Array:
        yt = unflatten(yt, y_dim)
        score = flatten(sde.score(key, t, yt, x, network))
        if langevin_kernel:
            if sde.precond:
                score = score
            else:
                score = sde.limiting_gram(x) @ score
        else:
            if sde.precond:
                score = sde.limiting_gram(x).solve(score)
            else:
                score = score
        return 0.5 * sde.beta_schedule(t) * score
        # return 0.5 * score
    
    def diffusion_langevin(t, yt, x):
        if langevin_kernel:
            return diffusion(t, yt, x)
            # return sde.limiting_gram(x)._add_diagonal(JITTER * identity(x.shape[0]*x.shape[1])).to_root()
        else:
            return jnp.sqrt(sde.beta_schedule(t)) * identity(yt.shape[-1])
            # return identity(yt.shape[-1])

    key, subkey = jax.random.split(key)
    shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    # bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
    langevin_terms = dfx.MultiTerm(
        dfx.ODETerm(reverse_drift_langevin),
        LinOpControlTerm(diffusion_langevin, bm)
    )

    def sample_marginal(key, t, x_context, y_context):
        if len(y_context) == 0:
            return y_context
        else:
            return flatten(sde.sample_marginal(key, t, x_context, y_context))


    def inner_loop(key, ys, t):
        print("compiling Langevin inner_loop")
        # reverse step
        yt, yt_context = ys
        yt_context = sample_marginal(key, t, x_context, y_context) # NOTE: should resample?
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

        # yt_m_dt, *_ = solver.step(
        #     langevin_terms,
        #     t - dt,
        #     t,
        #     # t + dt,
        #     yt_augmented,
        #     x_augmented,
        #     None,
        #     made_jump=False,
        # )

        yt_m_dt = yt_augmented
        yt_m_dt += lambda0 * psi * dt * reverse_drift_langevin(t - dt, yt_augmented, x_augmented)
        # noise = jnp.sqrt(psi) * langevin_terms.contr(t - dt, t)[1]
        noise = jnp.sqrt(psi) * jnp.sqrt(dt) * jax.random.normal(key, shape=yt_augmented.shape)
        yt_m_dt += diffusion_langevin(t - dt, yt_augmented, x_augmented) @ noise
        # yt_m_dt += langevin_terms.contr(t, t)[0] * langevin_terms.vf(t, yt_augmented, x_augmented)[0]
        # yt_m_dt += langevin_terms.vf(t, yt_augmented, x_augmented)[1] @ noise
        
        yt = yt_m_dt[num_context * y_dim :]
        # strip context from augmented state
        return (yt, yt_context), yt_m_dt

    def outer_loop(key, yt, t):
        print("compiling Euler-Maruyama outer_loop")
        # yt_context = sde.sample_marginal(key, t, x_context, y_context)
        yt_context = sample_marginal(key, t, x_context, y_context)
        # yt_context = y_context #NOTE: doesn't need to be noised?
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

        yt_m_dt, *_ = solver.step(
            terms_reverse,
            t,
            t - dt,
            yt_augmented,
            x_augmented,
            None,
            made_jump=False,
        )
        # yt_m_dt = yt_augmented
        # yt_m_dt += -dt * reverse_drift_sde(t, yt_augmented, x_augmented)
        # # yt_m_dt += terms_reverse.contr(t, t-dt) * terms_reverse.vf(t, yt_augmented, x_augmented)
        # noise = jax.random.normal(key, shape=yt_augmented.shape)
        # yt_m_dt += jnp.sqrt(dt) * diffusion(t, yt_augmented, x_augmented) @ noise

        yt = yt_m_dt[num_context * y_dim :]
        # if num_inner_steps > 0:
        def corrector(key, yt, yt_context, t):
            print("corrector")
            _, yt_m_dt = jax.lax.scan(
                lambda ys, key: inner_loop(key, ys, t),
                (yt, yt_context),
                jax.random.split(key, num_inner_steps),
            )
            yt = yt_m_dt[-1][num_context * y_dim :]
            return yt
    
        yt = jax.lax.cond(
            tau > t,
            corrector,
            lambda key, yt, yt_context, t: yt,
            key, yt, yt_context, t
        )
        return yt, yt

    key, subkey = jax.random.split(key)
    yT = flatten(sde.sample_prior(subkey, x_test))

    xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
    y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
    return unflatten(y0, y_dim)



num_steps = 50
num_inner_steps = 50
print(f"num_steps={num_steps} and num_inner_steps={num_inner_steps}")
import time
start = time.time()
# conditional_sample = jax.jit(lambda  x, y, x_eval, key: ndp.sde.conditional_sample2(sde, None, x, y, x_eval, key=key, num_steps=num_steps, num_inner_steps=num_inner_steps, langevin_kernel=True, alpha=3.))
conditional_sample = jax.jit(lambda  x, y, x_eval, key: conditional_sample2(sde, None, x, y, x_eval, key=key, num_steps=num_steps, num_inner_steps=num_inner_steps, langevin_kernel=True, tau=.5, psi=2., lambda0=1., prob_flow=True))

samples = jax.jit(jax.vmap(lambda key:conditional_sample(x_known, y_known, x_test, key=key)))(jax.random.split(key, num_samples))
end = time.time()
print(f"time={end - start:.2f}")
mse_mean_pred = jnp.sum((samples.mean(0) - y_test.mean(0)) ** 2, -1).mean(0)
print(f"mean mse={mse_mean_pred:.2f}")
sample_cov = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(samples)
true_cov = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(y_test)
mse_cov_pred = jnp.sum((sample_cov - true_cov).reshape(true_cov.shape[0], -1) ** 2, -1).mean(0)
print(f"cov mse={mse_cov_pred:.2f}")


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
#Likelihood evaluation

# from neural_diffusion_processes.sde import get_div_fn, div_noise

# @check_shapes("x: [N, x_dim]", "y: [N, y_dim]")#, "return: []")
# def log_prob(
#     sde: SDE,
#     network,
#     x,
#     y,
#     *,
#     key,
#     num_steps: int = 100,
#     solver: AbstractSolver = dfx.Tsit5(),
#     rtol: float = 1e-3,
#     atol: float = 1e-3,
#     hutchinson_type = 'None',
#     forward: bool = True,
#     ts = None
# ):
#     y_dim = y.shape[-1]
#     y = flatten(y)

#     if rtol is None or atol is None:
#         stepsize_controller = ConstantStepSize()
#     else:
#         stepsize_controller =  dfx.PIDController(rtol=rtol, atol=atol)

#     if forward:
#         t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
#     else:
#         t1, t0 = sde.beta_schedule.t0, sde.beta_schedule.t1
#     dt = (t1 - t0) / num_steps
#     print(t0, t1, dt)

#     reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
#         key, t, yt, arg, network
#     )
#     div_fn = get_div_fn(reverse_drift_ode, hutchinson_type)


#     @jax.jit
#     def logp_wrapper(t, carry, static_args):
#         yt, _ = carry
#         eps, x = static_args
#         yt = unflatten(yt, y_dim)

#         drift = flatten(reverse_drift_ode(t, yt, x))
#         logp = div_fn(t, yt, x, eps)
#         # drift = jnp.zeros_like(yt)
#         # logp = jnp.zeros(())
#         return drift, logp

#     terms = dfx.ODETerm(logp_wrapper)
#     #NOTE: should we resample?
#     eps = div_noise(key, y.shape, hutchinson_type)


#     # reverse_drift_ode = lambda t, yt, arg: flatten(sde.reverse_drift_ode(
#     #     key, t, unflatten(yt, y_dim), arg, network
#     # ))
#     # terms = dfx.ODETerm(reverse_drift_ode)

#     saveat = dfx.SaveAt(t1=True) if ts is None else dfx.SaveAt(ts=ts)
#     sol = dfx.diffeqsolve(
#         terms,
#         solver,
#         t0=t0,
#         t1=t1,
#         dt0=dt,
#         y0=(y, 0.0),
#         args=(eps, x),
#         adjoint=dfx.NoAdjoint(),
#         stepsize_controller=stepsize_controller,
#         saveat=saveat
#     )
#     yT, delta_logp = sol.ys
#     yT = unflatten(yT, y_dim)
#     nfe = sol.stats['num_steps']
#     # yT, delta_logp = yT.squeeze(0), delta_logp.squeeze(0)
#     logp_prior = jax.vmap(lambda y: sde.log_prob_prior(x, y))(yT)

#     return logp_prior, delta_logp, nfe, yT

key = jax.random.PRNGKey(1)
num_samples = 2
subkeys = jax.random.split(key, num=num_samples)

# dist = prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.02)(x)
dist_T = prior_gp(mean_function, k1, {"kernel": k1_params, "mean_function": {}}, obs_noise=0.)(x)
yT = dist_T.sample(seed=subkey, sample_shape=(num_samples))

# ts = get_timesteps(sde.beta_schedule.t0+1e-6, sde.beta_schedule.t1, num_ts=5)
ts = get_timesteps(sde.beta_schedule.t1, sde.beta_schedule.t0+1e-6, num_ts=5)
# ts = None
num_steps = 100
# solver = dfx.Tsit5()
solver = dfx.Euler()
# solver = dfx.Dopri5()
# rtol: float = 1e-3
# atol: float = 1e-4
rtol = atol = None
hutchinson_type = "None"

log_prior, delta_logp, nfe, ys = jax.vmap(lambda y, key: log_prob(sde, None, x, y, key=key, 
num_steps=num_steps, solver=solver, rtol=rtol, atol=atol, hutchinson_type=hutchinson_type, ts=ts, forward=False))(unflatten(yT, output_dim), subkeys)

plot(ys)

#%%
# Solves forward SDE for multiple initia states using vmap.
num_samples = 2
key, subkey = jax.random.split(key)
dist = prior_gp(mean_function, k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.)(x)
y0s = dist.sample(seed=subkey, sample_shape=(num_samples))
print("mean var", (jnp.std(y0s, 0) ** 2).mean())
true_logp = jax.vmap(dist.log_prob)(y0s).squeeze()
print("true_logp", true_logp.shape, true_logp)

# log_prior = jax.vmap(dist_T.log_prob)(y0s).squeeze()
# print("log_prior", log_prior.shape, log_prior)

y0s = unflatten(y0s, output_dim)

num_steps = 100
solver = dfx.Tsit5()
# solver = dfx.Heun()
# solver = dfx.Euler()
# solver = dfx.Dopri5()
rtol: float = 1e-6
atol: float = 1e-6
# rtol = atol = None
print(solver, rtol, atol)

key = jax.random.PRNGKey(4)
subkeys = jax.random.split(key, num=num_samples)
# ts = get_timesteps(sde.beta_schedule. t0,sde.beta_schedule.t1, num_ts=5)
# hutchinson_type="Gaussian"
hutchinson_type = "None"

ts = get_timesteps(sde.beta_schedule.t0, sde.beta_schedule.t1, num_ts=5)
# ts = None

model_logp, nfe = jax.vmap(jax.jit(lambda y, key: log_prob(sde, None, x, y, key=key, num_steps=num_steps, solver=solver, rtol=rtol, atol=atol, hutchinson_type=hutchinson_type, ts=ts, forward=True)))(y0s, subkeys)
# log_prior, delta_logp, nfe, yT = jax.vmap(jax.jit(lambda y, key: log_prob(sde, None, x, y, key=key, hutchinson_type=hutchinson_type, ts=ts)))(y0s, subkeys)

# ys = unflatten(y0s, output_dim)[:, None, ...]
# ys = yT
plot(yT, ts)

# print("mean var ys", (jnp.std(yT, 0) ** 2).mean())
# print("mean mean ys", (jnp.mean(yT, 0) ** 2).mean())
# log_prior = log_prior[:, -1]
# delta_logp = delta_logp[:, -1]

# model_logp = log_prior + delta_logp
# print("log_prior ode", log_prior.shape, log_prior)
# print("delta_logp", delta_logp.shape, delta_logp)
print("model_logp", model_logp.shape, model_logp)
print("true_logp", true_logp.shape, true_logp)
print("nfe", nfe)
print("norm diff", jnp.linalg.norm(model_logp-true_logp) / num_samples)

# %%
