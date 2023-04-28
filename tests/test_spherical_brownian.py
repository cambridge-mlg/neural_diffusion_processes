# %%
from __future__ import annotations
import os
import setGPU

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np

import neural_diffusion_processes as ndp
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

beta_schedule = ndp.sde.LinearBetaSchedule(t0=1e-5, beta0=1e-4, beta1=10.0)

metric = ndp.brownian_sde.SphericalMetric(2)

sde = ndp.brownian_sde.SphericalBrownian(
    None,
    None,
    None,
    beta_schedule,
    is_score_preconditioned=False,
    std_trick=False,
    residual_trick=False,
)

# %%
x = jnp.linspace(0, 5, 50)[:, None]
print(x.shape)
y_dim = 3
n_samples = 10
subkeys = jax.random.split(key, n_samples)

n = x.shape[0]

yT = jax.vmap(lambda key: sde.sample_prior(key, x))(subkeys)
print(yT.shape)
logp = jax.vmap(lambda y: sde.log_prob_prior(x, y))(yT)
print(logp.shape)

from neural_diffusion_processes.sde import flatten, unflatten
from neural_diffusion_processes.kernels import ProjectionOperator

# From Moser flow paper's codebase

# def P(z, is_detached=False):
#     def projection(v):
#         normal = self.surface.normal(z)
#         if is_detached:
#             normal = normal.detach()
#         return v - torch.bmm(v.view(-1, 1, dim), normal.view(-1, dim, 1)).view(
#             -1, 1
#         ) * normal / (normal**2).sum(dim=1, keepdim=True)

#     return projection


# # if isinstance(self.surface, ImplicitSphere):
# x = x / torch.norm(x, dim=1, keepdim=True)
# Px = P(x)
# return Px(self.v(x))


# P = ProjectionOperator(A)
y = yT[0]
P = sde.projection(y)
v = jax.random.normal(key, (n, y_dim))
u = unflatten(P @ flatten(v), y_dim)
print(u.shape)
# print(u)
# u = jax.vmap(jnp.matmul)(A, v)

# check this is zero
print(jax.vmap(jnp.dot)(u, y).sum())

# check this is zero
# print((jax.vmap(metric.to_tangent)(u, y) - u).sum())
print((metric.to_tangent(u, y) - u).sum())
x = metric.exp(u, y)
w = metric.log(x, y)
print(jax.vmap(jnp.dot)(w, y).sum())

x = metric.exp(w, y)
print((x - metric.exp(w, y)).sum())

# print((jax.vmap(metric.to_tangent)(u, y) - u).sum())

diffusion = sde.diffusion(0.5, y, x)
print(diffusion)

# %%
t = 0.1
network = lambda t, yt, x, key: jnp.zeros_like(yt)
y0 = jnp.array([1.0, 0.0, 0.0])
y0 = jnp.repeat(jnp.repeat(y0[None], n, 0)[None], n_samples, 0)
yt = jax.vmap(lambda y: sde.sample_marginal(key, t, x, y))(y0)

sde.loss_type = "ism"
loss = jax.vmap(lambda y: sde.loss(key, t, y, x, network))(yt)
print("loss", loss.shape)

sde.loss_type = "dsm"
loss = jax.vmap(lambda y: sde.loss(key, t, y, x, network))(yt)
print("loss", loss.shape)

# %%


def get_sphere_coords():
    radius = 1.0
    # set_aspect_equal_3d(ax)
    n = 200
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


def sphere_plot(ax, color="grey"):
    # assert manifold.dim == 2
    x, y, z = get_sphere_coords()
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=0, alpha=0.2)

    return ax


def remove_background(ax):
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax


def plot(traj, size=10, dpi=300, out="out", color="red"):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    # sphere = visualization.Sphere()
    sphere_color = (220 / 255, 220 / 255, 220 / 255)
    ax = sphere_plot(ax, color=sphere_color)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax = remove_background(ax)

    N, K, D = traj.shape
    colours = sns.color_palette("hls", K)
    colours = sns.cubehelix_palette(n_colors=K, reverse=False)
    alpha = np.linspace(0, 1, N + 1)
    # alpha = np.flip(alpha)
    ax.scatter(
        traj[-1, :, 0],
        traj[-1, :, 1],
        traj[-1, :, 2],
        s=100,
        marker="*",
        color=colours[0],
    )
    for k in range(K):
        c = sns.cubehelix_palette(n_colors=N, reverse=False, as_cmap=False)
        for n in range(N - 1):
            ax.plot(
                traj[n : n + 2, k, 0],
                traj[n : n + 2, k, 1],
                traj[n : n + 2, k, 2],
                lw=1.0,
                linestyle="-",
                # alpha = alpha[n],
                alpha=0.9,
                # color=colours[k])
                color=c[n],
            )
    for i in range(1):
        ax.scatter(
            traj[-1, :, 0],
            traj[-1, :, 1],
            traj[-1, :, 2],
            s=100,
            marker="o",
            alpha=1,
            color=colours[-1],
        )
    print("save")
    fig.tight_layout()
    plt.savefig(
        out + ".jpg", dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=-2.1
    )
    plt.close(fig)
    return fig


from neural_diffusion_processes.sde import sde_solve

from typing import Tuple

import jax.tree_util as jtu
from equinox.internal import ω

from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar
from diffrax.local_interpolation import LocalLinearInterpolation
from diffrax.solution import RESULTS
from diffrax.term import AbstractTerm
from diffrax.solver.euler import _SolverState, _ErrorEstimate


class SphericalGRW(dfx.Euler):
    """Euler's method.

    1st order explicit Runge--Kutta method. Does not support adaptive step sizing.

    When used to solve SDEs, converges to the Itô solution.
    """

    term_structure = jtu.tree_structure(0)
    interpolation_cls = LocalLinearInterpolation

    def retraction(self, v, x):
        def projection(point):
            return point / jnp.linalg.norm(point, axis=-1, keepdims=True)

        return flatten(projection(unflatten(x + v, 3)))

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        # y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        # y1 = self.retraction(terms.vf_prod(t0, y0, args, control) ** ω, y0**ω).ω
        y1 = self.retraction(terms.vf_prod(t0, y0, args, control), y0)
        # y1 = y0
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


t = 0.1
y0 = jnp.array([1.0, 0.0, 0.0])
y0 = jnp.repeat(jnp.repeat(y0[None], n, 0)[None], n_samples, 0)

kwargs = {"forward": True, "rtol": None, "num_steps": 10, "solver": SphericalGRW()}
# yt = jax.vmap(
#     lambda y: sde_solve(
#         sde, network, key=key, x=x, y=y, tf=None, prob_flow=True, **kwargs
#     )
# )(y0)
# print(yt.shape)

yt = jax.vmap(lambda y: sde.sample_marginal(key, t, x, y))(y0)
print(yt.shape)

plot(yt[0][None])
