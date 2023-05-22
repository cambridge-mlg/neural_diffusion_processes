import math

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib
from matplotlib.collections import EllipseCollection


import jax
import jax.numpy as jnp

norm = matplotlib.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)

def plot_scalar_field(
    X,
    Y,
    ax=None,
    colormap="viridis",
    zorder=1,
    n_axis=50,
    levels=8,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    raise NotImplementedError()
    X_1, X_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), num=n_axis),
        np.linspace(X[:, 1].min(), X[:, 1].max(), num=n_axis),
    )

    triang = tri.Triangulation(X[:, 0], X[:, 1])
    interpolator = tri.LinearTriInterpolator(triang, Y)
    Z = interpolator(X_1, X_2)

    ax.contourf(X_1, X_2, Z, cmap=colormap, zorder=zorder, levels=levels)

    return ax


def plot_vector_field(
    X, Y, ax=None, color=None, scale=15, width=None, label=None, zorder=1, cmap=None
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    cmap = cmap if cmap is not None else cm

    Y_norm = jnp.linalg.norm(Y, axis=-1)
    color = cmap(norm(Y_norm)) if color is None else color
    ax.quiver(
        X[:, 0],
        X[:, 1],
        Y[:, 0],
        Y[:, 1],
        color=color,
        scale=scale,
        width=width,
        label=label,
        zorder=zorder,
        pivot="mid",
    )
    return ax


def plot_covariances(
    X,
    covariances,
    ax=None,
    alpha=0.5,
    color="C0",
    edgecolor="k",
    scale=0.8,
    label=None,
    zorder=0,
):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
        x_lim = None
        y_lim = None
    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
    # ax.set_aspect('equal')

    def f(A):
        # cf https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipsis-in-a-scatterplot-using-matplotlib
        if len(A.shape) == 1:
            A = jnp.diag(A)

        lambda_, v = jnp.linalg.eigh(A)
        # u = v[:, 0]

        # angle = 360 * jnp.arctan2(u[1] / u[0]) / (2 * math.pi)
        idx = jnp.argmax(abs(lambda_))
        angle = jnp.rad2deg(jnp.arctan2(*v[:, idx][::-1]))
        width = jnp.sqrt(lambda_[idx]) * 2 * scale
        height = jnp.sqrt(lambda_[1-idx]) * 2 * scale

        # if (v[:, 0] < 0).sum() > 0:
        #     print("Error: Ill conditioned covariance in plot. Skipping")
        #     continue

        # Get the width and height of the ellipses (eigenvalues of A):
        # D = jnp.sqrt(v[:, 0])
        # return angle, D
        return angle, width, height

    # angle, D = jax.vmap(f)(covariances)
    angle, width, height = jax.vmap(f)(covariances)

    E = EllipseCollection(
        # widths=scale * D[:, 0],
        # heights=scale * D[:, 1],
        widths=width,
        heights=height,
        angles=angle,
        units="x",
        offsets=X,
        offset_transform=ax.transData,
        color=color,
        linewidth=1,
        alpha=alpha,
        # edgecolor=edgecolor,
        # facecolor="none",
        zorder=zorder,
    )
    ax.add_collection(E)
    # ax.autoscale_view(True)

    if label is not None:
        label_ellipse = Ellipse(
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
            xy=0,
            width=1,
            height=1,
        )
        ax.add_patch(label_ellipse)

    return ax