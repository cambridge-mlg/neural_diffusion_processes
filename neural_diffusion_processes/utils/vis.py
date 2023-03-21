import seaborn as sns
import matplotlib

import jax.numpy as jnp

norm = matplotlib.colors.Normalize()
cm = sns.cubehelix_palette(start=0.5, rot=-0.75, as_cmap=True, reverse=True)


def plot_vector_field(ax, x, y, scale=50, width=0.005, color=None):
    y_norm = jnp.linalg.norm(y, axis=-1)
    color = cm(norm(y_norm)) if color is None else color
    ax.quiver(
        x[:, 0],
        x[:, 1],
        y[:, 0],
        y[:, 1],
        color=color,
        scale=scale,
        width=width,
    )
    ax.set_aspect('equal')
