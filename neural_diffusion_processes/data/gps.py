from __future__ import annotations
from typing import Tuple, Iterator, Optional, Mapping

import jax
import jax.numpy as jnp
import jaxkern
from einops import rearrange

from .misc import sample_mvn, flatten, unflatten
from .kernels import sample_prior_gp, prior_gp, posterior_gp


def grid_2d(min_x, max_x, n_xaxis, min_y=None, max_y=None, n_yaxis=None, flatten=True):
    """
    Input:
        min_x,max_x,min_y,max_y: float - range of x-axis/y-axis
        n_x_axis,n_y_axis: int - number of points per axis
        flatten: Boolean - determines shape of output
    Output:
        torch.tensor - if flatten is True: shape (n_y_axis*n_x_axis,2)
                                          (element i*n_x_axis+j gives i-th element in y-grid
                                           and j-th element in  x-grid.
                                           In other words: x is periodic counter and y the stable counter)
                       if flatten is not True: shape (n_y_axis,n_x_axis,2)
    """
    if min_y is None:
        min_y = min_x
    if max_y is None:
        max_y = max_x
    if n_yaxis is None:
        n_yaxis = n_xaxis

    x = jnp.linspace(min_x, max_x, n_xaxis)
    y = jnp.linspace(min_y, max_y, n_yaxis)
    # TODO: reorder the grid - has repercussions later on
    Y, X = jnp.meshgrid(y, x)
    grid = jnp.stack((X, Y), 2)

    if flatten:
        grid = rearrange(grid, "x y d -> (x y) d")

    return grid


# def get_2d_grid(num, min_=-1, max_=1):
#     x = jnp.linspace(min_, max_, num)
#     x1, x2 = jnp.meshgrid(x, x)
#     x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
#     return x


def radial_grid_2d(max_r, n_axis):
    """
    Input:
        min_r: float - maximum radius from origin
        n_axis: float - number of points across the x diameter
    Output:
        torch.tensor - if flatten is True: shape (n_y_axis*n_x_axis,2)
    """
    grid = grid_2d(min_x=-max_r, max_x=max_r, n_xaxis=n_axis, flatten=True)

    in_radius_indices = jnp.linalg.norm(grid, axis=-1) <= max_r

    grid = grid[in_radius_indices]

    return grid


def get_vec_gp_data(
    key,
    kernel: jaxkern.base.AbstractKernel,
    mean_function,
    num_samples: int,
    *,
    x_radius: float,
    num_points: int,
    obs_noise: float,
    input_dim: int = 2,
    output_dim: int = 2,
    params: Optional[Mapping[str, float]] = None,
    **kwargs,
):
    # TODO: merge with get_gp_data function
    """
    Returns tuple of inputs and outputs. The outputs are drawn from a GP prior with a fixed kernel.
    """
    assert input_dim == 2
    assert output_dim == 2

    x = radial_grid_2d(x_radius, num_points)
    y = sample_prior_gp(
        key,
        mean_function,
        kernel, 
        params,
        x,
        num_samples=num_samples,
        obs_noise=obs_noise,
    )
    x = jnp.repeat(x[None, ...], y.shape[0], 0)
    return x, y

def get_vec_gp_log_prob(
    kernel: jaxkern.base.AbstractKernel,
    mean_function,
    obs_noise: float,
    params: Optional[Mapping[str, float]] = None,
    **kwargs,
    ):
    return lambda xc, yc, xs, ys: posterior_gp(mean_function, kernel, params, xc, yc, obs_noise)(xs).log_prob(flatten(ys))

def get_gp_data(
    key,
    kernel: jaxkern.base.AbstractKernel,
    num_samples: int,
    *,
    x_range=(-1.0, 1.0),
    num_points: int = 100,
    input_dim: int = 1,
    output_dim: int = 1,
    params: Optional[Mapping[str, float]] = None,
):
    """
    Returns tuple of inputs and outputs. The outputs are drawn from a GP prior with a fixed kernel.
    """
    assert input_dim == 1
    assert output_dim == 1

    if params is None:
        params = {
            "lengthscale": 0.2,
            "variance": 1.0,
        }

    def sample_single(key):
        input_key, output_key = jax.random.split(key, 2)
        x = jax.random.uniform(
            input_key,
            [num_points, 1],
            minval=x_range[0],
            maxval=x_range[1],
            dtype=jnp.float64,
        )
        x = x.sort(axis=0)
        gram = kernel.gram(params, x).to_dense()
        y = sample_mvn(output_key, jnp.zeros_like(x), gram)
        return x, y

    x, y = jax.vmap(sample_single)(jax.random.split(key, num_samples))
    return x, y
