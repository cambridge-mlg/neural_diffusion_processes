import os

import jax.numpy as jnp
import pandas as pd


TWOPI = 2 * jnp.pi
RADDEG = TWOPI/360

def storm_data(
            data_dir,
            max_len=50,
            **kwargs,
    ):
        data = pd.read_csv(os.path.join(data_dir, "storm", "all_processes.csv"), header=[0,1])
        lats = jnp.array(data['LAT'].to_numpy(), dtype=jnp.float32)[:, :max_len]
        lons = jnp.array(data['LON'].to_numpy(), dtype=jnp.float32)[:, :max_len]
        nan_index = jnp.isnan(lats) & jnp.isnan(lons)
        full_data = jnp.sum(nan_index[:, :max_len], axis=-1)==0
        latlons = jnp.stack(
            (lats, lons), axis=-1
        )[full_data] * RADDEG
        times = pd.to_timedelta(data['LAT'].columns).map(lambda x: x.total_seconds() / (60**2 * 24))[:max_len]
        return times, latlons

def proj_3d(x, reverse=False):
    if reverse:
        return jnp.stack(
            (
                jnp.arccos(x[..., 2]) - jnp.pi/2,
                jnp.sign(x[..., 1])*jnp.arccos(x[..., 0] / jnp.sqrt(1-x[..., 2]**2))
            ), axis=-1
        )
    else:
        lat = x[..., 0]
        lon = x[..., 1]
        return jnp.stack(
            (
                jnp.sin(lat+jnp.pi/2) * jnp.cos(lon),
                jnp.sin(lat+jnp.pi/2) * jnp.sin(lon),
                jnp.cos(lat+jnp.pi/2),
            ), axis=-1
        )

def proj_stereo(x, reverse=False):
    if reverse:
        norm = x[..., 0]**2 + x[..., 1]**2
        denom = 1 + norm
        x = jnp.stack(
            (
                2 * x[..., 0] / denom,
                2 * x[..., 1] / denom,
                (1 - norm) / denom
            ), axis=-1
        )
        return proj_3d(x, reverse=True)
    else:
        x = proj_3d(x)
        return x[..., :2] / (1+x[..., 2:3])
    
def proj_none(x, reverse=False):
    return x