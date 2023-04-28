import os

import jax.numpy as jnp
import pandas as pd

from check_shapes import check_shapes

TWOPI = 2 * jnp.pi
RADDEG = TWOPI / 360
LONOFFSET = 180
LONSTART = 0
LONSTOP = LONSTART + 360


# @check_shapes("return[0]: [B , N, 1]", "return[1]: [B, N, y_dim]")
def storm_data(
    data_dir: str,
    max_len=50,
    max_data_points=-1,
    basin="all",
    limit=False,
    normalise=True,
    **kwargs,
):
    data = pd.read_csv(
        os.path.join(data_dir, "storm", f"{basin}_processed.csv"), header=[0, 1]
    )
    lats = jnp.array(data["LAT"].to_numpy(), dtype=jnp.float32)[:, :max_len]
    lons = (
        jnp.array(data["LON"].to_numpy(), dtype=jnp.float32)[:, :max_len]
        + LONOFFSET
        + 180
    ) % (
        2 * 180
    ) - 180  # center the data over pacific - better center + plotting
    nan_index = jnp.isnan(lats) & jnp.isnan(lons)
    full_data = jnp.sum(nan_index[:, :max_len], axis=-1) == 0
    latlons = jnp.stack((lats, lons), axis=-1)[full_data] * RADDEG
    times = jnp.array(
        (
            pd.to_timedelta(data["LAT"].columns).map(
                lambda x: x.total_seconds() / (60**2 * 24)
            )[:max_len]
        ).to_numpy(),
        dtype=jnp.float32,
    )
    times = jnp.repeat(times[None, :, None], repeats=latlons.shape[0], axis=0)

    if limit:
        keep = jnp.all((latlons[..., 1] < 0) & (latlons[..., 1] > -100), axis=-1)
        latlons = latlons[keep]
        times = times[keep]

    if normalise:
        mean = jnp.mean(latlons, axis=(0, 1))
        std = jnp.std(latlons, axis=(0, 1))
        latlons = latlons - mean
        latlons = latlons / std
    else:
        mean = jnp.zeros_like(latlons[0, 0])
        std = jnp.ones_like(latlons[0, 0])

    if max_data_points > 0 and max_data_points < latlons.shape[0]:
        offset = 35
        latlons = latlons[offset : (offset + max_data_points)]
        times = times[offset : (offset + max_data_points)]

        latlons = jnp.repeat(latlons, 100, axis=0)
        times = jnp.repeat(times, 100, axis=0)

    return (times, latlons), (mean, std)


def proj_3d(x, reverse=False):
    if reverse:
        return jnp.stack(
            (
                jnp.arccos(x[..., 2]) - jnp.pi / 2,
                jnp.sign(x[..., 1])
                * jnp.arccos(x[..., 0] / jnp.sqrt(1 - x[..., 2] ** 2)),
            ),
            axis=-1,
        )
    else:
        lat = x[..., 0]
        lon = x[..., 1]
        return jnp.stack(
            (
                jnp.sin(lat + jnp.pi / 2) * jnp.cos(lon),
                jnp.sin(lat + jnp.pi / 2) * jnp.sin(lon),
                jnp.cos(lat + jnp.pi / 2),
            ),
            axis=-1,
        )


def proj_stereo(x, reverse=False):
    if reverse:
        norm = x[..., 0] ** 2 + x[..., 1] ** 2
        denom = 1 + norm
        x = jnp.stack(
            (2 * x[..., 0] / denom, 2 * x[..., 1] / denom, (1 - norm) / denom), axis=-1
        )
        return proj_3d(x, reverse=True)
    else:
        x = proj_3d(x)
        return x[..., :2] / (1 + x[..., 2:3])


def proj_none(x, reverse=False):
    return x
