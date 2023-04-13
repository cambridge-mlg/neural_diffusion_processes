import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

import gpjax
import jaxkern

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d

DATASET = "se"


def get_dataset(key):
    task = "training"
    total_num_samples = int(2**14)
    batch_size = 2
    return regression1d.get_dataset(DATASET, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=None)


key = jax.random.PRNGKey(1)
beta = ndp.sde.LinearBetaSchedule()
limiting_kernel = jaxkern.White()
hyps = {
    "mean_function": {},
    "kernel": limiting_kernel.init_params(None),
}
sde = ndp.sde.SDE(
    limiting_kernel,
    gpjax.mean_functions.Zero(),
    hyps,
    beta,
    is_score_preconditioned=False,
    std_trick=False,
    residual_trick=False,
    exact_score=True,
)

factory = regression1d._DATASET_FACTORIES[DATASET] 
mean0, kernel0, params0 = factory.mean, factory.kernel, factory.params
net = sde.get_exact_score(mean0, kernel0, params0)

key, dkey = jax.random.split(key)
# dataset = get_dataset(dkey)
# batch = next(dataset)
x = jnp.linspace(-2, 2, 60)[:, None]
mask1 = jnp.concatenate([
    jnp.zeros((50,)),
    jnp.ones((10,))
])
mask2 = jnp.concatenate([
    jnp.zeros((60,)),
    jnp.ones((0,))
])
mask3 = jnp.concatenate([
    jnp.ones((10,)),
    jnp.zeros((2,)),
    jnp.ones((10,)),
    jnp.zeros((3,)),
    jnp.ones((35,)),
])


x_corrupted = jnp.where(mask1[:,None] == 1., -666., x)
o1 = ndp.sde.sde_solve(
    sde, net, x_corrupted, mask1, key=key
)[0]
o2 = ndp.sde.sde_solve(
    sde, net, x, mask2, key=key
)[0]
o3 = ndp.sde.sde_solve(
    sde, net, x, mask3, key=key
)[0]

import matplotlib.pyplot as plt
plt.plot(x, o1, "x")
plt.plot(x, o2, 'o')
plt.plot(x, o3, 's')
plt.ylim(-5, 5)
plt.savefig("sample.png")

# import numpy as np
# np.testing.assert_array_almost_equal(o1[:50], o2)

