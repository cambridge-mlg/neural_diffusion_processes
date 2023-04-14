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
x = jnp.linspace(-2, 2, 100)[:, None]
x_context = jnp.linspace(-2, 2, 5)[:, None] + 1e-4
y_context = jnp.sin(x_context)

test_mask = jnp.concatenate([
    jnp.zeros((len(x) - 15,)),
    jnp.zeros((15,))
])
context_mask = jnp.concatenate([
    jnp.zeros((1,)),
    jnp.ones((1,)),
    jnp.zeros((1,)),
    jnp.ones((1,)),
    jnp.zeros((1,)),
])

# x_corrupted = jnp.where(mask1[:,None] == 1., -666., x)


y_preds = jax.vmap(lambda key: ndp.sde.conditional_sample2(
    sde, net, x_context, y_context, x,
    mask_context=context_mask,
    mask_test=test_mask,
    key=key,
    num_steps=500,
    num_inner_steps=10,
))(jax.random.split(key, 30))

import matplotlib.pyplot as plt
plt.plot(x[~test_mask.astype(jnp.bool_)], y_preds[:, ~test_mask.astype(jnp.bool_), 0].T, "C0", alpha=.3)
plt.plot(x_context[~context_mask.astype(jnp.bool_)], y_context[~context_mask.astype(jnp.bool_)], "kx")
plt.plot(x_context[context_mask.astype(jnp.bool_)], y_context[context_mask.astype(jnp.bool_)], "ko", mfc='none')
# plt.ylim(-5, 5)
plt.savefig("cond.png")

# import numpy as np
# np.testing.assert_array_almost_equal(o1[:50], o2)

