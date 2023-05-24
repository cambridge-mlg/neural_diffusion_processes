import time
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d


def benchmark_loop(loop, num_batches=100):
    times = []
    start_first = time.time()
    _ = next(loop)
    end_first = time.time()
    for _ in range(num_batches):
        start_batch = time.time()
        _ = next(loop)
        end_batch = time.time()
        times.append(end_batch - start_batch)
    times = np.array(times)
    print(f"""
Benchmark
---------
iter/s      : {num_batches / np.sum(times)}
s/iter      : {np.mean(times)}
s/iter (var): {np.var(times)}
time first  : {end_first - start_first}
min         : {np.min(times)}
max         : {np.max(times)}
"""
    )
    return times


key = jax.random.PRNGKey(0)
dataset = "se"
task = "training"
total_num_samples = int(2**14)
batch_size = 2
ds = regression1d.get_dataset(dataset, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=None)


batch: regression1d.DataBatch = next(ds)

# idx = batch.mask[0,:,0].nonzero()[0]

# print(batch.xs.shape)
# print(batch.xs[:,idx,:].shape)


@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask):
    model = ndp.models.attention.BiDimensionalAttentionModel(n_layers=5, hidden_dim=64, num_heads=8, init_zero=False)
    return model(x, y, t, mask)


# @jax.jit
# def net(params, t, yt, x, mask, *, key):
#     print("compiling net:", x.shape)
#     del key  # the network is deterministic
#     #NOTE: Network awkwardly requires a batch dimension for the inputs
#     return network.apply(params, t[None], yt[None], x[None])[0]


t = 1. * jnp.zeros((batch.ys.shape[0]))
params = network.init(key, t=t, y=batch.ys, x=batch.xs, mask=batch.mask)
net = jax.jit(lambda t, y, x, m: network.apply(params, t, y, x, m))


# @jax.jit
def func(batch):
    print("compiling func:", batch.xs.shape)
    t = 0.5 * jnp.ones((batch.ys.shape[0]))
    xs = batch.xs[:, ~batch.mask[0].astype(jnp.bool_)]
    ys = batch.ys[:, ~batch.mask[0].astype(jnp.bool_)]
    o = net(t, ys, xs, None)
    print(xs.shape)
    return jnp.mean(o)


class Loop:
    def __next__(self):
        batch = next(ds)
        func(batch)


benchmark_loop(Loop(), 10)

net.cache
