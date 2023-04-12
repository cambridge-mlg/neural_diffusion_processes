from typing import Optional

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d


def benchmark_dataloader(data_iter, num_batches=100):
    times = []
    _ = next(data_iter)
    for _ in range(num_batches):
        start_batch = time.time()
        batch = next(data_iter)
        end_batch = time.time()
        times.append(end_batch - start_batch)
    times = np.array(times)
    print(f"""
Benchmark
---------
iter/s      : {num_batches / np.sum(times)}
s/iter      : {np.mean(times)}
s/iter (var): {np.var(times)}
min         : {np.min(times)}
max         : {np.max(times)}
"""
    )
    return times
    




if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    dataset = "se"
    task = "training"
    total_num_samples = int(2**14)
    batch_size = 2
    ds = regression1d.get_dataset(dataset, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=1)
    print("JAX")
    benchmark_dataloader(ds)

    # while True:
    # for batch in ds:
    #     print('.', end='')
        # batch = next(ds)

    # @jax.jit

    # output_signature = (
    #     tf.TensorSpec(shape=(batch_size, None, 1), dtype=tf.float64)
    #     # tf.TensorSpec(shape=(batch_size, None, 1), dtype=tf.float64),
    # )
    # spec = tf.TensorSpec(shape=[None, None, None], dtype=tf.float64) 
    # ds = tf.data.Dataset.from_generator(
    #     lambda: data_generator(key, dataset, task, total_num_samples, batch_size),
    #     # output_types=tf.float64
    #     output_signature=spec,
    # )
    # options = tf.data.Options()
    # options.autotune.enabled = True
    # ds = ds.with_options(options)
    # ds = ds.as_numpy_iterator()

    # print("TFDS")
    # benchmark_dataloader(ds)
