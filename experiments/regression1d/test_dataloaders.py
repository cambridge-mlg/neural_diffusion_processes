from typing import Optional

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d

import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetFromGenerator:
    def __init__(self, generator):
        self._generator  = generator
        self._preprocess = []
    
    def map(self, function):
        self._preprocess.append(function)
    
    def __next__(self):
        batch = next(self._generator)
        for func in self._preprocess:
            batch = func(batch)
        return batch


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
    


def data_generator(key, dataset, task, total_num_samples, batch_size, num_epochs: Optional[int] = None):
    print("generator")
    assert total_num_samples % batch_size == 0

    @jax.jit
    def batch(key) -> ndp.data.DataBatch:
        data = regression1d.get_batch(key, batch_size, dataset, task)
        return data.xs

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint32)])
    def tf_batch(key):
        return tf.numpy_function(batch, [key], tf.float64, stateful=False)

    print("run once...")
    _ = batch(key)

    if num_epochs is None:
        num_epochs = np.inf
    
    count_epochs = 0
    while count_epochs < num_epochs:
        count_epochs += 1
        for _ in range(total_num_samples // batch_size):
            key, bkey = jax.random.split(key)
            yield tf_batch(bkey)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    dataset = "se"
    task = "training"
    total_num_samples = int(2**14)
    batch_size = 32
    jax_iter = data_generator(key, dataset, task, total_num_samples, batch_size)
    print("JAX")
    benchmark_dataloader(jax_iter)
    # exit(0)

    output_signature = (
        tf.TensorSpec(shape=(batch_size, None, 1), dtype=tf.float64)
        # tf.TensorSpec(shape=(batch_size, None, 1), dtype=tf.float64),
    )
    spec = tf.TensorSpec(shape=[None, None, None], dtype=tf.float64) 
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(key, dataset, task, total_num_samples, batch_size),
        # output_types=tf.float64
        output_signature=spec,
    )
    options = tf.data.Options()
    options.autotune.enabled = True
    ds = ds.with_options(options)
    ds = ds.as_numpy_iterator()

    print("TFDS")
    benchmark_dataloader(ds)
