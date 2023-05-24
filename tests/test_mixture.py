import jax
import jax.numpy as jnp
    

def fun(key, generators, *operands):
    if len(generators) == 1:
        return lambda: generators[0](*operands)

    key, subkey = jax.random.split(key)
    rand = jax.random.uniform(subkey, shape=())
    return lambda: jax.lax.cond(
        rand < 1./len(generators),
        generators[0],
        fun(key, generators[1:]),
        *operands
    )


@jax.jit
def sample(key):
    generators = [
        lambda: 1.0,
        lambda: 2.0,
        lambda: 3.0,
        lambda: 4.0,
    ]
    return fun(key, generators)()

    key, skey = jax.random.split(key)
    rand1, rand2, rand3 = jax.random.uniform(skey, shape=(3,))
    return jax.lax.cond(
        rand1 < 0.25,
        generators[0],
        lambda: jax.lax.cond(
            rand2 < 1./3,
            generators[1],
            lambda: jax.lax.cond(
                rand3 < 0.5,
                generators[2],
                generators[3],
            )
        )
    )

def sample2(key):
    generators = [
        lambda: 1.0,
        lambda: 2.0,
        lambda: 3.0,
        lambda: 4.0,
    ]
    i = jax.random.randint(key, shape=(1,), minval=0, maxval=len(generators))
    return generators[i]()

import matplotlib.pyplot as plt
import time

key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)

t0 = time.time()
samples1 = jax.vmap(sample)(jax.random.split(subkey, 10_000))
print(time.time() - t0)

key, subkey = jax.random.split(key)
t0 = time.time()
samples2 = jax.vmap(sample)(jax.random.split(subkey, 10_000))
print(time.time() - t0)

plt.hist([samples1, samples2], bins=4)
plt.savefig("hist.png")
