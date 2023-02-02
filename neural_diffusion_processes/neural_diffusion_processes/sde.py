from functools import partial
import dataclasses
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax as dfx
import numpy as np

from check_shapes import check_shapes


@dataclasses.dataclass
class LinearBetaSchedule:
    t0: float = 1e-5
    t1: float = 1.0
    beta0: float = 0.0
    beta1: float = 20.0

    def __call__(self, t):
        interval = self.t1 - self.t0
        normed_t = (t - self.t0) / interval
        return self.beta0 + normed_t * (self.beta1 - self.beta0)
    
    def B(self, t):
        r"""
        integrates \int_{s=0}^t beta(s) ds
        """
        interval = self.t1 - self.t0
        normed_t = (t - self.t0) / interval
        # TODO: Notice the additional scaling by the interval t1-t0.
        # This is not done in the package.
        return interval * (
            self.beta0 * normed_t + 0.5 * (normed_t ** 2) * (self.beta1 - self.beta0)
        )
