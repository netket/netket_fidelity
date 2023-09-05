import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


class GHZ(nn.Module):
    @partial(jax.vmap, in_axes=(None, 0), out_axes=(0))
    def f(self, x):
        return jax.lax.cond(x == 1, lambda x: 0.0, lambda x: -10000.0, x)

    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)

        N = x.shape[-1]

        out = jnp.absolute(jnp.sum(x.reshape((-1, N)), axis=-1) / N) + lam * 0

        return self.f(out)
