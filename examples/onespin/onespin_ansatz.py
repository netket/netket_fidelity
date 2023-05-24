import flax.linen as nn
from functools import partial
import jax
import jax.numpy as jnp

default_kernel_init = nn.initializers.normal(stddev=0.01)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0))
def spwf(sigma, theta, phi):
    return 0.5 * (1 - sigma) * jnp.cos(theta) + 0.5 * (1 + sigma) * jnp.exp(
        1j * phi
    ) * jnp.sin(theta)


class BlochSphere_1spin(nn.Module):
    @nn.compact
    def __call__(self, input):
        N = input.shape[-1]

        theta = self.param("theta", nn.initializers.normal(), (N,), float)
        phi = self.param("phi", nn.initializers.normal(), (N,), float)

        out_MF = spwf(input, theta, phi)

        # sum the output
        out_MF = jnp.log(out_MF)

        return out_MF.reshape(
            -1,
        )
