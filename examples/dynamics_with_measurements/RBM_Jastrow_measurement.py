import jax
import flax.linen as nn
from typing import Union, Any
import numpy as np
import netket.nn as nknn
import jax.numpy as jnp
from netket.utils.types import NNInitFunc
from flax.linen.dtypes import promote_dtype
from functools import partial

default_kernel_init = nn.initializers.normal(stddev=0.01)

tol = 1e-12


@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=(0))
def spwf(sigma, orbital_up, orbital_down, N):
    return 0.5 * (1 + sigma) * orbital_up + 0.5 * (1 - sigma) * orbital_down


class RBMJasMeas(nn.Module):
    r"""A Restricted Boltzmann Machine endowed with the Jastrow term to perform
    exact ZZ gates and the MF-like term to realize projective measurements
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        N = input.shape[-1]
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            out_RBM = x + out_bias
        else:
            out_RBM = x

        theta_zz = self.param(
            "theta_zz", nn.initializers.zeros, (N, N), self.param_dtype
        )
        orbital_up = self.param(
            "orbital_up", nn.initializers.ones, (N,), self.param_dtype
        )
        orbital_down = self.param(
            "orbital_down", nn.initializers.ones, (N,), self.param_dtype
        )

        theta_zz = theta_zz + theta_zz.T
        theta_zz, x_in = promote_dtype(theta_zz, input, dtype=None)
        out_Jas_zz = jnp.einsum("...i,ij,...j", x_in, theta_zz, x_in)

        out_MF = spwf(input, orbital_up, orbital_down, N)
        out_MF = jnp.sum(jnp.log(out_MF), axis=-1)

        return out_RBM + 1j * out_Jas_zz + out_MF
