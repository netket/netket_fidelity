import jax

import flax


def _logpsi_U(apply_fun, variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `unitary` with
    a jax-compatible operator.
    """
    variables_applyfun, U = flax.core.pop(variables, 'unitary')

    xp, mels = U.get_conn_padded(x)
    logpsi_xp = apply_fun(variables_applyfun, xp, *args)

    return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

#
# from flax import linen as nn
#
# class LogUPsi(nn.Module):
#     """
#     A Flax module working a bit like logpsi_U above.
#
#     Just set a vs.model_state = {'unitary':{'operator':your_jax_operator}}
#     """
#     log_psi : nn.Module
#
#     @nn.compact
#     def __call__(self, x, *args):
#         U = self.variable('unitary', 'operator', lambda : None)
#         if U.value is not None:
#             xp, mels = U.value.get_conn_padded(x)
#             logpsi_xp = self.log_psi(xp, *args)
#             return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)
#         else:
#             return self.log_psi(x, *args)
#
