import jax
import flax

from netket import jax as nkjax


def make_logpsi_U_afun(logpsi_fun, U, variables):
    """Wraps an apply_fun into another one that multiplies it by an
    Unitary transformation U.

    This wrapper is made such that the Unitary is passed as the model_state
    of the new wrapped function, and therefore changes to the angles/coefficients
    of the Unitary should not trigger recompilation.

    Args:
        logpsi_fun: a function that takes as input variables and samples
        U: a {class}`nk.operator.JaxDiscreteOperator`
        variables: The variables used to call *logpsi_fun*

    Returns:
        A tuple, where the first element is a new function with the same signature as
        the original **logpsi_fun** and a set of new variables to be used to call it.
    """
    # wrap apply_fun into logpsi logpsi_U
    logpsiU_fun = nkjax.HashablePartial(_logpsi_U_fun, logpsi_fun)

    # Insert a new 'model_state' key to store the Unitary. This only works
    # if U is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(variables, {"unitary": U})

    return logpsiU_fun, new_variables


def _logpsi_U_fun(apply_fun, variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `unitary` with
    a jax-compatible operator.
    """
    variables_applyfun, U = flax.core.pop(variables, "unitary")

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
