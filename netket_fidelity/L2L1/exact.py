from functools import partial

import jax.numpy as jnp
import jax

from netket import jax as nkjax
from netket.utils.dispatch import TrueT
from netket.vqs import ExactState, expect, expect_and_grad
from netket.utils import mpi
from netket.stats import Stats

from .operator import L2L1Operator


@expect.dispatch
def L2L1(vstate_new: ExactState, op: L2L1Operator):
    if op.hilbert != vstate_new.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op._state_diff, ExactState):
        raise TypeError("Can only compute L2L1 of exact states.")

    vstate_diff = op._state_diff

    return L2L1_sampling_ExactState(
        vstate_diff._apply_fun,
        vstate_diff.parameters,
        vstate_diff.model_state,
        vstate_diff._all_states,
        return_grad=False,
    )


@expect_and_grad.dispatch
def L2L1(
    vstate_new: ExactState,
    op: L2L1Operator,
    use_covariance: TrueT,
    *,
    mutable,
):
    if op.hilbert != vstate_new.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op._state_diff, ExactState):
        raise TypeError("Can only compute L2L1 of exact states.")

    vstate_diff = op._state_diff

    return L2L1_sampling_ExactState(
        vstate_diff._apply_fun,
        vstate_diff.parameters,
        vstate_diff.model_state,
        vstate_diff._all_states,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("return_grad"))
def L2L1_sampling_ExactState(
    afun_diff,
    params_diff,
    model_state_diff,
    sigma,
    return_grad,
):
    def expect_fun(params):
        L2 = jnp.sum(
            jnp.absolute(
                jnp.exp(afun_diff({"params": params, **model_state_diff}, sigma))
            )
            ** 4
        )
        L1 = jnp.sum(
            jnp.absolute(
                jnp.exp(afun_diff({"params": params, **model_state_diff}, sigma))
            )
            ** 2
        )

        return L2 / L1

    if not return_grad:
        L = expect_fun(params_diff)
        return Stats(mean=L, error_of_mean=0.0, variance=0.0)

    L, L_vjp_fun = nkjax.vjp(expect_fun, params_diff, conjugate=True)

    L_grad = L_vjp_fun(jnp.ones_like(L))[0]
    L_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], L_grad)
    L_stats = Stats(mean=L, error_of_mean=0.0, variance=0.0)

    return L_stats, L_grad
