from functools import partial

import jax.numpy as jnp
import jax

from netket import jax as nkjax
from netket.utils.dispatch import TrueT
from netket.vqs import ExactState, expect, expect_and_grad
from netket.utils import mpi
from netket.stats import Stats

from .operator import InfidelityOperatorStandard


@expect.dispatch
def infidelity(vstate: ExactState, op: InfidelityOperatorStandard):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op.target, ExactState):
        raise TypeError("Can only compute infidelity of exact states.")

    return infidelity_sampling_ExactState(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        op.target.to_array(),
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(
    vstate: ExactState,
    op: InfidelityOperatorStandard,
    use_covariance: TrueT,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op.target, ExactState):
        raise TypeError("Can only compute infidelity of exact states.")

    return infidelity_sampling_ExactState(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        op.target.to_array(),
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "return_grad"))
def infidelity_sampling_ExactState(
    afun,
    params,
    model_state,
    sigma,
    state_t,
    return_grad,
):
    def expect_fun(params):
        state = jnp.exp(afun({"params": params, **model_state}, sigma))
        state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))
        return jnp.abs(state.T.conj() @ state_t) ** 2

    if not return_grad:
        F = expect_fun(params)
        return Stats(mean=1 - F, error_of_mean=0.0, variance=0.0)

    F, F_vjp_fun = nkjax.vjp(expect_fun, params, conjugate=True)

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = Stats(mean=1 - F, error_of_mean=0.0, variance=0.0)

    return I_stats, I_grad
