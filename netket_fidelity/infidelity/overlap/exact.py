from typing import Optional
from functools import partial

import jax.numpy as jnp
import jax

from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.dispatch import TrueT
from netket.utils.numbers import is_scalar
from netket.vqs import ExactState, expect, expect_and_grad
from netket.utils import mpi

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
    target_wf,
    return_grad,
):

    N = sigma_new.shape[-1]
    n_chains_t = sigma_old.shape[-2]

    σ = sigma_new.reshape(-1, N)
    σ_t = sigma_old.reshape(-1, N)

    def expect_fun(params):
        state_fw = jnp.exp(afun({"params": params, **model_state}))
        state_fw = state_fw / jnp.sqrt(jnp.sum(jnp.abs(state_fw) ** 2))
        return jnp.abs(state_fw.T.conj() @ target_wf) ** 2

    if not return_grad:
        F, F_stats = expect_fun(params)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nkjax.vjp(expect_fun, params, has_aux=True, conjugate=True)

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
