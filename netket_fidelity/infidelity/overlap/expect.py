from typing import Optional
from functools import partial

import jax.numpy as jnp
import jax

from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.dispatch import TrueT
from netket.utils.numbers import is_scalar
from netket.vqs import MCState, expect, expect_and_grad
from netket import jax as nkjax
from netket.utils import mpi

from netket_fidelity.utils import expect_2distr

from .operator import InfidelityOperatorStandard


@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperatorStandard):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        vstate.samples,
        op.target.samples,
        op.cv_coeff,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(
    vstate: MCState,
    op: InfidelityOperatorStandard,
    use_covariance: TrueT,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        vstate.samples,
        op.target.samples,
        op.cv_coeff,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad"))
def infidelity_sampling_MCState(
    afun,
    afun_t,
    params,
    params_t,
    model_state,
    model_state_t,
    sigma,
    sigma_t,
    cv_coeff,
    return_grad,
):

    N = sigma.shape[-1]
    n_chains_t = sigma_t.shape[-2]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)

    def expect_kernel(params):
        def kernel_fun(params, params_t, σ, σ_t):
            W = {"params": params, **model_state}
            W_t = {"params": params_t, **model_state_t}

            log_val = afun_t(W_t, σ) + afun(W, σ_t) - afun(W, σ) - afun_t(W_t, σ_t)
            res = jnp.exp(log_val).real
            if cv_coeff is not None:
                res = res + cv_coeff * (jnp.exp(2 * log_val.real) - 1)
            return res

        log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        log_pdf_t = (
            lambda params, σ: 2 * afun_t({"params": params, **model_state_t}, σ).real
        )

        return expect_2distr(
            log_pdf,
            log_pdf_t,
            kernel_fun,
            params,
            params_t,
            σ,
            σ_t,
            n_chains=n_chains_t,
        )

    if not return_grad:
        F, F_stats = expect_kernel(params)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nkjax.vjp(
        expect_kernel, params, has_aux=True, conjugate=True
    )

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
