from functools import partial
import jax.numpy as jnp
import jax
from netket import jax as nkjax
from netket.utils import mpi
from netket.utils.dispatch import TrueT

from netket.vqs import MCState

import netket as nk

from netket_fidelity.utils.expect import expect_onedistr

from .operator import L2L1Operator


@nk.vqs.expect.dispatch
def L2L1(vstate_new: MCState, op: L2L1Operator):
    if op.hilbert != vstate_new.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op._state_diff, MCState):
        raise TypeError("Can only compute L2L1 of MC states.")
    
    vstate_diff = op._state_diff

    sigma = vstate_diff.samples

    return L2L1_sampling_MCState(
        vstate_diff._apply_fun,
        vstate_diff.parameters,
        vstate_diff.model_state,
        sigma,
        return_grad=False,
    )


@nk.vqs.expect_and_grad.dispatch
def L2L1(
    vstate_new: MCState,
    op: L2L1Operator,
    use_covariance: TrueT,
    *,
    mutable,
):
    if op.hilbert != vstate_new.hilbert:
        raise TypeError("Hilbert spaces should match")
    if not isinstance(op._state_diff, MCState):
        raise TypeError("Can only compute L2L1 of MC states.")
    
    vstate_diff = op._state_diff

    sigma = vstate_diff.samples

    return L2L1_sampling_MCState(
        vstate_diff._apply_fun,
        vstate_diff.parameters,
        vstate_diff.model_state,
        sigma,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("return_grad"))
def L2L1_sampling_MCState(
    afun_diff,
    params_diff,
    model_state_diff,
    sigma,
    return_grad,
):

    N = sigma.shape[-1]
    n_chains = sigma.shape[-2]

    σ = sigma.reshape(-1, N)

    def expect_kernel(params):
        def kernel_fun(params, σ):

            W_diff = {"params": params, **model_state_diff}

            num = jnp.exp(afun_diff(W_diff, σ)) ** 2

            return num

        log_pdf = (
            lambda params, σ: 2
            * afun_diff({"params": params, **model_state_diff}, σ).real
        )

        return expect_onedistr(
            log_pdf,
            kernel_fun,
            params,
            σ,
            n_chains=n_chains,
        )

    if not return_grad:
        L, L_stats = expect_kernel(params_diff)
        return L_stats

    L, L_vjp_fun, L_stats = nkjax.vjp(
        expect_kernel, params_diff, has_aux=True, conjugate=True
    )

    L_grad = L_vjp_fun(jnp.ones_like(L))[0]
    L_grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], L_grad)

    return L_stats, L_grad
