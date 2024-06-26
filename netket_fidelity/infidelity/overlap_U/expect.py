from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from netket import jax as nkjax
from netket.operator import DiscreteJaxOperator
from netket.vqs import MCState, expect, expect_and_grad, get_local_kernel_arguments
from netket.utils import mpi


from .operator import InfidelityOperatorUPsi


@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperatorUPsi, chunk_size: None):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
        op.cv_coeff,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: MCState,
    op: InfidelityOperatorUPsi,
    chunk_size: None,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    sigma, args = get_local_kernel_arguments(vstate, op._U)
    sigma_t, args_t = get_local_kernel_arguments(op.target, op._U_dagger)

    return infidelity_sampling_MCState(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma,
        args,
        sigma_t,
        args_t,
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
    args,
    sigma_t,
    args_t,
    cv_coeff,
    return_grad,
):
    N = sigma.shape[-1]
    n_chains_t = sigma_t.shape[-2]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)

    if isinstance(args, DiscreteJaxOperator):
        xp, mels = args.get_conn_padded(σ)
        xp_t, mels_t = args_t.get_conn_padded(σ_t)
    else:
        xp = args[0].reshape(σ.shape[0], -1, N)
        mels = args[1].reshape(σ.shape[0], -1)
        xp_t = args_t[0].reshape(σ_t.shape[0], -1, N)
        mels_t = args_t[1].reshape(σ_t.shape[0], -1)

    def expect_kernel(params):
        def kernel_fun(params_all, samples_all):
            params, params_t = params_all
            σ, σ_t = samples_all

            W = {"params": params, **model_state}
            W_t = {"params": params_t, **model_state_t}

            logpsi_t_xp = afun_t(W_t, xp)
            logpsi_xp_t = afun(W, xp_t)

            log_val = (
                logsumexp(logpsi_t_xp, axis=-1, b=mels)
                + logsumexp(logpsi_xp_t, axis=-1, b=mels_t)
                - afun(W, σ)
                - afun_t(W_t, σ_t)
            )
            res = jnp.exp(log_val).real
            if cv_coeff is not None:
                res = res + cv_coeff * (jnp.exp(2 * log_val.real) - 1)
            return res

        log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        log_pdf_t = (
            lambda params, σ: 2 * afun_t({"params": params, **model_state_t}, σ).real
        )

        def log_pdf_joint(params_all, samples_all):
            params, params_t = params_all
            σ, σ_t = samples_all
            log_pdf_vals = log_pdf(params, σ)
            log_pdf_t_vals = log_pdf_t(params_t, σ_t)
            return log_pdf_vals + log_pdf_t_vals

        return nkjax.expect(
            log_pdf_joint,
            kernel_fun,
            (
                params,
                params_t,
            ),
            (
                σ,
                σ_t,
            ),
            n_chains=n_chains_t,
        )

    if not return_grad:
        F, F_stats = expect_kernel(params)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nkjax.vjp(
        expect_kernel, params, has_aux=True, conjugate=True
    )

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)
    I_grad = jax.tree_util.tree_map(lambda x: -x, F_grad)
    I_stats = F_stats.replace(mean=1 - F)

    return I_stats, I_grad
