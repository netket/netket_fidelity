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

from ._operator import InfidelityOperatorUPsi


@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperator):
    sigma_new, args_new = nk.vqs.get_local_kernel_arguments(vstate, op._U)
    sigma_old, args_old_dagger = nk.vqs.get_local_kernel_arguments(
        op.target, op._U_dagger
    )

    return infidelity_sampling_old(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma_new,
        args_new,
        sigma_old,
        args_old_dagger,
        op.cv_coeff,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(
    vstate: MCState,
    op: InfidelityOperator,
    use_covariance: TrueT,
    *,
    mutable,
):
    sigma_new, args_new = nk.vqs.get_local_kernel_arguments(vstate, op._U)
    sigma_old, args_old_dagger = nk.vqs.get_local_kernel_arguments(
        op.target, op._U_dagger
    )

    return infidelity_sampling_old(
        vstate._apply_fun,
        op.target._apply_fun,
        vstate.parameters,
        op.target.parameters,
        vstate.model_state,
        op.target.model_state,
        sigma_new,
        args_new,
        sigma_old,
        args_old_dagger,
        op.cv_coeff,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad"))
def infidelity_sampling_old(
    afun,
    afun_t,
    params_new,
    params_old,
    model_state_new,
    model_state_old,
    sigma_new,
    args_new,
    sigma_old,
    args_old_dagger,
    c,
    return_grad,
):

    N = sigma_new.shape[-1]
    n_chains_new = sigma_new.shape[-2]

    sigma_new = sigma_new.reshape(-1, N)
    sigma_old = sigma_old.reshape(-1, N)

    conns_new = args_new[0].reshape(sigma_new.shape[0], -1, N)
    mels_new = args_new[1].reshape(sigma_new.shape[0], -1)
    conns_old_dagger = args_old_dagger[0].reshape(sigma_old.shape[0], -1, N)
    mels_old_dagger = args_old_dagger[1].reshape(sigma_old.shape[0], -1)

    n_conns_new = args_new[0].shape[-2]
    n_conns_old_dagger = args_old_dagger[0].shape[-2]

    n_samples = sigma_new.shape[0]

    conns_new_splitted = [
        c.reshape(n_samples, N) for c in jnp.split(conns_new, n_conns_new, axis=-2)
    ]
    conns_new_ravel = jnp.vstack(conns_new_splitted)

    conns_old_dagger_splitted = [
        c.reshape(n_samples, N)
        for c in jnp.split(conns_old_dagger, n_conns_old_dagger, axis=-2)
    ]
    conns_old_dagger_ravel = jnp.vstack(conns_old_dagger_splitted)

    params_new_real = jax.tree_map(lambda x: jnp.real(x), params_new)
    params_new_imag = jax.tree_map(lambda x: jnp.imag(x), params_new)

    def kernel_fun_real(params_new_real, params_old, sigma_new, sigma_old):

        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )

        variables_new = {"params": params_new, **model_state_new}
        variables_old = {"params": params_old, **model_state_old}

        logpsi0_new_ravel = afun_t(variables_old, conns_new_ravel)

        logpsi0_new_splitted = jnp.split(logpsi0_new_ravel, n_conns_new, axis=0)

        logpsi0_new = jnp.stack(logpsi0_new_splitted, axis=1)

        logpsi_new = jnp.expand_dims(afun(variables_new, sigma_new), -1)

        ratio_new = jnp.exp(logpsi0_new - logpsi_new)

        term_new = jnp.sum(mels_new * ratio_new, axis=1)

        logpsi_old_ravel = afun(variables_new, conns_old_dagger_ravel)

        logpsi_old_splitted = jnp.split(logpsi_old_ravel, n_conns_old_dagger, axis=0)

        logpsi_old = jnp.stack(logpsi_old_splitted, axis=1)

        logpsi0_old = jnp.expand_dims(afun_t(variables_old, sigma_old), -1)

        ratio_old = jnp.exp(logpsi_old - logpsi0_old)

        term_old = jnp.sum(mels_old_dagger * ratio_old, axis=1)

        return jnp.real(term_new * term_old) + c * (
            jnp.square(jnp.abs(term_new * term_old)) - 1
        )

    def kernel_fun_imag(params_new_imag, params_old, sigma_new, sigma_old):

        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )

        variables_new = {"params": params_new, **model_state_new}
        variables_old = {"params": params_old, **model_state_old}

        logpsi0_new_ravel = afun_t(variables_old, conns_new_ravel)

        logpsi0_new_splitted = jnp.split(logpsi0_new_ravel, n_conns_new, axis=0)

        logpsi0_new = jnp.stack(logpsi0_new_splitted, axis=1)

        logpsi_new = jnp.expand_dims(afun(variables_new, sigma_new), -1)

        ratio_new = jnp.exp(logpsi0_new - logpsi_new)

        term_new = jnp.sum(mels_new * ratio_new, axis=1)

        logpsi_old_ravel = afun(variables_new, conns_old_dagger_ravel)

        logpsi_old_splitted = jnp.split(logpsi_old_ravel, n_conns_old_dagger, axis=0)

        logpsi_old = jnp.stack(logpsi_old_splitted, axis=1)

        logpsi0_old = jnp.expand_dims(afun_t(variables_old, sigma_old), -1)

        ratio_old = jnp.exp(logpsi_old - logpsi0_old)

        term_old = jnp.sum(mels_old_dagger * ratio_old, axis=1)

        return jnp.real(term_new * term_old) + c * (
            jnp.square(jnp.abs(term_new * term_old)) - 1
        )

    def apply_fun_new_real(params_new_real):
        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )
        return afun({"params": params_new, **model_state_new}, sigma_new)

    def apply_fun_new_imag(params_new_imag):
        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )
        return afun({"params": params_new, **model_state_new}, sigma_new)

    _, Ok_vjp_real = nk.jax.vjp(apply_fun_new_real, params_new_real)
    __, Ok_vjp_imag = nk.jax.vjp(apply_fun_new_imag, params_new_imag)

    F_vals_real, F_vjp_real = nk.jax.vjp(
        kernel_fun_real, params_new_real, params_old, sigma_new, sigma_old
    )

    F_vals_imag, F_vjp_imag = nk.jax.vjp(
        kernel_fun_imag, params_new_imag, params_old, sigma_new, sigma_old
    )

    F_stats = nk.stats.statistics(F_vals_real.reshape((n_chains_new, -1)).T)

    F = F_stats.mean

    first_term_real = Ok_vjp_real(F_vals_real - F)[0]
    first_term_imag = Ok_vjp_imag(F_vals_imag - F)[0]

    second_term_real = F_vjp_real(jnp.ones_like(F_vals_real))[0]
    second_term_imag = F_vjp_imag(jnp.ones_like(F_vals_imag))[0]

    first_term_real = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        first_term_real,
        params_new_real,
    )
    first_term_imag = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        first_term_imag,
        params_new_imag,
    )

    F_grad_real = jax.tree_map(
        lambda x, y: nk.utils.mpi.mpi_mean_jax((x + y) / sigma_new.shape[0])[0],
        first_term_real,
        second_term_real,
    )

    F_grad_imag = jax.tree_map(
        lambda x, y: nk.utils.mpi.mpi_mean_jax((x + y) / sigma_new.shape[0])[0],
        first_term_imag,
        second_term_imag,
    )

    F_grad = jax.tree_map(lambda x, y: x + 1j * y, F_grad_real, F_grad_imag)
    I_grad = jax.tree_map(lambda x: -x, F_grad)

    I_stats = F_stats.replace(mean=1 - F)

    if return_grad:
        return I_stats, I_grad
    else:
        return I_stats
