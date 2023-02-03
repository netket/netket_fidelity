from typing import Optional
from functools import partial
import jax.numpy as jnp
import jax
import netket as nk
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.dispatch import TrueT

from netket_fidelity.utils import expect_2distr, sampling_Ustate


class InfidelityOperator(AbstractOperator):
    def __init__(
        self,
        state: nk.vqs.MCState,
        *,
        U: AbstractOperator = None,
        U_dagger: AbstractOperator = None,
        c: float = -0.5,
        is_unitary: bool = True,
        sampling_Uold=False,
        dtype: Optional[DType] = None,
    ):
        super().__init__(state.hilbert)

        self.error = False

        if (U is None and U_dagger is not None) or (U is not None and U_dagger is None):
            self.error = True
            print(" Pass also the dagger operator. ")

        if sampling_Uold is not True and is_unitary is not True:
            self.error = True
            print(
                " You can only work with unitary operators if you don't sample from the target state. "
            )
            print(" To do this, pass sampling_Uold=True as an argument. ")

        if sampling_Uold is True and U is not None and U_dagger is not None:
            logpsiU = nk.jax.HashablePartial(sampling_Ustate, state._apply_fun, U)
            self.vstate_Uold = nk.vqs.MCState(
                sampler=state.sampler,
                apply_fun=logpsiU,
                n_samples=state.n_samples,
                variables=state.variables,
            )
        else:
            self.vstate_Uold = None

        self.vstate_old = state
        self.U = U
        self.U_dagger = U_dagger
        self.c = c
        self.is_unitary = is_unitary
        self.sampling_Uold = sampling_Uold

        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True


@nk.vqs.expect.dispatch
def infidelity(vstate_new: nk.vqs.MCState, op: InfidelityOperator):

    if op.error:
        return

    vstate_old = op.vstate_old
    vstate_Uold = op.vstate_Uold
    U = op.U
    U_dagger = op.U_dagger
    c = op.c
    is_unitary = op.is_unitary
    sampling_Uold = op.sampling_Uold

    if U is None and U_dagger is None:
        sigma_old = vstate_old.samples
        sigma_new = vstate_new.samples

        return infidelity_sampling_Uold(
            vstate_new._apply_fun,
            vstate_old._apply_fun,
            vstate_new.parameters,
            vstate_old.parameters,
            vstate_new.model_state,
            vstate_old.model_state,
            sigma_new,
            sigma_old,
            c,
            return_grad=False,
        )

    if U is not None and U_dagger is not None:

        if sampling_Uold is True:

            sigma_Uold = vstate_Uold.samples
            sigma_new = vstate_new.samples

            return infidelity_sampling_Uold(
                vstate_new._apply_fun,
                vstate_Uold._apply_fun,
                vstate_new.parameters,
                vstate_Uold.parameters,
                vstate_new.model_state,
                vstate_Uold.model_state,
                sigma_new,
                sigma_Uold,
                c,
                return_grad=False,
            )

        if sampling_Uold is not True and is_unitary is True:
            sigma_old, args_old_dagger = nk.vqs.get_local_kernel_arguments(
                vstate_old, U_dagger
            )
            sigma_new, args_new = nk.vqs.get_local_kernel_arguments(vstate_new, U)

            return infidelity_sampling_old(
                vstate_new._apply_fun,
                vstate_old._apply_fun,
                vstate_new.parameters,
                vstate_old.parameters,
                vstate_new.model_state,
                vstate_old.model_state,
                sigma_new,
                args_new,
                sigma_old,
                args_old_dagger,
                c,
                return_grad=False,
            )


@nk.vqs.expect_and_grad.dispatch
def infidelity(
    vstate_new: nk.vqs.MCState,
    op: InfidelityOperator,
    use_covariance: TrueT,
    *,
    mutable,
):

    if op.error:
        return

    vstate_old = op.vstate_old
    vstate_Uold = op.vstate_Uold
    U = op.U
    U_dagger = op.U_dagger
    c = op.c
    is_unitary = op.is_unitary
    sampling_Uold = op.sampling_Uold

    if U is None and U_dagger is None:

        sigma_old = vstate_old.samples
        sigma_new = vstate_new.samples

        return infidelity_sampling_Uold(
            vstate_new._apply_fun,
            vstate_old._apply_fun,
            vstate_new.parameters,
            vstate_old.parameters,
            vstate_new.model_state,
            vstate_old.model_state,
            sigma_new,
            sigma_old,
            c,
            return_grad=True,
        )

    if U is not None and U_dagger is not None:

        if sampling_Uold is True:

            if sampling_Uold is True:

                sigma_Uold = vstate_Uold.samples
                sigma_new = vstate_new.samples

                return infidelity_sampling_Uold(
                    vstate_new._apply_fun,
                    vstate_Uold._apply_fun,
                    vstate_new.parameters,
                    vstate_Uold.parameters,
                    vstate_new.model_state,
                    vstate_Uold.model_state,
                    sigma_new,
                    sigma_Uold,
                    c,
                    return_grad=True,
                )

        if sampling_Uold is not True and is_unitary is True:
            sigma_old, args_old_dagger = nk.vqs.get_local_kernel_arguments(
                vstate_old, U_dagger
            )
            sigma_new, args_new = nk.vqs.get_local_kernel_arguments(vstate_new, U)

            return infidelity_sampling_old(
                vstate_new._apply_fun,
                vstate_old._apply_fun,
                vstate_new.parameters,
                vstate_old.parameters,
                vstate_new.model_state,
                vstate_old.model_state,
                sigma_new,
                args_new,
                sigma_old,
                args_old_dagger,
                c,
                return_grad=True,
            )


@partial(jax.jit, static_argnames=("apply_fun_old", "apply_fun_new", "return_grad"))
def infidelity_sampling_old(
    apply_fun_new,
    apply_fun_old,
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

        logpsi0_new_ravel = apply_fun_old(variables_old, conns_new_ravel)

        logpsi0_new_splitted = jnp.split(logpsi0_new_ravel, n_conns_new, axis=0)

        logpsi0_new = jnp.stack(logpsi0_new_splitted, axis=1)

        logpsi_new = jnp.expand_dims(apply_fun_new(variables_new, sigma_new), -1)

        ratio_new = jnp.exp(logpsi0_new - logpsi_new)

        term_new = jnp.sum(mels_new * ratio_new, axis=1)

        logpsi_old_ravel = apply_fun_new(variables_new, conns_old_dagger_ravel)

        logpsi_old_splitted = jnp.split(logpsi_old_ravel, n_conns_old_dagger, axis=0)

        logpsi_old = jnp.stack(logpsi_old_splitted, axis=1)

        logpsi0_old = jnp.expand_dims(apply_fun_old(variables_old, sigma_old), -1)

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

        logpsi0_new_ravel = apply_fun_old(variables_old, conns_new_ravel)

        logpsi0_new_splitted = jnp.split(logpsi0_new_ravel, n_conns_new, axis=0)

        logpsi0_new = jnp.stack(logpsi0_new_splitted, axis=1)

        logpsi_new = jnp.expand_dims(apply_fun_new(variables_new, sigma_new), -1)

        ratio_new = jnp.exp(logpsi0_new - logpsi_new)

        term_new = jnp.sum(mels_new * ratio_new, axis=1)

        logpsi_old_ravel = apply_fun_new(variables_new, conns_old_dagger_ravel)

        logpsi_old_splitted = jnp.split(logpsi_old_ravel, n_conns_old_dagger, axis=0)

        logpsi_old = jnp.stack(logpsi_old_splitted, axis=1)

        logpsi0_old = jnp.expand_dims(apply_fun_old(variables_old, sigma_old), -1)

        ratio_old = jnp.exp(logpsi_old - logpsi0_old)

        term_old = jnp.sum(mels_old_dagger * ratio_old, axis=1)

        return jnp.real(term_new * term_old) + c * (
            jnp.square(jnp.abs(term_new * term_old)) - 1
        )

    params_new_real = jax.tree_map(lambda x: jnp.real(x), params_new)
    params_new_imag = jax.tree_map(lambda x: jnp.imag(x), params_new)

    def apply_fun_new_real(params_new_real):
        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )
        return apply_fun_new({"params": params_new, **model_state_new}, sigma_new)

    def apply_fun_new_imag(params_new_imag):
        params_new = jax.tree_map(
            lambda x, y: x + 1j * y, params_new_real, params_new_imag
        )
        return apply_fun_new({"params": params_new, **model_state_new}, sigma_new)

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


"""
    N = sigma_new.shape[-1]
    n_chains_old = sigma_old.shape[-2]

    sigma_new = sigma_new.reshape(-1, N)
    sigma_old = sigma_old.reshape(-1, N)

    conns_new = args_new[0].reshape(sigma_new.shape[0], -1, N)
    mels_new = args_new[1].reshape(sigma_new.shape[0], -1)
    conns_old_dagger = args_old_dagger[0].reshape(sigma_old.shape[0], -1, N)
    mels_old_dagger = args_old_dagger[1].reshape(sigma_old.shape[0], -1)


    def kernel_shifted_expect(params_new):
        def kernel_fun(params_new, params_old, sigma_new, sigma_old):
            variables_new = {"params": params_new, **model_state_new}
            variables_old = {"params": params_old, **model_state_old}

            logpsi0_new = apply_fun_old(variables_old, conns_new)

            logpsi_new = jnp.expand_dims(apply_fun_new(variables_new, sigma_new), -1)

            ratio_new = jnp.exp(logpsi0_new - logpsi_new)

            term_new = jnp.sum(mels_new * ratio_new, axis=1)

            logpsi_old = apply_fun_new(variables_new, conns_old_dagger)

            logpsi0_old = jnp.expand_dims(apply_fun_old(variables_old, sigma_old), -1)

            ratio_old = jnp.exp(logpsi_old - logpsi0_old)

            term_old = jnp.sum(mels_old_dagger * ratio_old, axis=1)


            return jnp.real(term_new * term_old) + c * (
                jnp.square(jnp.abs(term_new * term_old)) - 1
            )


        log_pdf_new = lambda params, sigma: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun_new({"params": params}, sigma))))
        )

        log_pdf_old = lambda params, sigma: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun_old({"params": params}, sigma))))
        )

        return expect_2distr(
            log_pdf_new,
            log_pdf_old,
            kernel_fun,
            params_new,
            params_old,
            sigma_new,
            sigma_old,
            n_chains=n_chains_old,
        )


    if not return_grad:
        F, F_stats = kernel_shifted_expect(params_new)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nk.jax.vjp(
        kernel_shifted_expect, params_new, has_aux=True, conjugate=True
    )

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]

    F_grad = jax.tree_map(lambda x: nk.utils.mpi.mpi_mean_jax(x)[0], F_grad)

    I_grad = jax.tree_map(lambda x: -x, F_grad)

    I_stats = F_stats.replace(mean=1 - F)


    if(return_grad):
        return I_stats, I_grad
    else: 
        return I_stats

"""


@partial(jax.jit, static_argnames=("apply_fun_old", "apply_fun_new", "return_grad"))
def infidelity_sampling_Uold(
    apply_fun_new,
    apply_fun_old,
    params_new,
    params_old,
    model_state_new,
    model_state_old,
    sigma_new,
    sigma_old,
    c,
    return_grad,
):

    N = sigma_new.shape[-1]
    n_chains_old = sigma_old.shape[-2]

    sigma_new = sigma_new.reshape(-1, N)
    sigma_old = sigma_old.reshape(-1, N)

    def kernel_shifted_expect(params_new):
        def kernel_fun(params_new, params_old, sigma_new, sigma_old):
            variables_new = {"params": params_new, **model_state_new}
            variables_old = {"params": params_old, **model_state_old}

            logpsiold_new = apply_fun_old(variables_old, sigma_new)

            logpsinew_new = apply_fun_new(variables_new, sigma_new)

            term_new = jnp.exp(logpsiold_new) / jnp.exp(logpsinew_new)

            logpsinew_old = apply_fun_new(variables_new, sigma_old)

            logpsiold_old = apply_fun_old(variables_old, sigma_old)

            term_old = jnp.exp(logpsinew_old) / jnp.exp(logpsiold_old)

            return jnp.real(term_new * term_old) + c * (
                jnp.square(jnp.abs(term_new * term_old)) - 1
            )

        log_pdf_new = lambda params, sigma: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun_new({"params": params}, sigma))))
        )

        log_pdf_old = lambda params, sigma: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun_old({"params": params}, sigma))))
        )

        return expect_2distr(
            log_pdf_new,
            log_pdf_old,
            kernel_fun,
            params_new,
            params_old,
            sigma_new,
            sigma_old,
            n_chains=n_chains_old,
        )

    if not return_grad:
        F, F_stats = kernel_shifted_expect(params_new)
        return F_stats.replace(mean=1 - F)

    F, F_vjp_fun, F_stats = nk.jax.vjp(
        kernel_shifted_expect, params_new, has_aux=True, conjugate=True
    )

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]

    F_grad = jax.tree_map(lambda x: nk.utils.mpi.mpi_mean_jax(x)[0], F_grad)

    I_grad = jax.tree_map(lambda x: -x, F_grad)

    I_stats = F_stats.replace(mean=1 - F)

    if return_grad:
        return I_stats, I_grad
    else:
        return I_stats
