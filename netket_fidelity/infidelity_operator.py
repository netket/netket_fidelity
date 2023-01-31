from typing import Optional, Callable, Tuple
from functools import partial
import jax.numpy as jnp
import numpy as np
import jax
from jax.tree_util import register_pytree_node_class
import netket as nk
from netket.operator import AbstractOperator, DiscreteOperator
from netket.utils.types import DType, PyTree
from netket.utils.dispatch import TrueT
from netket.jax._vjp import vjp as nkvjp
from netket.stats import statistics as mpi_statistics, Stats


@register_pytree_node_class
class Rx(DiscreteOperator):
    def __init__(self, hi, idx, angle):
        super().__init__(hi)
        self.idx = idx
        self.angle = angle / 2

    @property
    def dtype(self):
        return complex

    def __hash__(self):
        return hash(("Rx", self.idx))

    def __eq__(self, o):
        if isinstance(o, Rx):
            return o.idx == self.idx
        return False

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.idx, self.angle)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Rx(xr, self.idx, self.angle)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Rx(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle))
    mels = mels.at[1].set(-1j * jnp.sin(angle))

    return conns, mels


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Rx):
    sigma = vstate.samples
    conns, mels = get_conns_and_mels_Rx(
        sigma.reshape(-1, vstate.hilbert.size), op.idx, op.angle
    )

    conns = conns.reshape((sigma.shape[0], sigma.shape[1], -1, vstate.hilbert.size))
    mels = mels.reshape((sigma.shape[0], sigma.shape[1], -1))

    return sigma, (conns, mels)


@register_pytree_node_class
class Ry(DiscreteOperator):
    def __init__(self, hi, idx, angle):
        super().__init__(hi)
        self.idx = idx
        self.angle = angle / 2

    @property
    def dtype(self):
        return complex

    def __hash__(self):
        return hash(("Ry", self.idx))

    def __eq__(self, o):
        if isinstance(o, Ry):
            return o.idx == self.idx
        return False

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.idx, self.angle)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Ry(xr, self.idx, self.angle)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Ry(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle))
    mels = mels.at[1].set((-1) ** ((conns.at[0, idx].get() + 1) / 2) * jnp.sin(angle))

    return conns, mels


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Ry):
    sigma = vstate.samples
    conns, mels = get_conns_and_mels_Ry(
        sigma.reshape(-1, vstate.hilbert.size), op.idx, op.angle
    )

    conns = conns.reshape((sigma.shape[0], sigma.shape[1], -1, vstate.hilbert.size))
    mels = mels.reshape((sigma.shape[0], sigma.shape[1], -1))

    return sigma, (conns, mels)


def expect_fid(
    log_pdf_new: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    log_pdf_old: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars_new: PyTree,
    pars_old: PyTree,
    σ_new: jnp.ndarray,
    σ_old: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """

    return _expect_fid(
        n_chains,
        log_pdf_new,
        log_pdf_old,
        expected_fun,
        pars_new,
        pars_old,
        σ_new,
        σ_old,
        *expected_fun_args,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _expect_fid(
    n_chains,
    log_pdf_new,
    log_pdf_old,
    expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    *expected_fun_args,
):

    L_σ = expected_fun(pars_new, pars_old, σ_new, σ_old, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ.T)

    return L̄_σ.mean, L̄_σ


def _expect_fwd_fid(
    n_chains,
    log_pdf_new,
    log_pdf_old,
    expected_fun,
    pars_new,
    pars_old,
    σ_new,
    σ_old,
    *expected_fun_args,
):
    L_σ = expected_fun(pars_new, pars_old, σ_new, σ_old, *expected_fun_args)
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars_new, pars_old, σ_new, σ_old, expected_fun_args, ΔL_σ)


def _expect_bwd_fid(n_chains, log_pdf_new, log_pdf_old, expected_fun, residuals, dout):
    pars_new, pars_old, σ_new, σ_old, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout
    log_p_old = log_pdf_old(pars_old, σ_old)

    def f(pars_new, pars_old, σ_new, σ_old, *cost_args):
        log_p = log_pdf_new(pars_new, σ_new) + log_p_old
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars_new, pars_old, σ_new, σ_old, *cost_args)
        out = term1 + term2
        out = out.mean()
        return out

    _, pb = nkvjp(f, pars_new, pars_old, σ_new, σ_old, *cost_args)

    grad_f = pb(dL̄)

    return grad_f


_expect_fid.defvjp(_expect_fwd_fid, _expect_bwd_fid)


def sampling_Ustate(apply_fun, U, variables, x):
    xp, mels = U.get_conn_padded(x)
    logpsi_xp = apply_fun(variables, xp)

    @jax.jit
    def sampling_Ustate_inner(logpsi_xp, mels):
        return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

    return sampling_Ustate_inner(logpsi_xp, mels)


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
            print(" To do this, pass is_unitary=True as an argument. ")

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

    return I_stats, I_grad


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

        return expect_fid(
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

    return I_stats, I_grad
