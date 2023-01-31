import pytest
import netket as nk
import jax.numpy as jnp
import numpy as np

from infidelity_operator.infidelity_operator import InfidelityOperator, Rx


def _setup():

    N = 10
    n_samples = 10000

    hi = nk.hilbert.Spin(0.5, N)

    U = Rx(hi, 0, np.pi / 4)
    U_dagger = Rx(hi, 0, -np.pi / 4)

    U_matrx = U.to_dense()

    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)

    model = nk.models.RBM(alpha=1, param_dtype=complex)

    vstate_old = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples)
    vstate_new = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples)

    return vstate_old, vstate_new, U, U_dagger, U_matrx


def _infidelity_exact(vstate, params_new, U_matrx):
    params_old = vstate.parameters
    state_old = vstate.to_array()

    def infidelity_expect(params):
        vstate.parameters = params
        state_new = vstate.to_array()
        vstate.parameters = params_old

        return 1 - jnp.square(
            jnp.absolute(state_new.conj().T @ U_matrx @ state_old)
        ) / ((state_new.conj().T @ state_new) * (state_old.conj().T @ state_old))

    infidelity_grad = nk.jax.grad(infidelity_expect)

    return infidelity_expect(params_new), infidelity_grad(params_new)


def test_infidelity():
    vstate_old, vstate_new, U, U_dagger, U_matrx = _setup()

    infidelity_exact, infidelity_grad_exact = _infidelity_exact(
        vstate_old, vstate_new.parameters, U_matrx
    )
    I = InfidelityOperator(vstate_old, U=U, U_dagger=U_dagger)
    infidelity_sampled, infidelity_grad_sampled = vstate_new.expect_and_grad(I)
    np.testing.assert_allclose(
        infidelity_sampled.mean,
        infidelity_exact,
        atol=3 * infidelity_sampled.error_of_mean,
    )
    # jax.tree_map(
    # 	lambda x, y: np.testing.assert_allclose(x, np.conjugate(y), rtol = 0.3),
    # 	infidelity_grad_sampled,
    # 	infidelity_grad_exact
    # )

    I = InfidelityOperator(vstate_old, U=U, U_dagger=U_dagger, sampling_Uold=True)
    infidelity_sampled, infidelity_grad_sampled = vstate_new.expect_and_grad(I)
    np.testing.assert_allclose(
        infidelity_sampled.mean,
        infidelity_exact,
        atol=3 * infidelity_sampled.error_of_mean,
    )
    # jax.tree_map(
    # 	lambda x, y: np.testing.assert_allclose(x, np.conjugate(y), rtol = 0.3),
    # 	infidelity_grad_sampled,
    # 	infidelity_grad_exact
    # )

    infidelity_exact, infidelity_grad_exact = _infidelity_exact(
        vstate_old, vstate_new.parameters, jnp.identity(U_matrx.shape[0])
    )
    I = InfidelityOperator(vstate_old)
    infidelity_sampled, infidelity_grad_sampled = vstate_new.expect_and_grad(I)
    np.testing.assert_allclose(
        infidelity_sampled.mean,
        infidelity_exact,
        atol=3 * infidelity_sampled.error_of_mean,
    )
    # jax.tree_map(
    # 	lambda x, y: np.testing.assert_allclose(x, np.conjugate(y), rtol = 0.3),
    # 	infidelity_grad_sampled,
    # 	infidelity_grad_exact
    # )
