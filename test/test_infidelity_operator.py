import pytest
from pytest import approx
import netket as nk
import jax.numpy as jnp
import numpy as np
import scipy 

from netket_fidelity._infidelity_operator import InfidelityOperator
from netket_fidelity.operator._1qubit_rotations import Rx
from netket_fidelity._infidelity_exact import _infidelity_exact
from netket_fidelity._finite_diff import central_diff_grad, same_derivatives

def _setup():

    N = 4
    n_samples = 1e6
    n_discard_per_chain = 1e3
    
    hi = nk.hilbert.Spin(0.5, N)

    U = Rx(hi, 0, 0.01)
    U_dagger = Rx(hi, 0, -0.01)

    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)

    model = nk.models.RBM(alpha=1, param_dtype=complex)

    vstate_old = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    vstate_new = nk.vqs.MCState(sampler=sampler, model=model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

    return vstate_old, vstate_new, U, U_dagger


def test_infidelity():
    vstate_old, vstate_new, U, U_dagger = _setup()
    U_sparse = U.to_sparse()

    params_0 = vstate_new.parameters
    params, unravel = nk.jax.tree_ravel(params_0)

    def _infidelity_exact_fun(params, vstate, U_sparse):
        return _infidelity_exact(unravel(params), vstate, U_sparse)

    I_exact = _infidelity_exact(
        vstate_new.parameters, vstate_old, U_sparse, 
    )

    grad_exact = central_diff_grad(_infidelity_exact_fun, params, 1.0e-5, vstate_old, U_sparse)

    for b in [True, False]: 

        I_op = InfidelityOperator(vstate_old, U=U, U_dagger=U_dagger, sampling_Uold = b, c = -0.5)
        
        I_stat1 = vstate_new.expect(I_op)
        I_stat, I_grad = vstate_new.expect_and_grad(I_op)
        
        I1_mean = np.asarray(I_stat1.mean)
        I_mean = np.asarray(I_stat.mean)
        err = 5 * I_stat1.error_of_mean

        np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err, rtol=err)

        assert I1_mean.real == approx(I_mean.real, abs=1e-5)
        assert np.asarray(I_stat1.variance) == approx(np.asarray(I_stat.variance), abs=1e-5)

        I_grad, _ = nk.jax.tree_ravel(I_grad)
        
        same_derivatives(I_grad, grad_exact, rel_eps=1e-1)


    I_exact = _infidelity_exact(
        vstate_new.parameters, vstate_old, scipy.sparse.identity(U_sparse.shape[0]), 
    )

    grad_exact = central_diff_grad(_infidelity_exact_fun, params, 1.0e-5, vstate_old, scipy.sparse.identity(U_sparse.shape[0]))

    I_op = InfidelityOperator(vstate_old, c = -0.5)
    
    I_stat1 = vstate_new.expect(I_op)
    I_stat, I_grad = vstate_new.expect_and_grad(I_op)
    
    I1_mean = np.asarray(I_stat1.mean)
    I_mean = np.asarray(I_stat.mean)
    err = 5 * I_stat1.error_of_mean

    np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err, rtol=err)

    assert I1_mean.real == approx(I_mean.real, abs=1e-5)
    assert np.asarray(I_stat1.variance) == approx(np.asarray(I_stat.variance), abs=1e-5)

    I_grad, _ = nk.jax.tree_ravel(I_grad)

    same_derivatives(I_grad, grad_exact, rel_eps = 1e-1)

