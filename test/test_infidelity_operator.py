import pytest
from pytest import approx
import netket as nk
import numpy as np

import netket_fidelity as nkf

from ._infidelity_exact import _infidelity_exact
from ._finite_diff import central_diff_grad, same_derivatives


def _setup():

    N = 3
    n_samples = 1e6
    n_discard_per_chain = 1e3

    hi = nk.hilbert.Spin(0.5, N)

    U = nkf.operator.Rx(hi, 0, 0.01)
    U_dag = nkf.operator.Rx(hi, 0, -0.01)

    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
    ma = nk.models.RBM(alpha=1)

    vs_t = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )
    vs = nk.vqs.MCState(
        sampler=sa,
        model=ma,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )

    vs_exact_t = nk.vqs.ExactState(
        hilbert=hi,
        model=ma,
    )
    vs_exact = nk.vqs.ExactState(
        hilbert=hi,
        model=ma,
    )

    return vs_t, vs, vs_exact_t, vs_exact, U, U_dag


@pytest.mark.parametrize("sample_Upsi", [False, True])
@pytest.mark.parametrize("is_identity", [False, True])
def test_MCState(sample_Upsi, is_identity):
    vs_t, vs, vs_exact_t, vs_exact, _U, _U_dag = _setup()

    if is_identity is False:
        U = _U
        U_dag = _U_dag

    else:
        U = None
        U_dag = None

    params, unravel = nk.jax.tree_ravel(vs.parameters)

    def _infidelity_exact_fun(params, vstate, U):
        return _infidelity_exact(unravel(params), vstate, U)

    I_exact = _infidelity_exact(
        vs.parameters,
        vs_t,
        U,
    )

    I_grad_exact = central_diff_grad(_infidelity_exact_fun, params, 1.0e-5, vs_t, U)

    I_op = nkf.infidelity.InfidelityOperator(
        vs_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
    )

    I_stat1 = vs.expect(I_op)
    I_stat, I_grad = vs.expect_and_grad(I_op)

    I1_mean = np.asarray(I_stat1.mean)
    I_mean = np.asarray(I_stat.mean)
    err = 5 * I_stat1.error_of_mean
    I_grad, _ = nk.jax.tree_ravel(I_grad)

    np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err)

    assert I1_mean.real == approx(I_mean.real, abs=1e-5)
    assert np.asarray(I_stat1.variance) == approx(np.asarray(I_stat.variance), abs=1e-5)

    same_derivatives(I_grad, I_grad_exact, rel_eps=1e-1)


@pytest.mark.parametrize("sample_Upsi", [False, True])
@pytest.mark.parametrize("is_identity", [False, True])
def test_ExactState(sample_Upsi, is_identity):
    vs_t, vs, vs_exact_t, vs_exact, _U, _U_dag = _setup()

    if is_identity is False:
        U = _U
        U_dag = _U_dag

    else:
        U = None
        U_dag = None

    params, unravel = nk.jax.tree_ravel(vs_exact.parameters)

    def _infidelity_exact_fun(params, vstate, U):
        return _infidelity_exact(unravel(params), vstate, U)

    I_exact = _infidelity_exact(
        vs_exact.parameters,
        vs_exact_t,
        U,
    )

    I_grad_exact = central_diff_grad(
        _infidelity_exact_fun, params, 1.0e-5, vs_exact_t, U
    )

    I_op = nkf.infidelity.InfidelityOperator(
        vs_exact_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
    )

    I_stat1 = vs_exact.expect(I_op)
    I_stat, I_grad = vs_exact.expect_and_grad(I_op)

    I1_mean = np.asarray(I_stat1.mean)
    I_mean = np.asarray(I_stat.mean)
    err = 5 * I_stat1.error_of_mean
    I_grad, _ = nk.jax.tree_ravel(I_grad)

    np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err)

    assert I1_mean.real == approx(I_mean.real, abs=1e-5)
    assert np.asarray(I_stat1.variance) == approx(np.asarray(I_stat.variance), abs=1e-5)

    same_derivatives(I_grad, I_grad_exact, rel_eps=1e-1)
