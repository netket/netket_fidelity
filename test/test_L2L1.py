import pytest
from pytest import approx
import netket as nk
import numpy as np
import jax 

import netket_fidelity as nkf

from ._L2L1_exact import _L2L1_exact
from ._finite_diff import central_diff_grad, same_derivatives
from netket_fidelity.utils.sampling_diffstate import _logpsi_diff, _logpsi_diff_noU

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

    return vs_t, vs, vs_exact_t, vs_exact, U


@pytest.mark.parametrize("is_identity", [False, True])
def test_MCState(is_identity):
    vs_t, vs, vs_exact_t, vs_exact, U = _setup()

    if is_identity is False:
        
        logpsi_diff = jax.tree_util.Partial(
            nk.jax.HashablePartial(
                _logpsi_diff, vs_t._apply_fun, vs._apply_fun, U
            ),
            vs_t.variables, 
        )
        
    else:
        
        logpsi_diff = jax.tree_util.Partial(
            nk.jax.HashablePartial(
                _logpsi_diff_noU, vs_t._apply_fun, vs._apply_fun,
            ),
            vs_t.variables, 
        )
        
        U = None


    vs_diff = nk.vqs.MCState(
        sampler=vs.sampler,
        apply_fun=logpsi_diff,
        n_samples=vs.n_samples,
        variables=vs.variables,
        n_discard_per_chain=vs.n_discard_per_chain,
    )

    params, unravel = nk.jax.tree_ravel(vs.parameters)

    def _L2L1_exact_fun(params, vstate, U):
        return _L2L1_exact(unravel(params), vstate, U)

    L2L1_exact = _L2L1_exact(
        vs.parameters,
        vs_t,
        U,
    )

    L2L1_grad_exact = central_diff_grad(_L2L1_exact_fun, params, 1.0e-5, vs_t, U)

    L2L1_op = nkf.L2L1.L2L1Operator(vs_diff)

    L2L1_stat1 = vs.expect(L2L1_op)
    L2L1_stat, L2L1_grad = vs.expect_and_grad(L2L1_op)

    L2L1_mean1 = np.asarray(L2L1_stat1.mean)
    L2L1_mean = np.asarray(L2L1_stat.mean)
    err = 5 * L2L1_stat1.error_of_mean
    L2L1_grad, _ = nk.jax.tree_ravel(L2L1_grad)

    np.testing.assert_allclose(L2L1_exact.real, L2L1_mean1.real, rtol=5e-2)

    assert L2L1_mean1.real == approx(L2L1_mean.real, abs=1e-5)
    assert np.asarray(L2L1_stat1.variance) == approx(np.asarray(L2L1_stat.variance), abs=1e-5)

    same_derivatives(L2L1_grad, L2L1_grad_exact, rel_eps=2e-1)

@pytest.mark.parametrize("is_identity", [False, True])
def test_ExactState(is_identity):
    vs_t, vs, vs_exact_t, vs_exact, U = _setup()

    if is_identity is False:
        
        logpsi_diff = jax.tree_util.Partial(
            nk.jax.HashablePartial(
                _logpsi_diff, vs_exact_t._apply_fun, vs_exact._apply_fun, U
            ),
            vs_exact_t.variables, 
        )
        
    else:
        
        logpsi_diff = jax.tree_util.Partial(
            nk.jax.HashablePartial(
                _logpsi_diff_noU, vs_exact_t._apply_fun, vs_exact._apply_fun,
            ),
            vs_exact_t.variables, 
        )
        
        U = None

    vs_diff_exact = nk.vqs.ExactState(
        hilbert=vs_exact.hilbert, 
        apply_fun=logpsi_diff,
        variables=vs_exact.variables,
    )

    params, unravel = nk.jax.tree_ravel(vs_exact.parameters)

    def _L2L1_exact_fun(params, vstate, U):
        return _L2L1_exact(unravel(params), vstate, U)

    L2L1_exact = _L2L1_exact(
        vs_exact.parameters,
        vs_exact_t,
        U,
    )

    L2L1_grad_exact = central_diff_grad(_L2L1_exact_fun, params, 1.0e-5, vs_exact_t, U)

    L2L1_op = nkf.L2L1.L2L1Operator(vs_diff_exact)

    L2L1_stat1 = vs_exact.expect(L2L1_op)
    L2L1_stat, L2L1_grad = vs_exact.expect_and_grad(L2L1_op)

    L2L1_mean1 = np.asarray(L2L1_stat1.mean)
    L2L1_mean = np.asarray(L2L1_stat.mean)
    L2L1_grad, _ = nk.jax.tree_ravel(L2L1_grad)

    np.testing.assert_allclose(L2L1_exact.real, L2L1_mean1.real, rtol=1e-12)

    assert L2L1_mean1.real == approx(L2L1_mean.real, abs=1e-12)
    assert np.asarray(L2L1_stat1.variance) == approx(np.asarray(L2L1_stat.variance), abs=1e-12)

    same_derivatives(L2L1_grad, L2L1_grad_exact, rel_eps=1e-12)
