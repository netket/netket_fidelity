from typing import Optional

from netket.operator import AbstractOperator, Adjoint
from netket.vqs import VariationalState
from netket.utils.types import DType

import netket

if hasattr(netket.vqs, "FullSumState"):
    from netket.vqs import FullSumState
else:
    from netket.vqs import ExactState as FullSumState

from .overlap import InfidelityOperatorStandard, InfidelityUPsi
from .overlap_U import InfidelityOperatorUPsi


def InfidelityOperator(
    target: VariationalState,
    *,
    U: AbstractOperator = None,
    U_dagger: AbstractOperator = None,
    cv_coeff: Optional[float] = None,
    is_unitary: bool = False,
    dtype: Optional[DType] = None,
    sample_Upsi=False,
):
    r"""
    Operator I_op computing the infidelity I among two variational states |ψ⟩ and |Φ⟩ as:

    .. math::

        I = 1 - |⟨ψ|Φ⟩|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩ = 1 - ⟨ψ|I_op|ψ⟩ / ⟨ψ|ψ⟩

    where:

     .. math::

        I_op = |Φ⟩⟨Φ| / ⟨Φ|Φ⟩

    The state |Φ⟩ can be an autonomous state |Φ⟩ =|ϕ⟩ or an operator U applied to it, namely
    |Φ⟩  = U|ϕ⟩. I_op is defined by the state |ϕ⟩ (called target) and, possibly, by the operator U.
    If U is not passed, it is assumed |Φ⟩ =|ϕ⟩.

    The Monte Carlo estimator of I is:

    ..math::

        I = \mathbb{E}_{χ}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|Φ⟩ ⟨η|ψ⟩ / ⟨σ|ψ⟩ ⟨η|Φ⟩ ]

    where χ(σ, η) = |Ψ(σ)|^2 |Φ(η)|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩. In practice, since I is a real quantity, Re{I_loc(σ,η)}
    is used. This estimator can be utilized both when |Φ⟩ =|ϕ⟩ and when |Φ⟩ = U|ϕ⟩, with U a (unitary or
    non-unitary) operator. In the second case, we have to sample from U|ϕ⟩ and this is implemented in
    the function :ref:`jax.:ref:`InfidelityUPsi`. This works only with the operators provdided in the package.
    We remark that sampling from U|ϕ⟩ requires to compute connected elements of U and so is more expensive
    than sampling from an autonomous state. The choice of this estimator is specified by passing
    `sample_Upsi=True`, while the flag argument `is_unitary` indicates whether U is unitary or not.

    If U is unitary, the following alternative estimator can be used:

    ..math::

        I = \mathbb{E}_{χ'}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|U|ϕ⟩ ⟨η|ψ⟩ / ⟨σ|U^{\dagger}|ψ⟩ ⟨η|ϕ⟩ ].

    where χ'(σ, η) = |Ψ(σ)|^2 |ϕ(η)|^2 / ⟨ψ|ψ⟩ ⟨ϕ|ϕ⟩. This estimator is more efficient since it does not
    require to sample from U|ϕ⟩, but only from |ϕ⟩. This choice of the estimator is the default and it works only
    with `is_unitary==True` (besides `sample_Upsi=False`). When |Φ⟩ = |ϕ⟩ the two estimators coincides.

    To reduce the variance of the estimator, the Control Variates (CV) method can be applied. This consists
    in modifying the estimator into:

    ..math::

        I_loc^{CV} = Re{I_loc(σ,η)} - c (|1 - I_loc(σ,η)^2| - 1)

    where c ∈ \mathbb{R}. The constant c is chosen to minimize the variance of I_loc^{CV} as:

    ..math::

        c* = Cov_{χ}[ |1-I_loc|^2, Re{1-I_loc}] / Var_{χ}[ |1-I_loc|^2 ],

    where Cov[..., ...] indicates the covariance and Var[...] the variance. In the relevant limit
    |Ψ⟩ →|Φ⟩, we have c*→-1/2. The value -1/2 is adopted as default value for c in the infidelity
    estimator. To not apply CV, set c=0.

    Args:
        target: target variational state |ϕ⟩.
        U: operator U.
        U_dagger: dagger operator U^{\dagger}.
        cv_coeff: Control Variates coefficient c.
        is_unitary: flag specifiying the unitarity of U. If True with `sample_Upsi=False`, the second estimator is used.
        dtype: The dtype of the output of expectation value and gradient.
        sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False`, an error occurs.

    Returns:
        Infidelity operator for which computing expected value and gradient.

    Example:
        import netket as nk
        import netket_fidelity as nkf

        hi = nk.hilbert.Spin(0.5, 4)
        sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
        model = nk.models.RBM(alpha=1, param_dtype=complex)
        target_vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)

        # To optimise the overlap with |ϕ⟩
        I_op = nkf.InfidelityOperator(target_vstate)

        # To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and |ϕ⟩
        U = nkf.operator.Rx(0.3)
        I_op = nkf.InfidelityOperator(target_vstate, U=U, is_unitary=True)

        # To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and U|ϕ⟩
        I_op = nkf.InfidelityOperator(target_vstate, U=U, sample_Upsi=True)

    """
    if U is None:
        return InfidelityOperatorStandard(target, cv_coeff=cv_coeff, dtype=dtype)
    else:
        if U_dagger is None:
            U_dagger = U.H
        if isinstance(U_dagger, Adjoint):
            raise TypeError(
                "Must explicitly pass a jax-compatible operator as `U_dagger`."
                "You either did not pass `U_dagger` explicitly or you used `U.H` but should"
                "use operators coming from `netket_fidelity`."
            )

        if isinstance(target, FullSumState):
            return InfidelityOperatorUPsi(
                U,
                target,
                U_dagger=U_dagger,
                cv_coeff=cv_coeff,
                dtype=dtype,
            )

        if not is_unitary and not sample_Upsi:
            raise ValueError(
                "Non-unitary operators can only be handled by sampling from the state U|ψ⟩. "
                "This is more expensive and disabled by default."
                ""
                "If your operator is Unitary, please specify so by passing `is_unitary=True` as a "
                "keyword argument."
                ""
                "If your operator is not unitary, please specify `sample_Upsi=True` explicitly to"
                "sample from that state."
                "You can also sample from U|ψ⟩ if your operator is unitary."
                ""
            )

        if sample_Upsi:
            return InfidelityUPsi(U, target, cv_coeff=cv_coeff, dtype=dtype)
        else:
            return InfidelityOperatorUPsi(
                U,
                target,
                U_dagger=U_dagger,
                cv_coeff=cv_coeff,
                dtype=dtype,
                is_unitary=True,
            )
