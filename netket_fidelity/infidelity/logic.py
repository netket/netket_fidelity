from typing import Optional

from netket.operator import AbstractOperator, Adjoint
from netket.vqs import VariationalState, FullSumState
from netket.utils.types import DType

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
    Operator I_op computing the infidelity I among two variational states
    :math:`|\psi\rangle` and :math:`|\phi\rangle` as:

    .. math::

        I = 1 - \frac{|⟨\Psi|\Phi⟩|^2 }{ ⟨\Psi|\Psi⟩ ⟨\Phi|\Phi⟩ } = 1 - \frac{⟨\Psi|\hat{I}_{op}|\Psi⟩ }{ ⟨\Psi|\Psi⟩ }

    where:

    .. math::

        I_{op} = \frac {|\Phi\rangle\langle\Phi| }{ \langle\Phi|\Phi\rangle }

    The state :math:`|\phi\rangle` can be an autonomous state :math:`|\Phi\rangle = |\phi\rangle`
    or an operator :math:`U` applied to it, namely
    :math:`|\Phi\rangle  = U|\phi\rangle`. :math:`I_{op}` is defined by the
    state :math:`|\phi\rangle` (called target) and, possibly, by the operator
    :math:`U`. If :math:`U` is not specified, it is assumed :math:`|\Phi\rangle = |\phi\rangle`.

    The Monte Carlo estimator of I is:

    .. math::

        I = \mathbb{E}_{χ}[ I_{loc}(\sigma,\eta) ] =
            \mathbb{E}_{χ}\left[\frac{⟨\sigma|\Phi⟩ ⟨\eta|\Psi⟩}{⟨σ|\Psi⟩ ⟨η|\Phi⟩}\right]

    where the sampled probability distribution :math:`χ` is defined as:

    .. math::

        \chi(\sigma, \eta) = \frac{|\psi(\sigma)|^2 |\Phi(\eta)|^2}{
        \langle\Psi|\Psi\rangle  \langle\Phi|\Phi\rangle}.

    In practice, since I is a real quantity, :math:`\rm{Re}[I_{loc}(\sigma,\eta)]`
    is used. This estimator can be utilized both when :math:`|\Phi\rangle =|\phi\rangle` and
    when :math:`|\Phi\rangle =U|\phi\rangle`, with :math:`U` a (unitary or non-unitary) operator.
    In the second case, we have to sample from :math:`U|\phi\rangle` and this is implemented in
    the function :class:`netket_fidelity.infidelity.InfidelityUPsi` .

    This works only with the operators provdided in the package.
    We remark that sampling from :math:`U|\phi\rangle` requires to compute connected elements of
    :math:`U` and so is more expensive than sampling from an autonomous state.
    The choice of this estimator is specified by passing  :code:`sample_Upsi=True`,
    while the flag argument :code:`is_unitary` indicates whether :math:`U` is unitary or not.

    If :math:`U` is unitary, the following alternative estimator can be used:

    .. math::

        I = \mathbb{E}_{χ'}\left[ I_{loc}(\sigma, \eta) \right] =
            \mathbb{E}_{χ}\left[\frac{\langle\sigma|U|\phi\rangle \langle\eta|\psi\rangle}{
            \langle\sigma|U^{\dagger}|\psi\rangle ⟨\eta|\phi⟩} \right].

    where the sampled probability distribution :math:`\chi` is defined as:

    .. math::

        \chi'(\sigma, \eta) = \frac{|\psi(\sigma)|^2 |\phi(\eta)|^2}{
            \langle\Psi|\Psi\rangle  \langle\phi|\phi\rangle}.

    This estimator is more efficient since it does not require to sample from
    :math:`U|\phi\rangle`, but only from :math:`|\phi\rangle`.
    This choice of the estimator is the default and it works only
    with `is_unitary==True` (besides :code:`sample_Upsi=False` ).
    When :math:`|\Phi⟩ = |\phi⟩` the two estimators coincides.

    To reduce the variance of the estimator, the Control Variates (CV) method can be applied. This consists
    in modifying the estimator into:

    .. math::

        I_{loc}^{CV} = \rm{Re}\left[I_{loc}(\sigma, \eta)\right] - c \left(|1 - I_{loc}(\sigma, \eta)^2| - 1\right)

    where :math:`c ∈ \mathbb{R}`. The constant c is chosen to minimize the variance of
    :math:`I_{loc}^{CV}` as:

    .. math::

        c* = \frac{\rm{Cov}_{χ}\left[ |1-I_{loc}|^2, \rm{Re}\left[1-I_{loc}\right]\right]}{
            \rm{Var}_{χ}\left[ |1-I_{loc}|^2\right] },

    where :math:`\rm{Cov}\left\cdot, \cdot\right]` indicates the covariance and :math:`\rm{Var}\left[\cdot\right]` the variance.
    In the relevant limit :math:`|\Psi⟩ \rightarrow|\Phi⟩`, we have :math:`c^\star \rightarrow -1/2`. The value :math:`-1/2` is
    adopted as default value for c in the infidelity
    estimator. To not apply CV, set c=0.

    Args:
        target: target variational state :math:`|\phi⟩` .
        U: operator :math:`\hat{U}`.
        U_dagger: dagger operator :math:`\hat{U^\dagger}`.
        cv_coeff: Control Variates coefficient c.
        is_unitary: flag specifiying the unitarity of :math:`\hat{U}`. If True with
            :code:`sample_Upsi=False`, the second estimator is used.
        dtype: The dtype of the output of expectation value and gradient.
        sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False` , an error occurs.

    Returns:
        Infidelity operator for which computing expected value and gradient.

    Examples:

        >>> import netket as nk
        >>> import netket_fidelity as nkf
        >>>
        >>> hi = nk.hilbert.Spin(0.5, 4)
        >>> sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
        >>> model = nk.models.RBM(alpha=1, param_dtype=complex)
        >>> target_vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)
        >>>
        >>> # To optimise the overlap with |ϕ⟩
        >>> I_op = nkf.InfidelityOperator(target_vstate)
        >>>
        >>> # To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and |ϕ⟩
        >>> U = nkf.operator.Rx(0.3)
        >>> I_op = nkf.InfidelityOperator(target_vstate, U=U, is_unitary=True)
        >>>
        >>> # To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and U|ϕ⟩
        >>> I_op = nkf.InfidelityOperator(target_vstate, U=U, sample_Upsi=True)

    """
    if U is None:
        return InfidelityOperatorStandard(target, cv_coeff=cv_coeff, dtype=dtype)
    else:
        if U_dagger is None:
            U_dagger = U.H
        if isinstance(U_dagger, Adjoint):
            raise TypeError(
                "Must explicitly pass a jax-compatible operator as `U_dagger`. "
                "You either did not pass `U_dagger` explicitly or you used `U.H` but should "
                "use operators coming from `netket_fidelity`. "
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
                "This is more expensive and disabled by default. "
                ""
                "If your operator is Unitary, please specify so by passing `is_unitary=True` as a "
                "keyword argument. "
                ""
                "If your operator is not unitary, please specify `sample_Upsi=True` explicitly to"
                "sample from that state. "
                "You can also sample from U|ψ⟩ if your operator is unitary. "
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
