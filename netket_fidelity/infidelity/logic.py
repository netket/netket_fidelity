from typing import Optional

from netket.operator import AbstractOperator, Adjoint
from netket.vqs import VariationalState
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
