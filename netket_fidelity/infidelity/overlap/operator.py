from typing import Optional

import jax.numpy as jnp

import flax

from netket import jax as nkjax
from netket.operator import AbstractOperator, DiscreteJaxOperator
from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState, MCState, FullSumState

from netket_fidelity.utils.sampling_Ustate import _logpsi_U


class InfidelityOperatorStandard(AbstractOperator):
    def __init__(
        self,
        target: VariationalState,
        *,
        cv_coeff: Optional[float] = None,
        dtype: Optional[DType] = None,
    ):
        super().__init__(target.hilbert)

        if not isinstance(target, VariationalState):
            raise TypeError("The first argument should be a variational target.")

        if cv_coeff is not None:
            cv_coeff = jnp.array(cv_coeff)

            if (not is_scalar(cv_coeff)) or jnp.iscomplex(cv_coeff):
                raise TypeError("`cv_coeff` should be a real scalar number or None.")

            if isinstance(target, FullSumState):
                cv_coeff = None

        self._target = target
        self._cv_coeff = cv_coeff
        self._dtype = dtype

    @property
    def target(self):
        return self._target

    @property
    def cv_coeff(self):
        return self._cv_coeff

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"InfidelityOperator(target={self.target}, cv_coeff={self.cv_coeff})"


def InfidelityUPsi(
    U: AbstractOperator,
    state: VariationalState,
    *,
    cv_coeff: Optional[float] = None,
    dtype: Optional[DType] = None,
):

    if not isinstance(U, DiscreteJaxOperator):
        raise TypeError(
            "In order to sample from the state U|psi>, U must be"
            "an instance of DiscreteJaxOperator."
        )

    logpsiU = nkjax.HashablePartial(_logpsi_U, state._apply_fun)
    target = MCState(
        sampler=state.sampler,
        apply_fun=logpsiU,
        n_samples=state.n_samples,
        variables=flax.core.copy(state.variables, {"unitary": U}),
    )

    return InfidelityOperatorStandard(target, cv_coeff=cv_coeff, dtype=dtype)
