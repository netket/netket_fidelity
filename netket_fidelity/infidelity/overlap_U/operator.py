from typing import Optional
import jax.numpy as jnp
import jax
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState, ExactState


class InfidelityOperatorUPsi(AbstractOperator):
    def __init__(
        self,
        U: AbstractOperator,
        state: VariationalState,
        *,
        cv_coeff: Optional[float] = None,
        U_dagger: AbstractOperator,
        is_unitary: bool = False,
        dtype: Optional[DType] = None,
    ):
        super().__init__(state.hilbert)

        if not isinstance(state, VariationalState):
            raise TypeError("The first argument should be a variational state.")

        if not is_unitary and not isinstance(state, ExactState):
            raise ValueError(
                "Only works with unitary gates. If the gate is non unitary"
                " then you must sample from it. Use a different operator."
            )

        if cv_coeff is not None:
            cv_coeff = jnp.array(cv_coeff)

            if (not is_scalar(cv_coeff)) or jnp.iscomplex(cv_coeff):
                raise TypeError("`cv_coeff` should be a real scalar number or None.")

            if isinstance(state, ExactState):
                cv_coeff = None

        self._target = state
        self._cv_coeff = cv_coeff
        self._dtype = dtype

        self._U = U
        self._U_dagger = U_dagger

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
        return f"InfidelityOperatorUPsi(target=U@{self.target}, U={self._U}, cv_coeff={self.cv_coeff})"
