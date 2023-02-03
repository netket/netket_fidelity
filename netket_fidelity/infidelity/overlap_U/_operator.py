from typing import Optional
import jax.numpy as jnp
import jax
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import AbstractVariationalState, ExactState


class InfidelityOperatorUPsi(AbstractOperator):
    def __init__(
        self,
        state: AbstractVariationalState,
        U: AbstractOperator,
        *,
        control_variates: Optional[float] = None,
        U_dagger: AbstractOperator,
        is_unitary: bool = False,
        dtype: Optional[DType] = None,
    ):
        super().__init__(state.hilbert)

        if not isinstance(state, AbstractVariationalState):
            raise TypeError("The first argument should be a variational state.")

        if not is_unitary:
            raise ValueError(
                "Only works with unitary gates. If the gate is non unitary"
                "then you must sample from it. Use a different operator."
            )

        if control_variates is not None:
            control_variates = jnp.array(control_variates)

            if (not is_scalar(control_variates)) or jnp.iscomplex(control_variates):
                raise TypeError(
                    "control_variates should be a real scalar number or None."
                )

            if isinstance(state, ExactState):
                raise ValueError("With ExactState the control variate should be None")

        self._target = state
        self._cv_coeff = control_variates
        self._dtype = dtype

        self._U = U
        self._U_dagger = U_dagger

    def target(self):
        return self._target

    def cv_coeff(self):
        return self._cv_coeff

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True
