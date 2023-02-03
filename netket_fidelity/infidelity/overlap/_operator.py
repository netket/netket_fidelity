from typing import Optional
import jax.numpy as jnp
import jax
from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import AbstractVariationalState, ExactState

from netket_fidelity.utils import sampling_Ustate


class InfidelityOperatorStandard(AbstractOperator):
    def __init__(
        self,
        target: AbstractVariationalState,
        *,
        control_variates: Optional[float] = None,
        dtype: Optional[DType] = None,
    ):
        super().__init__(target.hilbert)

        if not isinstance(target, AbstractVariationalState):
            raise TypeError("The first argument should be a variational target.")

        if control_variates is not None:
            control_variates = jnp.array(control_variates)

            if (not is_scalar(control_variates)) or jnp.iscomplex(control_variates):
                raise TypeError(
                    "control_variates should be a real scalar number or None."
                )

            if isinstance(target, ExactState):
                raise ValueError("With ExactState the control variate should be None")

        self._target = target
        self._cv_coeff = control_variates
        self._dtype = dtype

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

    def __repr__(self):
        return f"InfidelityOperator(target={self.target}, cv_coeff={self.cv_coeff})"


def InfidelityUPsi(
    self,
    state: AbstractVariationalState,
    U: AbstractOperator,
    *,
    control_variates: Optional[float] = None,
    dtype: Optional[DType] = None,
):

    logpsiU_apply_fun = nkjax.HashablePartial(sampling_Ustate, state._apply_fun, U)
    target = MCState(
        sampler=state.sampler,
        apply_fun=logpsiU_apply_fun,
        n_samples=state.n_samples,
        variables=state.variables,
    )

    return InfidelityOperatorStandard(
        target, control_variates=control_variates, dtype=dtype
    )
