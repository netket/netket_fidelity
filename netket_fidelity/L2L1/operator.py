from typing import Optional
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.vqs import VariationalState

class L2L1Operator(AbstractOperator):
    def __init__(
        self,
        state_diff: VariationalState,
        *,
        dtype: Optional[DType] = None,
    ):
        super().__init__(state_diff.hilbert)

        if not isinstance(state_diff, VariationalState):
            raise TypeError("The first argument should be a variational state.")

        self._state_diff = state_diff
        self._dtype = dtype
    
    @property
    def state_diff(self):
        return self._state_diff

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"L2L1Operator(state_diff={self.state_diff})"
