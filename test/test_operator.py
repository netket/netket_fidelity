import pytest
import netket as nk
import numpy as np


import netket_fidelity as nkf


N = 3
hi = nk.hilbert.Spin(0.5, N)

operators = {}

operators["Rx"] = nkf.operator.Rx(hi, 1, 0.23)
operators["Ry"] = nkf.operator.Ry(hi, 1, 0.43)
operators["Hadamard"] = nkf.operator.Hadamard(hi, 0)


@pytest.mark.parametrize(
    "operator",
    [pytest.param(val, id=f"operator={name}") for name, val in operators.items()],
)
def test_operator_dense_and_conversion(operator):
    op_dense = operator.to_dense()
    op_local_dense = operator.to_local_operator().to_dense()

    np.testing.assert_allclose(op_dense, op_local_dense)
    assert operator.hilbert == operator.to_local_operator().hilbert
