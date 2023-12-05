import pytest
import netket as nk
import numpy as np
from netket_fidelity.operator import singlequbit_gates as sg
from jax import numpy as jnp
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


def test_get_conns_and_mels():
    hi_spin = nk.hilbert.Spin(s=0.5, N=3)
    hi_qubit = nk.hilbert.Qubit(N=3)

    local_state_spin = hi_spin.local_states
    local_state_qubit = hi_qubit.local_states

    sigma_2_qubit = hi_qubit.numbers_to_states(2)
    sigma_7_qubit = hi_qubit.numbers_to_states(7)
    sigma_2_spin = hi_spin.numbers_to_states(2)
    sigma_7_spin = hi_spin.numbers_to_states(7)

    sigma_qubit = jnp.array([sigma_2_qubit, sigma_7_qubit])
    sigma_spin = jnp.array([sigma_2_spin, sigma_7_spin])

    conns_rx_qubit, mels_rx_qubit = sg.get_conns_and_mels_Rx(
        sigma_qubit, 0, np.pi / 2, local_state_qubit
    )
    conns_ry_qubit, mels_ry_qubit = sg.get_conns_and_mels_Ry(
        sigma_qubit, 0, np.pi / 2, local_state_qubit
    )
    conns_h_qubit, mels_h_qubit = sg.get_conns_and_mels_Hadamard(
        sigma_qubit, 0, local_state_qubit
    )

    conns_rx_spin, mels_rx_spin = sg.get_conns_and_mels_Rx(
        sigma_spin, 0, np.pi / 2, local_state_spin
    )
    conns_ry_spin, mels_ry_spin = sg.get_conns_and_mels_Ry(
        sigma_spin, 0, np.pi / 2, local_state_spin
    )
    conns_h_spin, mels_h_spin = sg.get_conns_and_mels_Hadamard(
        sigma_spin, 0, local_state_spin
    )

    conns_check_qubit = jnp.array(
        [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]
    )

    conns_check_spin = jnp.array(
        [[[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]], [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]]
    )

    mels_check_qubit_rx = jnp.array(
        [[0.70710678 + 0.0j, 0.0 - 0.70710678j], [0.70710678 + 0.0j, 0.0 - 0.70710678j]]
    )
    mels_check_qubit_ry = jnp.array(
        [
            [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            [0.70710678 + 0.0j, -0.70710678 + 0.0j],
        ]
    )
    mels_check_qubit_h = jnp.array(
        [[0.70710678, 0.70710678], [-0.70710678, 0.70710678]]
    )

    mels_check_spin_rx = jnp.array(
        [[0.70710678 + 0.0j, 0.0 - 0.70710678j], [0.70710678 + 0.0j, 0.0 - 0.70710678j]]
    )
    mels_check_spin_ry = jnp.array(
        [
            [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            [0.70710678 + 0.0j, -0.70710678 + 0.0j],
        ]
    )
    mels_check_spin_h = jnp.array([[0.70710678, 0.70710678], [-0.70710678, 0.70710678]])

    np.testing.assert_allclose(conns_rx_qubit, conns_check_qubit)
    np.testing.assert_allclose(conns_ry_qubit, conns_check_qubit)
    np.testing.assert_allclose(conns_h_qubit, conns_check_qubit)
    np.testing.assert_allclose(conns_rx_spin, conns_check_spin)
    np.testing.assert_allclose(conns_ry_spin, conns_check_spin)
    np.testing.assert_allclose(conns_h_spin, conns_check_spin)

    np.testing.assert_allclose(mels_rx_qubit, mels_check_qubit_rx)
    np.testing.assert_allclose(mels_ry_qubit, mels_check_qubit_ry)
    np.testing.assert_allclose(mels_h_qubit, mels_check_qubit_h)
    np.testing.assert_allclose(mels_rx_spin, mels_check_spin_rx)
    np.testing.assert_allclose(mels_ry_spin, mels_check_spin_ry)
    np.testing.assert_allclose(mels_h_spin, mels_check_spin_h)
