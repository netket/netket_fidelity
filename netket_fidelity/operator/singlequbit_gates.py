from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class

from netket.hilbert import Spin
from netket.operator import DiscreteJaxOperator, spin


@register_pytree_node_class
class Rx(DiscreteJaxOperator):
    def __init__(self, hi, idx, angle):
        # if not isinstance(hi, Spin):
        #     raise TypeError("""The Hilbert space used by Rx must be a `Spin` space.
        #
        #         This limitation could be lifted by 'fixing' the method
        #         `get_conn_and_mels` to work with arbitrary hilbert spaces, which
        #         should be relatively straightforward to do, but we have not done so
        #         yet.
        #         """)
        super().__init__(hi)
        self._idx = idx
        self._angle = angle

    @property
    def angle(self):
        """
        The angle of this rotation.
        """
        return self._angle

    @property
    def idx(self):
        """
        The qubit id on which this rotation acts
        """
        return self._idx

    @property
    def dtype(self):
        return complex

    @property
    def H(self):
        return Rx(self.hilbert, self.idx, -self.angle)

    def __eq__(self, o):
        if isinstance(o, Rx):
            return o.idx == self.idx and o.angle == self.angle
        return False

    def tree_flatten(self):
        children = (self.angle,)
        aux_data = (
            self.hilbert,
            self.idx,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (angle,) = children
        return cls(*aux_data, angle)

    @property
    def max_conn_size(self) -> int:
        return 2

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Rx(xr, self.idx, self.angle)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels

    def to_local_operator(self):
        ctheta = np.cos(self.angle / 2)
        stheta = np.sin(self.angle / 2)
        return ctheta - 1j * stheta * spin.sigmax(self.hilbert, self.idx)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Rx(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())
    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle / 2))
    mels = mels.at[1].set(-1j * jnp.sin(angle / 2))

    return conns, mels


@register_pytree_node_class
class Ry(DiscreteJaxOperator):
    def __init__(self, hi, idx, angle):
        if not isinstance(hi, Spin):
            raise TypeError(
                """The Hilbert space used by Rx must be a `Spin` space.

                This limitation could be lifted by 'fixing' the method
                `get_conn_and_mels` to work with arbitrary hilbert spaces, which
                should be relatively straightforward to do, but we have not done so
                yet.
                """
            )

        super().__init__(hi)
        self._idx = idx
        self._angle = angle

    @property
    def angle(self):
        """
        The angle of this rotation.
        """
        return self._angle

    @property
    def idx(self):
        """
        The qubit id on which this rotation acts
        """
        return self._idx

    @property
    def dtype(self):
        return complex

    @property
    def H(self):
        return Ry(self.hilbert, self.idx, -self.angle * 2)

    @property
    def max_conn_size(self) -> int:
        return 2

    def __eq__(self, o):
        if isinstance(o, Ry):
            return o.idx == self.idx and o.angle == self.angle
        return False

    def tree_flatten(self):
        children = (self.angle,)
        aux_data = (
            self.hilbert,
            self.idx,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (angle,) = children
        return cls(*aux_data, angle)

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Ry(xr, self.idx, self.angle)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels

    def to_local_operator(self):
        ctheta = np.cos(self.angle / 2)
        stheta = np.sin(self.angle / 2)
        return ctheta + 1j * stheta * spin.sigmay(self.hilbert, self.idx)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Ry(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle / 2))
    mels = mels.at[1].set(
        (-1) ** ((conns.at[0, idx].get() + 1) / 2) * jnp.sin(angle / 2)
    )

    return conns, mels


@register_pytree_node_class
class Hadamard(DiscreteJaxOperator):
    def __init__(self, hi, idx):
        # if not isinstance(hi, Spin):
        #     raise TypeError("""The Hilbert space used by Rx must be a `Spin` space.
        #
        #         This limitation could be lifted by 'fixing' the method
        #         `get_conn_and_mels` to work with arbitrary hilbert spaces, which
        #         should be relatively straightforward to do, but we have not done so
        #         yet.
        #         """)
        self._local_states = hi.local_states
        super().__init__(hi)
        self._idx = idx

    @property
    def idx(self):
        """
        The qubit id on which this hadamard gate acts upon.
        """
        return self._idx

    @property
    def dtype(self):
        return np.float64

    @property
    def H(self):
        return Hadamard(self.hilbert, self.idx)

    def to_local_operator(self):
        sq2 = np.sqrt(2)
        return (
            spin.sigmaz(self.hilbert, self.idx) + spin.sigmax(self.hilbert, self.idx)
        ) / sq2

    def __eq__(self, o):
        if isinstance(o, Hadamard):
            return o.idx == self.idx
        return False

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.idx)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    @property
    def max_conn_size(self) -> int:
        return 2

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Hadamard(xr, self.idx, self._local_states)
        xp = xp.reshape(x.shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(x.shape[:-1] + mels.shape[-1:])
        return xp, mels

    @jax.jit
    def get_conn_flattened(self, x, sections):
        xp, mels = self.get_conn_padded(x)
        sections[:] = np.arange(2, mels.size + 2, 2)

        xp = xp.reshape(-1, self.hilbert.size)
        mels = mels.reshape(
            -1,
        )
        return xp, mels


@partial(jax.vmap, in_axes=(0, None, [None, None]), out_axes=(0, 0))
def get_conns_and_mels_Hadamard(sigma, idx, local_states):
    assert sigma.ndim == 1

    state_0 = jnp.asarray(local_states[0], dtype=sigma.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=sigma.dtype)

    cons = jnp.tile(sigma, (2, 1))
    current_state = sigma[idx]
    flipped_state = jnp.where(current_state == state_0, state_1, state_0)
    cons = cons.at[1, idx].set(flipped_state)

    mels = jnp.zeros(2, dtype=float)
    mels = mels.at[1].set(1 / jnp.sqrt(2))
    state_value = cons.at[0, idx].get()
    mels_value = jnp.where(state_value == local_states[0], 1, -1) / jnp.sqrt(2)
    mels = mels.at[0].set(mels_value)

    return cons, mels
