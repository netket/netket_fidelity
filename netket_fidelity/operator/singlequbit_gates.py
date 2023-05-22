from functools import partial
import jax.numpy as jnp
import numpy as np
import jax
from jax.tree_util import register_pytree_node_class
import netket as nk
from netket.operator import DiscreteOperator


@register_pytree_node_class
class Rx(DiscreteOperator):
    def __init__(self, hi, idx, angle):
        super().__init__(hi)
        self.idx = idx
        self.angle = angle / 2

    @property
    def dtype(self):
        return complex

    @property
    def H(self):
        return Rx(self.hilbert, self.idx, -self.angle * 2)

    def __hash__(self):
        return hash(("Rx", self.idx))

    def __eq__(self, o):
        if isinstance(o, Rx):
            return o.idx == self.idx and o.angle == self.angle
        return False

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.idx, self.angle)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

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


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Rx(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle))
    mels = mels.at[1].set(-1j * jnp.sin(angle))

    return conns, mels


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Rx):
    sigma = vstate.samples
    conns, mels = get_conns_and_mels_Rx(
        sigma.reshape(-1, vstate.hilbert.size), op.idx, op.angle
    )

    conns = conns.reshape((sigma.shape[0], sigma.shape[1], -1, vstate.hilbert.size))
    mels = mels.reshape((sigma.shape[0], sigma.shape[1], -1))

    return sigma, (conns, mels)


@register_pytree_node_class
class Ry(DiscreteOperator):
    def __init__(self, hi, idx, angle):
        super().__init__(hi)
        self.idx = idx
        self.angle = angle / 2

    @property
    def dtype(self):
        return complex

    @property
    def H(self):
        return Ry(self.hilbert, self.idx, -self.angle * 2)

    def __hash__(self):
        return hash(("Ry", self.idx))

    def __eq__(self, o):
        if isinstance(o, Ry):
            return o.idx == self.idx and o.angle == self.angle
        return False

    def tree_flatten(self):
        children = ()
        aux_data = (self.hilbert, self.idx, self.angle)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

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


@partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, 0))
def get_conns_and_mels_Ry(sigma, idx, angle):
    assert sigma.ndim == 1

    conns = jnp.tile(sigma, (2, 1))
    conns = conns.at[1, idx].set(-conns.at[1, idx].get())

    mels = jnp.zeros(2, dtype=complex)
    mels = mels.at[0].set(jnp.cos(angle))
    mels = mels.at[1].set((-1) ** ((conns.at[0, idx].get() + 1) / 2) * jnp.sin(angle))

    return conns, mels


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Ry):
    sigma = vstate.samples
    conns, mels = get_conns_and_mels_Ry(
        sigma.reshape(-1, vstate.hilbert.size), op.idx, op.angle
    )

    conns = conns.reshape((sigma.shape[0], sigma.shape[1], -1, vstate.hilbert.size))
    mels = mels.reshape((sigma.shape[0], sigma.shape[1], -1))

    return sigma, (conns, mels)


@register_pytree_node_class
class Hadamard(DiscreteOperator):
    def __init__(self, hi, idx):
        super().__init__(hi)
        self.idx = idx

    @property
    def dtype(self):
        return np.float64

    @property
    def H(self):
        return Hadamard(self.hilbert, self.idx)

    def __hash__(self):
        return hash(("Hadamard", self.idx))

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

    @jax.jit
    def get_conn_padded(self, x):
        xr = x.reshape(-1, x.shape[-1])
        xp, mels = get_conns_and_mels_Hadamard(xr, self.idx)
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


@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
def get_conns_and_mels_Hadamard(sigma, idx):
    assert sigma.ndim == 1

    cons = jnp.tile(sigma, (2, 1))
    cons = cons.at[1, idx].set(-cons.at[1, idx].get())

    mels = jnp.zeros(2, dtype=float)
    mels = mels.at[1].set(1 / jnp.sqrt(2))
    mels = mels.at[0].set(((-1) ** ((cons.at[0, idx].get() + 1) / 2)) / jnp.sqrt(2))

    return cons, mels


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: Hadamard):
    sigma = vstate.samples
    conns, mels = get_conns_and_mels_Hadamard(
        sigma.reshape(-1, vstate.hilbert.size),
        op.idx,
    )

    conns = conns.reshape((sigma.shape[0], sigma.shape[1], -1, vstate.hilbert.size))
    mels = mels.reshape((sigma.shape[0], sigma.shape[1], -1))

    return sigma, (conns, mels)
