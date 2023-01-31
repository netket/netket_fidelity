import jax.numpy as jnp
import netket as nk

def _infidelity_exact(params_new, vstate, U_sparse):
    params_old = vstate.parameters
    state_old = vstate.to_array()

    vstate.parameters = params_new
    state_new = vstate.to_array()
    vstate.parameters = params_old

    return 1 - jnp.square(
        jnp.absolute(state_new.conj().T @ U_sparse @ state_old)
    ) / ((state_new.conj().T @ state_new) * (state_old.conj().T @ state_old))

