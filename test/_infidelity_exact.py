import jax.numpy as jnp


def _infidelity_exact(params_new, vstate, U):
    params_old = vstate.parameters
    state_old = vstate.to_array()

    vstate.parameters = params_new
    state_new = vstate.to_array()
    vstate.parameters = params_old

    if U is None:
        return 1 - jnp.absolute(state_new.conj().T @ state_old) ** 2 / (
            (state_new.conj().T @ state_new) * (state_old.conj().T @ state_old)
        )

    else:
        return 1 - jnp.absolute(state_new.conj().T @ U.to_sparse() @ state_old) ** 2 / (
            (state_new.conj().T @ state_new) * (state_old.conj().T @ state_old)
        )
