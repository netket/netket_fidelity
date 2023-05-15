import jax.numpy as jnp


def _L2L1_exact(params_new, vstate, U):
    params_old = vstate.parameters
    state_old = vstate.to_array(normalize=False)

    vstate.parameters = params_new
    state_new = vstate.to_array(normalize=False)
    vstate.parameters = params_old

    if U is None:      
        return jnp.sum(jnp.absolute(state_new - state_old)**2) / jnp.sum(jnp.absolute(state_new - state_old))
  
    else: 
        return jnp.sum(jnp.absolute(state_new - U@state_old)**2) / jnp.sum(jnp.absolute(state_new - U@state_old))

