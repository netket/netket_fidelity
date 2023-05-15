import jax
import jax.numpy as jnp


def _logpsi_diff(afun_t, afun, U, variables_t,  variables, x):
    
    xp, mels = U.get_conn_padded(x)
    logphi_xp = afun_t(variables_t, xp)
    logUphi = jax.scipy.special.logsumexp(logphi_xp, axis=-1, b=mels)
    
    logpsi = afun(variables, x)
    
    return 0.5* jnp.log(jnp.absolute(jnp.exp(logUphi) - jnp.exp(logpsi)))


def _logpsi_diff_noU(afun_t, afun, variables_t,  variables, x):
    
    logphi = afun_t(variables_t, x)
    logpsi = afun(variables, x)
    
    return 0.5* jnp.log(jnp.absolute(jnp.exp(logphi) - jnp.exp(logpsi)))