import jax


def _logpsi_U(apply_fun, U, variables, x):
    xp, mels = U.get_conn_padded(x)
    logpsi_xp = apply_fun(variables, xp)

    return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

