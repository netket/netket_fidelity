import jax


def sampling_Ustate(apply_fun, U, variables, x):
    xp, mels = U.get_conn_padded(x)
    logpsi_xp = apply_fun(variables, xp)

    @jax.jit
    def sampling_Ustate_inner(logpsi_xp, mels):
        return jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)

    return sampling_Ustate_inner(logpsi_xp, mels)
