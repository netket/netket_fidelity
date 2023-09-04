import numpy as np
import jax.numpy as jnp
import qutip

def _Renyi2_exact(vstate, subsys):
    sigma = vstate.hilbert.all_states()
    N = sigma.shape[-1]

    state = vstate.to_array()
    state_qutip = qutip.Qobj(np.array(state))

    state_qutip.dims = [[2] * N, [1] * N]

    mask = np.zeros(N, dtype=bool)

    if(len(subsys) == mask.size or len(subsys) == 0):
        return 0

    else:
        mask[subsys] = True

        rdm = state_qutip.ptrace(np.arange(N)[mask])

        n = 2
        out = np.log2(np.trace(np.linalg.matrix_power(rdm, n))) / (1 - n)

        return np.absolute(out.real)