from functools import partial

import jax.numpy as jnp
import jax
import netket as nk
import qutip
import numpy as np

from netket.utils.dispatch import TrueT
from netket.vqs import FullSumState, expect
from netket.stats import statistics as mpi_statistics

from .operator import Renyi2Operator


@expect.dispatch
def Renyi2(vstate: FullSumState, op: Renyi2Operator):
    
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return Renyi2_sampling_FullSumState(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,   
        op.subsys,
    )



#@partial(jax.jit, static_argnames=("afun"))
def Renyi2_sampling_FullSumState(
    afun,
    params,
    model_state,
    sigma,
    subsys, 
):

    N = sigma.shape[-1]
    
    state = jnp.exp(afun({"params": params, **model_state}, sigma))
    state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))
    
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