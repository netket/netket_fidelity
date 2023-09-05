from typing import Optional, Any
from functools import partial

import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.operator import AbstractOperator
from netket.utils.types import DType
from netket.utils.dispatch import TrueT


class Renyi2EntanglementEntropy(AbstractOperator):
    def __init__(
        self,
        hilbert: None, 
        subsys: jnp.array,
        *,
        dtype: Optional[DType] = None,
    ):
        
        r"""
    Operator computing the Rényi2 entanglement entropy of a state |ψ⟩ for a partition with subsystem A:

    .. math::

        S_2 = -\log_2 Tr_A ρ

    where ρ = |ψ⟩⟨ψ| is the density matrix of the system and Tr_A indicates the partial trace over the subsystem A. 

    The Monte Carlo estimator of S_2 is:

    ..math::

        S_2 = - \log \mathbb{E}_{Π(σ,η) Π(σ',η')}[ψ(σ,η') ψ(σ',η) / ψ(σ,η) ψ(σ',η')] 
        
    where σ \in A, η \in Ā and Π(σ, η) = |Ψ(σ,η)|^2 / ⟨ψ|ψ⟩. 

    Args:
        hilbert: hilbert space of the system.
        subsys: subsystem in the partition.

    Returns:
        Rényi2 operator for which computing expected value.

    Example:
    """
        
        super().__init__(hilbert)

        self._dtype = dtype
        self._subsys= np.sort(np.array(subsys))
        self._sys = jnp.arange(hilbert.size)

        if(
            self._subsys.size > self._sys.size
            or jnp.where(self._subsys < 0)[0].size > 0 
            or jnp.where(self._subsys > self._sys[-1])[0].size > 0 
        ): 

            print("Invalid partition")

    @property
    def dtype(self):
        return self._dtype

    @property
    def subsys(self):
        return self._subsys

    @property
    def sys(self):
        return self._sys

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"Renyi2EntanglementEntropy(subsys={self.subsys})"

