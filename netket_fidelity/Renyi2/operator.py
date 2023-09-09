from typing import Optional

import jax.numpy as jnp
import numpy as np

from netket.operator import AbstractOperator
from netket.utils.types import DType


class Renyi2EntanglementEntropy(AbstractOperator):
    def __init__(
        self,
        hilbert: None,
        subsystem: jnp.array,
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
            subsystem: list of the indices identifying the degrees of freedom in one subsystem of the full system.
                All indices should be integers between 0 and hilbert.size

        Returns:
            Rényi2 operator for which computing expected value.

        Example:
        """

        super().__init__(hilbert)

        self._dtype = dtype
        self._subsystem = np.sort(np.array(subsystem))

        if (
            self._subsystem.size > hilbert.size
            or np.where(self._subsystem < 0)[0].size > 0
            or np.where(self._subsystem > hilbert.size)[0].size > 0
        ):

            print("Invalid partition")

    @property
    def dtype(self):
        return self._dtype

    @property
    def subsystem(self):
        r"""
        list of indices for the degrees of freedom in the subsystem
        """
        return self._subsystem

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"Renyi2EntanglementEntropy(hilbert={self.hilbert}, subsys={self.subsystem})"
