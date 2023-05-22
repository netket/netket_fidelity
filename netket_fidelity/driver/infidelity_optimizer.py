from netket.stats import Stats

from netket.driver.abstract_variational_driver import AbstractVariationalDriver

from .infidelity_optimizer_common import info
from netket_fidelity.infidelity import InfidelityOperator


class InfidelityOptimizer(AbstractVariationalDriver):
    def __init__(
        self,
        target_state,
        U,
        vstate,
        optimizer,
        U_dagger=None,
        sr=None,
        is_unitary=False,
        cv_coeff=None,
    ):
        super().__init__(vstate, optimizer, minimized_quantity_name="Infidelity")

        self.sr = sr
        self._I_op = InfidelityOperator(
            target_state, U=U, U_dagger=U, is_unitary=True, cv_coeff=-1 / 2
        )

    def _forward_and_backward(self):
        self.state.reset()
        self._I_op.target.reset()

        I_stats, I_grad = self.state.expect_and_grad(self._I_op)

        # TODO
        self._loss_stats = I_stats
        self._loss_grad = I_grad

        if self.sr is not None:
            self._S = self.state.quantum_geometric_tensor(self.sr)
            self._dp = self._S(self._loss_grad)
        else:
            self._dp = self._loss_grad

        return self._dp

    @property
    def infidelity(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "InfidelityOptimiser("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Propagator    ", self._I_op.U),
                ("Optimizer      ", self._optimizer),
                ("Preconditioner ", self.preconditioner),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)

    def info(self):
        pass
