from typing import Optional
from netket.stats import Stats

from netket.driver.abstract_variational_driver import AbstractVariationalDriver
from netket.callbacks import ConvergenceStopping

from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)

from netket_fidelity.infidelity import InfidelityOperator

from .infidelity_optimizer_common import info


def _to_tuple(maybe_iterable):
    """
    _to_tuple(maybe_iterable)

    Ensure the result is iterable. If the input is not iterable, it is wrapped into a tuple.
    """
    if hasattr(maybe_iterable, "__iter__"):
        surely_iterable = tuple(maybe_iterable)
    else:
        surely_iterable = (maybe_iterable,)

    return surely_iterable


class InfidelityOptimizer(AbstractVariationalDriver):
    def __init__(
        self,
        target_state,
        optimizer,
        *,
        variational_state,
        U=None,
        U_dagger=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        is_unitary=False,
        sample_Upsi=False, 
        cv_coeff=-0.5,
    ):
        r"""
        Constructs a driver training the state to match the target state.

        The target state is either `math`:\ket{\psi}` or `math`:\hat{U}\ket{\psi}`
        depending on the provided inputs.

        Operator I_op computing the infidelity I among two variational states |ψ⟩ and |Φ⟩ as:

        .. math::

            I = 1 - |⟨ψ|Φ⟩|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩ = 1 - ⟨ψ|I_op|ψ⟩ / ⟨ψ|ψ⟩

        where:

         .. math::

            I_op = |Φ⟩⟨Φ| / ⟨Φ|Φ⟩

        The state |Φ⟩ can be an autonomous state |Φ⟩ =|ϕ⟩ or an operator U applied to it, namely
        |Φ⟩  = U|ϕ⟩. I_op is defined by the state |ϕ⟩ (called target) and, possibly, by the operator U.
        If U is not passed, it is assumed |Φ⟩ =|ϕ⟩.

        The Monte Carlo estimator of I is:

        ..math::

            I = \mathbb{E}_{χ}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|Φ⟩ ⟨η|ψ⟩ / ⟨σ|ψ⟩ ⟨η|Φ⟩ ]

        where χ(σ, η) = |Ψ(σ)|^2 |Φ(η)|^2 / ⟨ψ|ψ⟩ ⟨Φ|Φ⟩. In practice, since I is a real quantity, Re{I_loc(σ,η)}
        is used. This estimator can be utilized both when |Φ⟩ =|ϕ⟩ and when |Φ⟩ = U|ϕ⟩, with U a (unitary or
        non-unitary) operator. In the second case, we have to sample from U|ϕ⟩ and this is implemented in
        the function :ref:`jax.:ref:`InfidelityUPsi`. This works only with the operators provdided in the package.
        We remark that sampling from U|ϕ⟩ requires to compute connected elements of U and so is more expensive
        than sampling from an autonomous state. The choice of this estimator is specified by passing
        `sample_Upsi=True`, while the flag argument `is_unitary` indicates whether U is unitary or not.

        If U is unitary, the following alternative estimator can be used:

        ..math::

            I = \mathbb{E}_{χ'}[ I_loc(σ,η) ] = \mathbb{E}_{χ}[ ⟨σ|U|ϕ⟩ ⟨η|ψ⟩ / ⟨σ|U^{\dagger}|ψ⟩ ⟨η|ϕ⟩ ].

        where χ'(σ, η) = |Ψ(σ)|^2 |ϕ(η)|^2 / ⟨ψ|ψ⟩ ⟨ϕ|ϕ⟩. This estimator is more efficient since it does not
        require to sample from U|ϕ⟩, but only from |ϕ⟩. This choice of the estimator is the default and it works only
        with `is_unitary==True` (besides `sample_Upsi=False`). When |Φ⟩ = |ϕ⟩ the two estimators coincides.

        To reduce the variance of the estimator, the Control Variates (CV) method can be applied. This consists
        in modifying the estimator into:

        ..math::

            I_loc^{CV} = Re{I_loc(σ,η)} - c (|1 - I_loc(σ,η)^2| - 1)

        where c ∈ \mathbb{R}. The constant c is chosen to minimize the variance of I_loc^{CV} as:

        ..math::

            c* = Cov_{χ}[ |1-I_loc|^2, Re{1-I_loc}] / Var_{χ}[ |1-I_loc|^2 ],

        where Cov[..., ...] indicates the covariance and Var[...] the variance. In the relevant limit
        |Ψ⟩ →|Φ⟩, we have c*→-1/2. The value -1/2 is adopted as default value for c in the infidelity
        estimator. To not apply CV, set c=0.

        Args:
            target_state: target variational state |ϕ⟩.
            optimizer: the optimizer to use to use (from optax)
            variational_state: the variational state to train
            U: operator U.
            U_dagger: dagger operator U^{\dagger}.
            cv_coeff: Control Variates coefficient c.
            is_unitary: flag specifiying the unitarity of U. If True with `sample_Upsi=False`, the second estimator is used.
            dtype: The dtype of the output of expectation value and gradient.
            sample_Upsi: flag specifiying whether to sample from |ϕ⟩ or from U|ϕ⟩. If False with `is_unitary=False`, an error occurs.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        self._cv = cv_coeff

        self._preconditioner = preconditioner

        self._I_op = InfidelityOperator(
            target_state,
            U=U,
            U_dagger=U_dagger,
            is_unitary=is_unitary,
            cv_coeff=cv_coeff,
            sample_Upsi=sample_Upsi, 
        )

    def _forward_and_backward(self):
        self.state.reset()
        self._I_op.target.reset()

        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._I_op)

        # if it's the identity it does
        self._dp = self.preconditioner(self.state, self._loss_grad, self.step_count)

        return self._dp

    def run(
        self,
        n_iter,
        out=None,
        *args,
        target_infidelity=None,
        callback=lambda *x: True,
        **kwargs,
    ):
        """
        Executes the Infidelity optimisation, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `out`,
        overwriting files with the same prefix.

        Args:
            n_iter: the total number of iterations
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
            obs: An iterable containing all observables that should be computed
            target_infidelity: An optional floating point number that specifies when to stop the optimisation.
                This is used to construct a {class}`netket.callbacks.ConvergenceStopping` callback that stops
                the optimisation when that value is reached. You can also build that object manually for more
                control on the stopping criteria.
            step_size: Every how many steps should observables be logged to disk (default=1)
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to stop training given a condition
        """
        callbacks = _to_tuple(callback)

        if target_infidelity is not None:
            cb = ConvergenceStopping(
                target_infidelity, smoothing_window=20, patience=30
            )
            callbacks = callbacks + (cb,)

        super().run(n_iter, out, *args, callback=callbacks, **kwargs)

    @property
    def cv(self) -> Optional[float]:
        """
        Return the coefficient for the Control Variates
        """
        return self._cv

    @property
    def infidelity(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.

        Often, this is taken to be :func:`nk.optimizer.SR`. If it is set to
        `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: Optional[PreconditionerT]):
        if val is None:
            val = identity_preconditioner

        self._preconditioner = val

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
