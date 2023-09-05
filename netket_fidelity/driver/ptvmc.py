import jax.numpy as jnp

from .infidelity_optimizer import InfidelityOptimizer

from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)


class PTVMC:
    def __init__(
        self,
        target_state,
        U,
        vstate,
        optimizer,
        tf,
        dt,
        n_iter,
        obs=None,
        U_dagger=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        is_unitary=False,
        cv_coeff=None,
    ):
        self._dt = dt
        self._tf = tf
        self._ts = jnp.arange(0, self._tf, self._dt)
        self._n_iter = n_iter

        self._obs = obs
        self._obs_dict = {"obs": []}

        self._te = InfidelityOptimizer(
            target_state,
            optimizer,
            variational_state=vstate,
            U=U,
            U_dagger=U_dagger,
            preconditioner=preconditioner,
            is_unitary=is_unitary,
            cv_coeff=cv_coeff,
        )

    def run(self):
        for t in self._ts:
            print(f"Time t = {t}: ")
            print("##########################################")

            self._te.run(self._n_iter)
            self._te._I_op.target.parameters = self._te.state.parameters

            if self._obs is not None:
                self._obs_dict["obs"].append(self._te.state.expect(self._obs))

            print("##########################################")
            print("\n")
