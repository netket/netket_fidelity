import netket as nk
import netket_fidelity as nkf
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax
import jax

from RBM_Jastrow_measurement import RBMJasMeas
from netket_fidelity.renyi2 import Renyi2EntanglementEntropy

# Set the parameters
L = 2
# L = 3
N = L**2
J = -1.0
h_critical = 3.044 * np.abs(J)
h = -h_critical / 8

dt = 0.1
tf = 2.0
ts = jnp.arange(0, tf, dt)

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, N)
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=False)

sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = RBMJasMeas(alpha=1, param_dtype=complex)

phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)

# Choose the number of iterations, the learning rate and the optimizer
n_iter = 300
lr = 0.01
optimizer = nk.optimizer.Adam(learning_rate=lr)

# Prepare the initial state \bigotimes_{i=}^{N} |+⟩_i
with open("initial_state_2x2.mpack", "rb") as file:
    phi.variables = flax.serialization.from_bytes(phi.variables, file.read())

with open("initial_state_2x2.mpack", "rb") as file:
    psi.variables = flax.serialization.from_bytes(psi.variables, file.read())

"""
with open("initial_state_3x3.mpack", "rb") as file:
    phi.variables = flax.serialization.from_bytes(phi.variables, file.read())

with open("initial_state_3x3.mpack", "rb") as file:
    psi.variables = flax.serialization.from_bytes(psi.variables, file.read())
"""

# Create the X-rotations in the Ising propagator
Uxs = []
Uxs_dagger = []

for i in range(N):
    Uxs.append(nkf.operator.Rx(hi, i, 2 * dt * h))
    Uxs_dagger.append(nkf.operator.Rx(hi, i, 2 * dt * h))

# Create the random keys for performing the random measurements
key_meas = jax.random.PRNGKey(seed=1234)
key_spin = jax.random.PRNGKey(seed=5678)

# Instantiate the Renyi2 entropy to monitor
subsys = [x for x in range(N // 2)]
S2op = Renyi2EntanglementEntropy(hi, subsys)


# Compute the probabilities for the measurement outcomes of a spin
def probabilities_measurement(vstate, spin):
    sigma = vstate.samples
    sigma = sigma.reshape((-1, vstate.hilbert.size))

    n_samples = sigma.shape[0]

    sigma_ = sigma[:, spin]
    sigma_ = (sigma_ + 1) / 2

    prob_up = jnp.sum(sigma_) / n_samples
    prob_down = 1 - prob_up

    return prob_down, prob_up


# Perform the projective measurement exactly
def projective_measurement(phi, psi, p, key_meas, key_spin):
    for i in range(psi.hilbert.size):
        key_meas, subkey_meas = jax.random.split(key_meas)

        if jax.random.uniform(subkey_meas) < p:
            print("Measurement!")
            prob_down, prob_up = probabilities_measurement(psi, i)

            key_spin, subkey_spin = jax.random.split(key_spin)

            params = flax.core.unfreeze(psi.parameters)
            params = jax.tree_map(lambda x: jnp.array(x), params)
            if jax.random.uniform(subkey_spin) < prob_up.real:
                params["orbital_down"] = params["orbital_down"].at[i].set(1e-12)
            else:
                params["orbital_up"] = params["orbital_up"].at[i].set(1e-12)
            psi.parameters = params
            phi.parameters = params

    print("\n")

    return phi, psi, key_meas, key_spin


def dynamics_with_measurements(
    phi, optimizer, psi, J, Uxs, Uxs_dagger, ts, n_iter, p, S2op, key_meas, key_spin
):
    obs_dict = {"S2": []}

    for t in ts:
        print(f"Time t = {t}: ")
        print("##########################################")

        # Projective measurement
        if t > 0.0:
            phi, psi, key_meas, key_spin = projective_measurement(
                phi, psi, p, key_meas, key_spin
            )

        print("\n Unitary dynamics: \n")

        # ZZ diagonal term
        params = flax.core.unfreeze(psi.parameters)
        params = jax.tree_map(lambda x: jnp.array(x), params)
        for l, m in g.edges():
            params["theta_zz"] = (
                params["theta_zz"]
                .at[l, m]
                .set(params["theta_zz"].at[l, m].get() - J * dt / 4)
            )
        psi.parameters = params
        phi.parameters = params

        # X terms
        for i in range(len(Uxs)):
            te = nkf.driver.InfidelityOptimizer(
                phi,
                optimizer,
                U=Uxs[i],
                U_dagger=Uxs_dagger[i],
                variational_state=psi,
                is_unitary=True,
                cv_coeff=-0.5,
            )
            te.run(n_iter=n_iter)
            phi.parameters = psi.parameters

        # ZZ diagonal term
        params = flax.core.unfreeze(psi.parameters)
        params = jax.tree_map(lambda x: jnp.array(x), params)
        for l, m in g.edges():
            params["theta_zz"] = (
                params["theta_zz"]
                .at[l, m]
                .set(params["theta_zz"].at[l, m].get() - J * dt / 4)
            )
        psi.parameters = params
        phi.parameters = params

        obs_dict["S2"].append(psi.expect(S2op))

        print("\n")

    return psi, obs_dict


# Run the evolution
p = 0.1
psi, obs_dict = dynamics_with_measurements(
    phi, optimizer, psi, J, Uxs, Uxs_dagger, ts, n_iter, p, S2op, key_meas, key_spin
)

obs_mean = np.array([x.mean for x in obs_dict["S2"]])
obs_error = np.array([x.error_of_mean for x in obs_dict["S2"]])

# Plot the results
fig = plt.figure(figsize=(8, 8))
plt.errorbar(ts, obs_mean, obs_error)
plt.xlabel(r"$t$")
plt.ylabel(r"$S_2$")
plt.legend()
plt.tight_layout()
plt.savefig("S2_dynamics_with_measurements.pdf", bbox_inches="tight")
plt.show()
