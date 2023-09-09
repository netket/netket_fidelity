import netket as nk
import netket_fidelity as nkf
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import flax
import numpy as np

from GHZ_state import GHZ
from RBM_Jastrow_ansatz import RBM_Jastrow

# Set the parameters
N = 2
h = -1.0
J = -1.0

tf = 2.0
dt = 0.01
ts = jnp.arange(0, tf, dt)

# Create the Hilbert space and the variational states |ψ⟩, |ϕ⟩ and |GHZ⟩
hi = nk.hilbert.Spin(0.5, N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
GHZ_vstate = nk.vqs.MCState(sampler=sampler, model=GHZ(), n_samples=1000)

model = RBM_Jastrow(alpha=1, param_dtype=complex)
phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)

# Choose the number of iterations, the learning rate and the optimizer
n_iter = 1000
lr = 0.01
optimizer = nk.optimizer.Adam(learning_rate=lr)

# Prepare the |GHZ⟩ on the RBM
print("Preparing the initial state: ")
te = nkf.driver.InfidelityOptimizer(
    GHZ_vstate, optimizer, variational_state=phi, cv_coeff=-0.5
)
te.run(n_iter=n_iter)
psi.parameters = phi.parameters

# Create the X-rotations
Uxs = []
Uxs_dagger = []

for i in range(N):
    Uxs.append(nkf.operator.Rx(hi, i, 2 * dt * h))
    Uxs_dagger.append(nkf.operator.Rx(hi, i, -2 * dt * h))

# Instantiate the observable to monitor
obs = nk.operator.spin.sigmaz(hi, 0) * nk.operator.spin.sigmaz(hi, 1)

# Function doing the TFI dynamics with Trotterized p-tVMC
def Trotter_Ising(phi, optimizer, psi, Uxs, Uxs_dagger, J, ts, n_iter, obs=None):

    if obs is not None:
        obs_dict = {"obs": []}

    for t in ts:

        if obs is not None:
            obs_dict["obs"].append(phi.expect(obs))

        print(f"Time t = {t}: ")
        print("##########################################")

        params = flax.core.unfreeze(phi.parameters)
        for (l, m) in g.edges():
            params["theta_zz"] = (
                params["theta_zz"]
                .at[l, m]
                .set(params["theta_zz"].at[l, m].get() - J * dt / 2)
            )
        psi.parameters = params
        phi.parameters = params

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

        params = flax.core.unfreeze(phi.parameters)
        for (l, m) in g.edges():
            params["theta_zz"] = (
                params["theta_zz"]
                .at[l, m]
                .set(params["theta_zz"].at[l, m].get() - J * dt / 2)
            )
        psi.parameters = params
        phi.parameters = params

        print("##########################################")
        print("\n")

    if obs is not None:
        return psi, obs_dict

    else:
        return psi


# Run the evolution
psi, obs_dict = Trotter_Ising(
    phi, optimizer, psi, Uxs, Uxs_dagger, J, ts, n_iter=n_iter, obs=obs
)

obs_mean = np.array([x.mean for x in obs_dict["obs"]])
obs_error = np.array([x.error_of_mean for x in obs_dict["obs"]])

# Plot the results
fig = plt.figure(figsize=(8, 8))
plt.errorbar(ts, obs_mean, obs_error)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle \sigma_1^z \sigma_2^z \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig("twospins_GHZ.pdf", bbox_inches="tight")
plt.show()
