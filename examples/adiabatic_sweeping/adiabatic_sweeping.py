import netket as nk
import netket_fidelity as nkf
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax

# Set the parameters
N = 10
Γ = -1.0
γ = 1.0

dt = 0.05
tf = 15
ts = jnp.arange(0, tf, dt)
T = 8

ts = jnp.arange(0, tf, dt)

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, N)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = nk.models.RBM(alpha=1, param_dtype=complex)

phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)

# Choose the number of iterations, the learning rate and the optimizer
n_iter = 1000
lr = 0.01
optimizer = nk.optimizer.Adam(learning_rate=lr)

# Prepare the initial state \bigotimes_{i=}^{N} |+⟩_i
with open("initial_state.mpack", "rb") as file:
    phi.variables = flax.serialization.from_bytes(phi.variables, file.read())

with open("initial_state.mpack", "rb") as file:
    psi.variables = flax.serialization.from_bytes(psi.variables, file.read())

# Instantiate the observable to monitor
obs = sum([nk.operator.spin.sigmaz(hi, i) for i in range(N)]) / N

# Function doing the adiabtic dynamics with Trotterized p-tVMC
def Trotter_adiabatic(phi, optimizer, psi, γ, Γ, ts, n_iter, obs=None):

    if obs is not None:
        obs_dict = {"obs": []}

    for t in ts:
        print(f"Time t = {t}: ")
        print("##########################################")

        # Calculate the time-dependent couplings
        if t < T:
            γt = γ * t / T
            Γt = Γ * (1 - t / T)
        else:
            γt = γ * (1 - (t - T) / T)
            Γt = Γ * (t - T) / T

        params = flax.core.unfreeze(phi.parameters)
        params["visible_bias"] = params["visible_bias"] + 1j * γt * dt / 2
        psi.parameters = params
        phi.parameters = params

        # Create the X-rotations
        Uxs = []
        Uxs_dagger = []

        for i in range(N):
            Uxs.append(nkf.operator.Rx(hi, i, 2 * dt * Γt))
            Uxs_dagger.append(nkf.operator.Rx(hi, i, -2 * dt * Γt))

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

        params = flax.core.unfreeze(phi.parameters)
        params["visible_bias"] = params["visible_bias"] + 1j * γt * dt / 2
        psi.parameters = params
        phi.parameters = params

        if obs is not None:
            obs_dict["obs"].append(psi.expect(obs))

        print("##########################################")
        print("\n")

    if obs is not None:
        return psi, obs_dict

    else:
        return psi


# Run the evolution
psi, obs_dict = Trotter_adiabatic(phi, optimizer, psi, γ, Γ, ts, n_iter=n_iter, obs=obs)

# Plot the results
fig = plt.figure(figsize=(8, 8))
plt.errorbar(ts, obs_dict["obs"].mean, obs_dict["obs"].error_of_mean)
plt.xlabel(r"Time $t$")
plt.ylabel(r"$\langle \sigma_i^z \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig("adiabatic_sweeping.pdf", bbox_inches="tight")
plt.show()
