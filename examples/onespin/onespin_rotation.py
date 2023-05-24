import netket as nk
import netket_fidelity as nkf
import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
import flax

from onespin_ansatz import BlochSphere_1spin

# Set the parameters 
N = 1 
hy = 1.

tf = 2.0
dt = 0.01
ts = jnp.arange(0, tf, dt)

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, N)
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = BlochSphere_1spin()
phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)

# Set the initial state to be |+⟩
params = flax.core.unfreeze(phi.parameters)
params["theta"] = jnp.array([jnp.pi/4])
params["phi"] = jnp.array([0.])

phi.parameters = params
psi.parameters = params

# Create the transformation U
U = nkf.operator.Ry(hi, 0, -2*dt*hy)
U_dagger = nkf.operator.Ry(hi, 0, 2*dt*hy)

# Instantiate the observable to monitor
obs = nk.operator.spin.sigmaz(hi, 0)

# Choose the number of iterations, the learning rate and the optimizer
n_iter = 1000
lr = 0.01
optimizer = nk.optimizer.Adam(learning_rate=lr)

# Create the p-tVMC driver
te_ptvmc =  nkf.driver.ptvmc.ptvmc(phi, U, psi, optimizer, tf, dt, n_iter, obs=obs, U_dagger=U_dagger, is_unitary=True, cv_coeff=-0.5)

# Run the driver 
te_ptvmc.run()

# Plot the results 
fig = plt.figure(figsize=(8, 8))
plt.errorbar(ts, te_ptvmc._te.obs_dict['obs'].mean, te_ptvmc._te.obs_dict['obs'].error_of_mean)
plt.xlabel(r"Time $t$")
plt.ylabel(r"$\langle \sigma^z \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig("onespin_rotation.pdf", bbox_inches='tight')
plt.show()