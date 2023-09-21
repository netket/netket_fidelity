# projected time-dependent Variational Monte Carlo (p-tVMC)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8344170.svg)](https://doi.org/10.5281/zenodo.8344170)
![Tests status](https://github.com/netket/netket_fidelity/actions/workflows/CI.yml/badge.svg)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)

This is a package for the projected time-dependent Variational Monte Carlo (p-tVMC) method based on infidelity optimization for variational simulation of quantum dynamics. 
See the paper "Unbiasing time-dependent Variational Monte Carlo with projected quantum evolution" for reference. 

The p-tVMC can be used to simulate the evolution generated by an arbitrary transformation U, by iteratively minimizing the infidelity among the variational ansatz with free parameters |ψ⟩ and the state U|ϕ⟩ where U is an arbitrary transformation and |ϕ⟩ is a known state (such as an ansatz with known parameters). 
There are no restrictions on U from the moment a function that computes its connected elements is implemented.
The package supports the possibility to sample from the states |ψ⟩ and U|ϕ⟩, which can be used for any transformation U (unitary and non-unitary), and to sample from the states |ψ⟩ and |ϕ⟩, which is possible only for a unitary U (exploiting the norm conservation).
To sample from U|ϕ⟩ a `jax` compatible operator for U must be used, and the package exports few examples of them (the Ising Transverse Field Ising Hamiltonian, Rx and Ry single qubit rotations and the Hadamard gate). 
In addition, the code includes the possibility to use the Control Variates (CV) correction on the infidelity stochastic estimator to improve its signal to noise ratio and reduce the sampling overhead by orders of magnitudes.

## Content of the repository

- **netket_fidelity** : folder containing the following several subfolders: 
    - **infidelity**: contains the infidelity operator. 
    - **operator**: contains the `jax`-compatible operators for U.
    - **driver**: contains the driver for infidelity optimization. 
- **examples**: folder containing some examples of application. 
- **test**: folder containing tests for the infidelity stochastic estimation and for the `jax`-compatible rotation operators.

## Installation

This package is not registered on PyPi, so you must install it directly from GitHub.
You can install either:
 
 - The latest version of the code available on GitHub, which might or might not work at the moment (in case it does not work, do open an issue with us). To do so, run the following line in your commandline:

```bash
pip install git+https://github.com/netket/netket_fidelity
```

- The version corresponding to the revised version of the manuscript we submitted on the ArXiV/Quantum Journal (September 2023). 

```bash
pip install "git+https://github.com/netket/netket_fidelity@v0.0.2"
```
- You can download this repository and install it manually in editable mode

```bash
git clone https://github.com/netket/netket_fidelity
pip install -e ./netket_fidelity
```
 
## Example of usage

```python
import netket as nk
import netket_fidelity as nkf

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, 4)
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = nk.models.RBM(alpha=1, param_dtype=complex, use_visible_bias=False)
phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)

# Transformation U
U = nkf.operator.Hadamard(hi, 0)

# Instantiate the operator to optimize the infidelity with U|ϕ⟩ by sampling from |ψ⟩ and |ϕ⟩
I_op = nkf.infidelity.InfidelityOperator(phi, U=U, U_dagger=U, is_unitary=True, cv_coeff=-1/2)

# Create the driver
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
te =  nkf.driver.infidelity_optimizer.InfidelityOptimizer(phi, optimizer, U=U, U_dagger=U, variational_state=psi, is_unitary=True, cv_coeff=-0.5)

# Run the driver
te.run(n_iter=100)
```

How to cite
-----------

If you use ``netket_fidelity`` in your work, please consider citing it as:

::

@software{netket_fidelity,
  author = {Sinibaldi, Alessandro and Vicentini, Filippo},
  title = {netket\_fidelity package},
  url = {https://github.com/netket/netket_fidelity},
  doi = {10.5281/zenodo.8344170},
  version = {0.0.2},
  year = {2023}
}