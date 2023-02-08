This is a package for the infidelity operator to use in NetKet. It supports the use of a
unitary/non-unitary gate and the control variates method on the infidelity estimator. 

python
```
import netket as nk
import netket_fidelity as nkf

hi = nk.hilbert.Spin(0.5, 4)
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = nk.models.RBM(alpha=1, param_dtype=complex)
target_vstate = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)

# To optimise the overlap with |ϕ⟩
I_op = nkf.InfidelityOperator(target_vstate)

# To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and |ϕ⟩
U = nkf.operator.Rx(0.3)
I_op = nkf.InfidelityOperator(target_vstate, U=U, is_unitary=True)

# To optimise the overlap with U|ϕ⟩ by sampling from |ψ⟩ and U|ϕ⟩
I_op = nkf.InfidelityOperator(target_vstate, U=U, sample_Upsi=True)
```

The package also exports a few special operators that are jax compatible and work well with the infidelityoperator. Those are needed for some options.
