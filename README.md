This is a package for the infidelity operator to use in NetKet. It supports the use of a
unitary/non-unitary gate and the control variates method on the infidelity estimator. 

python
```
import netket_fidelity as nkf


# To optimise the overlap with another state
I_op = nkf.InfidelityOperator(target_vstate)

# To optimise the overlap with U|psi> by sampling from psi and psi_target
U = nkf.operator.Rx(0.3)
I_op = nkf.InfidelityOperator(target_vstate, U=U, is_unitary=True)

# To optimise the overlap with U|psi> by sampling from psi and Upsi
I_op = nkf.InfidelityOperator(target_vstate, U=U, sample_Upsi=True)
```

The package also exports a few special operators that are jax compatible and work well with the infidelityoperator. Those are needed for some options.