# Custom implementation guidelines

`pySDC` solves (non-)linear ODE of the form

$$
\frac{du}{dt} = f(u, t), \quad u(0)=u_0, \quad t \in [0, T].
$$

where $u$ a vector or scalar, representing one or more variables.

The type of variable, the definition of the right-hand side $f(u,t)$ and initial solution value $u_0$ are defined in a given `Problem` class.

... to be continued ...

## Implementing a custom problem class

Any problem class inherit from the same base class, that is (currently) the `ptype` class from the module `pySDC.code.Problem`.
Each custom problem should inherit from this base class, like for the following example template :

- right-hand side $f(u,t)=\lambda u +  ct$ with
    - $\lambda$ one or more complex values (in vector form)
    - $c$ one scalar value

```python
import numpy as np

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ProblemError
from pySDC.implementations.datatype_classes.mesh import mesh

class MyCustomProblem(ptype):
    # 1) Provide datatype class as class attribute
    dtype_u = mesh  # -> used for u values
    dtype_f = mesh  # -> used for f(u,t) values

    # 2) Define constructor
    def __init__(self, lam, c):

        # Store lambda values into a numpy array (with copy) + check
        lam = np.array(lam)
        if len(lam.shape) > 1:
            raise ProblemError(f"lambda values must be given as 1D vector, got shape {lam.shape}")

        # Call base constructor
        super().__init__(init=(lam.size, None, lam.dtype))

        # Register parameters
        self._makeAttributeAndRegister('lam', 'c', localVars=locals(), readOnly=True)

    # 3) Define RHS function
    def eval_f(self, u, t):
        f = self.f_init  # Generate new datatype to store result
        f[:] = self.lam*u + self.c*t  # Compute RHS value
        return f
```

:bell: The `_makeAttributeAndRegister` automatically add `lam` and `c`, and register them in a list of parameters that are printed in the outputs of `pySDC`.
If you set `readOnly=True`, then those parameters cannot be changed after initializing the problem (if not specifies, use `readOnly=False`).

:arrow_left: [Back to Naming Conventions](./03_naming_conventions.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to Documenting Code](./05_documenting_code.md)