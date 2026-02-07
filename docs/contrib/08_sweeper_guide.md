# Sweeper Implementations Guide

This guide provides comprehensive documentation for all sweeper implementations in pySDC, explaining their purpose, parameters, and usage.

## Overview

Sweepers are the core components in pySDC that define how collocation nodes are updated during the SDC iteration. They implement different time integration methods and are responsible for:

- Computing integrals over collocation nodes
- Updating node values during SDC sweeps
- Managing integration matrices (QI, QE, etc.)
- Handling problem-specific solution strategies

All sweepers inherit from the base `Sweeper` class in `pySDC.core.sweeper` and must implement at least:
- `integrate()`: Computes the integral of the right-hand side
- `update_nodes()`: Updates values at collocation nodes (corresponds to one SDC sweep)

## Integration Matrices

Sweepers use **QDelta matrices** (denoted as QI, QE, etc.) to define how right-hand side evaluations contribute to the solution at each node. These matrices are:

- **QI (Implicit)**: Lower triangular matrix with zeros on the first column - used for implicit terms
- **QE (Explicit)**: Strictly lower triangular matrix - used for explicit terms
- **Q (Full)**: The full collocation matrix from `self.coll.Qmat`

Common QDelta types:
- `'IE'`: Implicit Euler
- `'EE'`: Explicit Euler  
- `'LU'`: LU decomposition
- `'MIN'`: Minimal correction
- And many more (see [qmat documentation](https://qmat.readthedocs.io))

## Available Sweeper Implementations

### 1. generic_implicit

**File**: `pySDC/implementations/sweeper_classes/generic_implicit.py`

**Description**: The foundational implicit sweeper for SDC methods. Uses a single lower triangular integration matrix.

**When to Use**:
- Standard SDC with implicit time integration
- When you want full control over the integration matrix
- As a base for understanding how SDC works

**Key Parameters**:
- `QI`: Type of implicit integration matrix (default: `'IE'` for Implicit Euler)
- `num_nodes`: Number of collocation nodes (required)
- `quad_type`: Quadrature type (e.g., `'RADAU-RIGHT'`, `'LOBATTO'`, `'GAUSS'`)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'QI': 'LU'  # or 'IE', 'MIN', etc.
}

description = {
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    # ... other parameters
}
```

**Key Attributes**:
- `self.QI`: The lower triangular implicit integration matrix

---

### 2. explicit

**File**: `pySDC/implementations/sweeper_classes/explicit.py`

**Description**: Explicit-only sweeper using explicit Euler as the base integrator.

**When to Use**:
- Non-stiff problems where explicit methods are stable
- Fast computations without implicit solves
- Problems where CFL conditions are easily satisfied

**Key Parameters**:
- `QE`: Type of explicit integration matrix (default: `'EE'` for Explicit Euler)
- `num_nodes`: Number of collocation nodes (required)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.explicit import explicit

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'QE': 'EE'
}

description = {
    'sweeper_class': explicit,
    'sweeper_params': sweeper_params,
}
```

**Key Attributes**:
- `self.QE`: The strictly lower triangular explicit integration matrix

---

### 3. imex_1st_order

**File**: `pySDC/implementations/sweeper_classes/imex_1st_order.py`

**Description**: First-order IMEX (Implicit-Explicit) sweeper that splits the right-hand side into implicit and explicit parts.

**When to Use**:
- Problems with additive splitting: `du/dt = f_impl(u) + f_expl(u)`
- Stiff and non-stiff components (treat stiff implicitly, non-stiff explicitly)
- Reaction-diffusion equations, advection-diffusion problems

**Key Parameters**:
- `QI`: Type of implicit integration matrix (default: `'IE'`)
- `QE`: Type of explicit integration matrix (default: `'EE'`)
- `num_nodes`: Number of collocation nodes (required)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'QI': 'LU',
    'QE': 'EE'
}

description = {
    'sweeper_class': imex_1st_order,
    'sweeper_params': sweeper_params,
}
```

**Requirements**: Your problem class must return `dtype_f` objects with `.impl` and `.expl` attributes.

**Key Attributes**:
- `self.QI`: Implicit integration matrix
- `self.QE`: Explicit integration matrix

---

### 4. imex_1st_order_mass

**File**: `pySDC/implementations/sweeper_classes/imex_1st_order_mass.py`

**Description**: IMEX sweeper with support for mass/weighting matrices, extending `imex_1st_order`.

**When to Use**:
- Problems with mass matrices: `M * du/dt = f(u)`
- Finite element formulations with mass matrices
- Weighted time derivatives

**Key Parameters**: Same as `imex_1st_order`

**Special Requirements**:
- Only works on the finest level
- Must set `do_coll_update = False` in sweeper parameters
- Problem must implement mass matrix operations

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'do_coll_update': False,  # Required!
}

description = {
    'sweeper_class': imex_1st_order_mass,
    'sweeper_params': sweeper_params,
}
```

---

### 5. multi_implicit

**File**: `pySDC/implementations/sweeper_classes/multi_implicit.py`

**Description**: First-order sweeper for problems with two separate components, each with its own implicit integration strategy.

**When to Use**:
- Two-component systems with different physics (fast/slow, particle/field)
- Multi-physics coupling requiring different integration strategies
- Problems where components need different implicit matrices

**Key Parameters**:
- `Q1`: Integration matrix for first component (default: `'IE'`)
- `Q2`: Integration matrix for second component (default: `'IE'`)
- `num_nodes`: Number of collocation nodes (required)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'Q1': 'LU',  # For first component
    'Q2': 'MIN'  # For second component
}

description = {
    'sweeper_class': multi_implicit,
    'sweeper_params': sweeper_params,
}
```

**Requirements**: Problem must return `dtype_f` with `.comp1` and `.comp2` attributes.

---

### 6. verlet

**File**: `pySDC/implementations/sweeper_classes/verlet.py`

**Description**: Velocity-Verlet integrator for second-order problems (position/velocity formulation).

**When to Use**:
- Particle dynamics and N-body simulations
- Hamiltonian systems requiring symplectic integration
- Second-order ODEs: `d²u/dt² = f(u, v, t)` where `v = du/dt`

**Key Parameters**:
- `QI`: Implicit integration matrix (default: `'IE'`)
- `QE`: Explicit integration matrix (default: `'EE'`)
- `num_nodes`: Number of collocation nodes (required)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.verlet import verlet

sweeper_params = {
    'quad_type': 'LOBATTO',
    'num_nodes': 3,
}

description = {
    'sweeper_class': verlet,
    'sweeper_params': sweeper_params,
}
```

**Requirements**: Problem must provide position and velocity components in `dtype_u`.

**Key Attributes**:
- `self.QI`, `self.QE`: Base integration matrices
- `self.QQ`, `self.Qx`, `self.QT`: Derived matrices for position/velocity updates

---

### 7. boris_2nd_order

**File**: `pySDC/implementations/sweeper_classes/boris_2nd_order.py`

**Description**: Second-order Boris integrator for charged particles in electromagnetic fields.

**When to Use**:
- Charged particle dynamics in magnetic/electric fields
- Plasma physics simulations
- When Lorentz force requires special treatment for stability

**Key Parameters**:
- `QI`: Implicit integration matrix (default: `'IE'`)
- `QE`: Explicit integration matrix (default: `'EE'`)
- `num_nodes`: Number of collocation nodes (required)

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

sweeper_params = {
    'quad_type': 'LOBATTO',
    'num_nodes': 3,
}

description = {
    'sweeper_class': boris_2nd_order,
    'sweeper_params': sweeper_params,
}
```

**Requirements**: Problem must implement `boris_solver()` method for velocity rotation.

**Key Attributes**:
- `self.S`, `self.ST`, `self.SQ`, `self.Sx`: Node-to-node matrices for position updates

---

### 8. RungeKutta

**File**: `pySDC/implementations/sweeper_classes/Runge_Kutta.py`

**Description**: Family of Runge-Kutta methods adapted to the sweeper interface. Note: These are **direct solvers** (single iteration), not iterative like standard SDC.

**When to Use**:
- When you want a classical RK method within the pySDC framework
- Benchmarking against standard methods
- Embedded methods for error estimation

**Available RK Schemes**:
- **Explicit**: `ForwardEuler`, `ExplicitMidpointMethod`, `RK4`, `Heun_Euler`, `Cash_Karp`, `DOPRI5`, `DOPRI853`, `Bogacki_Shampine`
- **Implicit**: `BackwardEuler`, `CrankNicolson`, `ImplicitMidpointMethod`, `Pareschi_Russo`
- **DIRK**: `DIRK43`, `ESDIRK43`, `ESDIRK53`
- **IMEX**: `ARK548L2SA`, `ARK548L2SAERK`, `ARK548L2SAESDIRK`, `ARK54`, `IMEXEXP3`, `EDIRK4`, `EDIRK5`

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4

sweeper_params = {}

description = {
    'sweeper_class': RK4,
    'sweeper_params': sweeper_params,
}
```

**Important Notes**:
- RK methods are **one-pass** solvers - SDC iteration won't improve them
- Set `maxiter = 1` in level parameters when using RK sweepers
- Some methods are embedded (provide error estimates)

---

### 9. RungeKuttaNystrom

**File**: `pySDC/implementations/sweeper_classes/Runge_Kutta_Nystrom.py`

**Description**: Runge-Kutta-Nyström methods for second-order ODEs in position/velocity form, avoiding reduction to first-order systems.

**When to Use**:
- Second-order systems where maintaining structure is important
- Orbital mechanics, molecular dynamics
- When you want RK-type methods for second-order problems

**Available Schemes**:
- `Velocity_Verlet`: Symplectic velocity-Verlet scheme
- `RKN`: Generic RKN method

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.Runge_Kutta_Nystrom import Velocity_Verlet

sweeper_params = {}

description = {
    'sweeper_class': Velocity_Verlet,
    'sweeper_params': sweeper_params,
}
```

**Key Attributes**:
- `self.Qx`: Position integration matrix from Butcher tableau

---

### 10. MultiStep

**File**: `pySDC/implementations/sweeper_classes/Multistep.py`

**Description**: Framework for multistep methods (Adams-Bashforth, Adams-Moulton, BDF).

**When to Use**:
- When you need variable-order methods
- Smooth problems where multistep methods excel
- As an alternative to Runge-Kutta for efficiency

**Available Schemes**:
- **Adams-Bashforth**: Explicit multistep (1-5 steps)
- **Adams-Moulton**: Implicit multistep
- **BDF**: Backward differentiation formulas

**Usage Example**:
```python
from pySDC.implementations.sweeper_classes.Multistep import AdamsBashforthExplicit1Step

sweeper_params = {
    'num_nodes': 1,  # Multistep methods typically use single node
}

description = {
    'sweeper_class': AdamsBashforthExplicit1Step,
    'sweeper_params': sweeper_params,
}
```

**Key Attributes**:
- `self.alpha`: Coefficients for solution history
- `self.beta`: Coefficients for RHS history

---

## MPI-Parallelized Sweepers

Some sweepers have MPI-parallelized variants for distributed memory parallelism:

### generic_implicit_MPI

**File**: `pySDC/implementations/sweeper_classes/generic_implicit_MPI.py`

Parallelized version of `generic_implicit` using MPI for node-level parallelism.

**Classes**: `SweeperMPI` (base), `generic_implicit_MPI`

**When to Use**: Same as `generic_implicit`, but when you want to parallelize sweeps across nodes using MPI.

### imex_1st_order_MPI

**File**: `pySDC/implementations/sweeper_classes/imex_1st_order_MPI.py`

MPI-parallelized IMEX sweeper.

**When to Use**: Same as `imex_1st_order`, but with MPI parallelization.

---

## Choosing the Right Sweeper

### Decision Tree

1. **What type of problem do you have?**
   - **Standard ODE**: → `generic_implicit` or `explicit`
   - **Stiff + Non-stiff split**: → `imex_1st_order`
   - **Second-order (position/velocity)**: → `verlet` or `boris_2nd_order` (if EM fields)
   - **Two separate components**: → `multi_implicit`
   - **Mass matrix**: → `imex_1st_order_mass`

2. **Do you need a specific method rather than SDC?**
   - **Classical Runge-Kutta**: → `RungeKutta` variants
   - **Multistep methods**: → `MultiStep` variants

3. **Do you need parallelism?**
   - **MPI parallelization**: → Use `*_MPI` variants

### Performance Considerations

- **Explicit sweepers** are faster per iteration but may need smaller time steps
- **Implicit sweepers** allow larger time steps but require solving nonlinear systems
- **IMEX sweepers** balance both by treating only stiff terms implicitly
- **RK methods** don't benefit from SDC iteration - set `maxiter=1`
- **MPI sweepers** add communication overhead but enable node-level parallelism

---

## Common Parameters

All sweepers support these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_nodes` | int | **Required** | Number of collocation nodes |
| `quad_type` | str | `'RADAU-RIGHT'` | Quadrature type for collocation |
| `node_type` | str | Based on `quad_type` | Distribution of collocation nodes |
| `do_coll_update` | bool | `False` | Whether to perform collocation update |
| `initial_guess` | str | `'spread'` | Initial guess strategy (`'spread'`, `'zero'`, `'random'`) |
| `skip_residual_computation` | tuple | `()` | Skip residual computation at specific iterations (performance) |

### Common Quadrature Types

- `'RADAU-RIGHT'`: Radau IIA (includes right endpoint)
- `'RADAU-LEFT'`: Radau IA (includes left endpoint)
- `'LOBATTO'`: Lobatto (includes both endpoints)
- `'GAUSS'`: Gauss-Legendre (no endpoints)

---

## Implementing Custom Sweepers

To implement your own sweeper:

1. **Inherit from `Sweeper`** (or existing sweeper)
2. **Implement required methods**:
   - `integrate()`: Compute RHS integral
   - `update_nodes()`: Update node values (one sweep)
   - Optionally: `compute_residual()`, `compute_end_point()`

3. **Define integration matrices** in `__init__`:
   ```python
   self.QI = self.get_Qdelta_implicit(qd_type='IE')
   self.QE = self.get_Qdelta_explicit(qd_type='EE')
   ```

4. **Follow naming conventions** (see [naming_conventions.md](03_naming_conventions.md))

**Example skeleton**:
```python
from pySDC.core.sweeper import Sweeper

class my_custom_sweeper(Sweeper):
    """
    Custom sweeper for my specific problem type
    """
    
    def __init__(self, params, level):
        # Set defaults for custom parameters
        if 'my_param' not in params:
            params['my_param'] = 'default_value'
        
        super().__init__(params, level)
        
        # Initialize integration matrices
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)
    
    def integrate(self):
        """Compute integral of RHS"""
        L = self.level
        P = L.prob
        
        me = []
        for m in range(1, self.coll.num_nodes + 1):
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]
        
        return me
    
    def update_nodes(self):
        """Update values at collocation nodes"""
        # Your implementation here
        pass
```

---

## Advanced Topics

### QDelta Matrix Types

The `qmat` library provides many QDelta matrix types. Common ones:

- **IE (Implicit Euler)**: Simple, robust, first-order
- **LU**: LU decomposition-based
- **MIN**: Minimal correction
- **MIN3**: Minimal correction variant  
- **TRAP**: Trapezoidal rule
- **BEPAR**: Backward Euler parallelizable
- **PIC**: Picard iteration
- **VDHS**: Van der Houwen-Sommeijer

See [qmat documentation](https://qmat.readthedocs.io) for complete list.

### Initial Guess Strategies

The `initial_guess` parameter controls how node values are initialized:

- `'spread'`: Spread initial value to all nodes (most common)
- `'zero'`: Initialize with zeros
- `'random'`: Random initialization (useful for testing)
- `'copy'`: Copy from previous iteration

### Residual Computation

To skip residual computation for performance:
```python
sweeper_params = {
    'skip_residual_computation': (1, 2, 3),  # Skip at iterations 1, 2, 3
}
```

This can speed up iteration but prevents residual-based monitoring.

---

## References

- **pySDC Core**: `pySDC/core/sweeper.py` - Base sweeper class
- **qmat Library**: [https://qmat.readthedocs.io](https://qmat.readthedocs.io) - QDelta matrix generation
- **Tutorial**: See `pySDC/tutorial/` for practical examples
- **Academic Paper**: Ruprecht & Speck, "Spectral deferred corrections with fast-wave slow-wave splitting", ACM TOMS 2019

---

## Examples from Tutorials

### Example 1: Basic Implicit Sweeper

From tutorial step 1:
```python
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'QI': 'IE'
}

level_params = {
    'dt': 0.1,
}

description = {
    'sweeper_class': generic_implicit,
    'sweeper_params': sweeper_params,
    'level_params': level_params,
}
```

### Example 2: IMEX for Advection-Diffusion

```python
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

sweeper_params = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 5,
    'QI': 'LU',  # For diffusion (stiff, implicit)
    'QE': 'EE',  # For advection (non-stiff, explicit)
}

description = {
    'sweeper_class': imex_1st_order,
    'sweeper_params': sweeper_params,
}
```

### Example 3: Using RK4 for Comparison

```python
from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4

level_params = {
    'dt': 0.1,
    'maxiter': 1,  # Important: RK doesn't benefit from iteration!
}

description = {
    'sweeper_class': RK4,
    'sweeper_params': {},
    'level_params': level_params,
}
```

---

## Troubleshooting

### Common Issues

1. **"need num_nodes to instantiate step"**
   - Solution: Always provide `num_nodes` in `sweeper_params`

2. **"Problem dtype_f has no attribute 'impl'"**
   - Solution: Your problem must return proper `dtype_f` for IMEX sweepers

3. **"Mass matrix sweeper only works on finest level"**
   - Solution: Use `imex_1st_order_mass` only on finest level, set `do_coll_update=False`

4. **RK methods not converging with multiple iterations**
   - Solution: RK methods are one-pass; set `maxiter=1` in level params

5. **Poor performance with wrong sweeper choice**
   - Solution: Match sweeper to problem stiffness (IMEX for mixed, explicit for non-stiff)

---

*Last updated: February 2026*
