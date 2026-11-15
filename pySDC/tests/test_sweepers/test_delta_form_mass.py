"""
The delta-form sweepers in mass-matrix form must reproduce the mass-matrix sweepers they are the counterparts of.
"""

import pytest

SWEEPER_PARAMS = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}


def mass_heat_class(with_delta_solve):
    """
    A small problem in mass form, M u' = A u with a non-diagonal M, so that a sweeper that drops the mass matrix
    somewhere gets a different answer. With `with_delta_solve`, it also solves the correction equation itself.
    """
    import numpy as np
    from pySDC.core.problem import Problem
    from pySDC.implementations.datatype_classes.mesh import mesh

    class MassHeat(Problem):
        dtype_u = mesh
        dtype_f = mesh

        def __init__(self, nvars=15):
            super().__init__(init=(nvars, None, np.dtype('float64')))
            self._makeAttributeAndRegister('nvars', localVars=locals(), readOnly=True)
            h = 1 / (nvars + 1)
            ones = np.ones(nvars)
            # P1 mass and stiffness matrices with zero Dirichlet values
            self.M = h / 6 * (4 * np.diag(ones) + np.diag(ones[1:], 1) + np.diag(ones[1:], -1))
            self.A = -(2 * np.diag(ones) - np.diag(ones[1:], 1) - np.diag(ones[1:], -1)) / h
            self.x = np.linspace(h, 1 - h, nvars)

        def _wrap(self, values):
            me = self.dtype_u(self.init)
            me[:] = values
            return me

        def eval_f(self, u, t):
            return self._wrap(self.A @ u)

        def apply_mass_matrix(self, u):
            return self._wrap(self.M @ u)

        def solve_system(self, rhs, factor, u0, t):
            return self._wrap(np.linalg.solve(self.M - factor * self.A, rhs))

        def u_exact(self, t):
            return self._wrap(np.sin(np.pi * self.x))

    if not with_delta_solve:
        return MassHeat

    class MassHeatDelta(MassHeat):
        def solve_system_delta(self, rhs, alpha, u_old, f_old, t):
            self.delta_solves = getattr(self, 'delta_solves', 0) + 1
            return self._wrap(np.linalg.solve(self.M - alpha * self.A, rhs))

    return MassHeatDelta


def run(
    problem_class, problem_params, sweeper_class, dt, nsteps, maxiter, num_procs=1, sweeper_params=(), **description
):
    """Run with a fixed number of iterations and return the controller and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': {**SWEEPER_PARAMS, **dict(sweeper_params)},
        'level_params': {'dt': dt, 'restol': -1.0},
        'step_params': {'maxiter': maxiter},
        **description,
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=nsteps * dt)
    return controller, uend


@pytest.mark.base
@pytest.mark.parametrize('with_delta_solve', [False, True])
@pytest.mark.parametrize('QI', ['LU', 'MIN-SR-S', 'PIC'])
@pytest.mark.parametrize('maxiter', [1, 4])
def test_delta_implicit_mass_matches_generic_implicit_mass(with_delta_solve, QI, maxiter):
    """Both correction paths, and the zero diagonal of PIC, where the correction needs a mass solve."""
    import numpy as np
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    # Picard is explicit, so it needs a step within the stability limit of the stiff problem
    dt = 1e-4 if QI == 'PIC' else 0.05
    problem_class = mass_heat_class(with_delta_solve)
    _, u_ref = run(problem_class, {}, generic_implicit_mass, dt, 3, maxiter, sweeper_params={'QI': QI})
    controller, u = run(problem_class, {}, delta_implicit_mass, dt, 3, maxiter, sweeper_params={'QI': QI})

    assert np.allclose(u, u_ref, rtol=0, atol=1e-13), abs(u - u_ref)
    assert with_delta_solve == hasattr(controller.MS[0].levels[0].prob, 'delta_solves')


@pytest.mark.base
def test_linear_implicit_is_refused():
    from pySDC.core.errors import ParameterError
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit_mass

    with pytest.raises(ParameterError, match='linear_implicit'):
        run(mass_heat_class(False), {}, delta_implicit_mass, 0.05, 1, 1, sweeper_params={'linear_implicit': True})


@pytest.mark.firedrake
@pytest.mark.parametrize('imex', [False, True])
@pytest.mark.parametrize('QI', ['LU', 'PIC'])
def test_delta_mass_sweepers_on_forms(imex, QI):
    """
    The nonlinear manufactured problem with time-dependent Dirichlet values, fully implicit and IMEX: the delta form
    has to impose the boundary data on the new value, not on the correction.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake, FormFiredrakeIMEX
    from pySDC.implementations.sweeper_classes.delta_form import delta_imex_1st_order_mass, delta_implicit_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.tests.test_problems.test_form_firedrake import nonlinear_forms

    problem_class = FormFiredrakeIMEX if imex else FormFiredrake
    sweepers = (
        (imex_1st_order_mass, delta_imex_1st_order_mass) if imex else (generic_implicit_mass, delta_implicit_mass)
    )

    ends = []
    for sweeper_class in sweepers:
        V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
        forms = nonlinear_forms(V, 0.1, imex, smooth=False)
        # Picard is explicit, so it needs a step within the stability limit of the diffusion
        dt = 1e-3 if QI == 'PIC' else 0.1
        controller, uend = run(problem_class, forms, sweeper_class, dt, 3, 4, sweeper_params={'QI': QI})
        ends.append(uend)

    diff = np.abs(ends[0].dat.data_ro - ends[1].dat.data_ro).max()
    assert diff < 1e-12, f'{sweepers[1].__name__} differs from {sweepers[0].__name__} by {diff:.2e}'
    # and the iteration did something
    P = controller.MS[0].levels[0].prob
    assert abs(ends[1] - P.u_exact(3 * dt)) < 1e-3


@pytest.mark.firedrake
@pytest.mark.parametrize('num_procs', [1, 4])
def test_delta_mass_sweeper_in_a_hierarchy(num_procs):
    """MLSDC and PFASST with the mass transfer: the coarse levels carry M u0 and the delta form has to honour that."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit_mass
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy
    from pySDC.tests.test_problems.test_form_firedrake import nonlinear_forms

    ends = []
    for sweeper_class in (generic_implicit_mass, delta_implicit_mass):
        hierarchy = fd.MeshHierarchy(fd.UnitIntervalMesh(4), 2)
        per_level = [nonlinear_forms(fd.FunctionSpace(mesh, 'CG', 2), 0.1, False, smooth=False) for mesh in hierarchy]
        problem_params = {key: [forms[key] for forms in per_level[::-1]] for key in per_level[0]}
        _, uend = run(
            FormFiredrake,
            problem_params,
            sweeper_class,
            0.125,
            4,
            5,
            num_procs=num_procs,
            base_transfer_class=base_transfer_mass,
            space_transfer_class=MeshToMeshFiredrakeHierarchy,
        )
        ends.append(uend)

    diff = np.abs(ends[0].dat.data_ro - ends[1].dat.data_ro).max()
    assert diff < 1e-12, f'the delta form differs from the mass sweeper by {diff:.2e}'
