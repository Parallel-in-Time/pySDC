import pytest


def get_gusto_stepper(eqns, method, spatial_methods, dirname='./tmp'):
    from gusto import IO, OutputParameters, PrescribedTransport
    import sys

    if '--running-tests' not in sys.argv:
        sys.argv.append('--running-tests')

    output = OutputParameters(dirname=dirname, dumpfreq=15)
    io = IO(method.domain, output)
    return PrescribedTransport(eqns, method, io, False, transport_method=spatial_methods)


def tracer_setup(tmpdir='./tmp', degree=1, small_dt=False, comm=None):
    from firedrake import (
        IcosahedralSphereMesh,
        PeriodicIntervalMesh,
        ExtrudedMesh,
        SpatialCoordinate,
        as_vector,
        sqrt,
        exp,
        pi,
        COMM_WORLD,
    )
    from gusto import OutputParameters, Domain, IO
    from gusto.core.logging import logger, INFO
    from collections import namedtuple

    logger.setLevel(INFO)

    opts = ('domain', 'tmax', 'io', 'f_init', 'f_end', 'degree', 'uexpr', 'umax', 'radius', 'tol')
    TracerSetup = namedtuple('TracerSetup', opts)
    TracerSetup.__new__.__defaults__ = (None,) * len(opts)

    radius = 1
    comm = COMM_WORLD if comm is None else comm
    mesh = IcosahedralSphereMesh(radius=radius, refinement_level=3, degree=1, comm=comm)
    x = SpatialCoordinate(mesh)

    # Parameters chosen so that dt != 1
    # Gaussian is translated from (lon=pi/2, lat=0) to (lon=0, lat=0)
    # to demonstrate that transport is working correctly
    if small_dt:
        dt = pi / 3.0 * 0.005
    else:
        dt = pi / 3.0 * 0.02

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    domain = Domain(mesh, dt, family="BDM", degree=degree)
    io = IO(domain, output)

    umax = 1.0
    uexpr = as_vector([-umax * x[1] / radius, umax * x[0] / radius, 0.0])

    tmax = pi / 2
    f_init = exp(-x[2] ** 2 - x[0] ** 2)
    f_end = exp(-x[2] ** 2 - x[1] ** 2)

    tol = 0.05

    return TracerSetup(domain, tmax, io, f_init, f_end, degree, uexpr, umax, radius, tol)


@pytest.fixture
def setup():
    return tracer_setup()


def get_gusto_advection_setup(use_transport_scheme, imex, setup):
    from gusto import ContinuityEquation, AdvectionEquation, split_continuity_form, DGUpwind
    from gusto.core.labels import time_derivative, transport, implicit, explicit

    domain = setup.domain
    V = domain.spaces("DG")

    eqn = ContinuityEquation(domain, V, "f")
    eqn = split_continuity_form(eqn)

    transport_methods = [DGUpwind(eqn, 'f')]
    spatial_methods = None

    if use_transport_scheme:
        spatial_methods = transport_methods

    if imex:
        eqn.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
        eqn.label_terms(lambda t: t.has_label(transport), explicit)
    else:
        eqn.label_terms(lambda t: not t.has_label(time_derivative), implicit)

    return eqn, domain, spatial_methods, setup


def get_initial_conditions(timestepper, setup):
    timestepper.fields("f").interpolate(setup.f_init)
    timestepper.fields("u").project(setup.uexpr)
    return timestepper


@pytest.mark.firedrake
def test_generic_gusto_problem(setup):
    from pySDC.implementations.problem_classes.GenericGusto import GenericGusto
    from firedrake import norm, Constant
    import numpy as np
    from gusto import ThetaMethod

    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(False, False, setup)

    dt = 1e-1
    domain.dt = Constant(dt)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Prepare different methods
    # ------------------------------------------------------------------------ #

    problem = GenericGusto(eqns, solver_parameters=solver_parameters)
    stepper_backward = get_gusto_stepper(
        eqns, ThetaMethod(domain, theta=1.0, solver_parameters=solver_parameters), spatial_methods
    )
    stepper_forward = get_gusto_stepper(
        eqns, ThetaMethod(domain, theta=0.0, solver_parameters=solver_parameters), spatial_methods
    )

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_backward, stepper_forward]:
        get_initial_conditions(stepper, setup)

    u_start = problem.u_init
    u_start.assign(stepper_backward.fields('f'))

    un = problem.solve_system(u_start, dt, u_start)
    fn = problem.eval_f(un)

    u02 = un - dt * fn

    error = abs(u_start - u02) / abs(u_start)

    assert error < np.finfo(float).eps * 1e2

    # test forward Euler step
    stepper_forward.run(t=0, tmax=dt)
    un_ref = problem.dtype_u(problem.init)
    un_ref.assign(stepper_forward.fields('f'))
    un_forward = u_start + dt * problem.eval_f(u_start)
    error = abs(un_forward - un_ref) / abs(un_ref)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'Forward Euler does not match reference implementation! Got relative difference of {error}'

    # test backward Euler step
    stepper_backward.run(t=0, tmax=dt)
    un_ref = problem.dtype_u(problem.init)
    un_ref.assign(stepper_backward.fields('f'))
    error = abs(un - un_ref) / abs(un_ref)

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'Backward Euler does not match reference implementation! Got relative difference of {error}'


class Method(object):
    imex = False

    @staticmethod
    def get_pySDC_method():
        raise NotImplementedError

    @staticmethod
    def get_Gusto_method():
        raise NotImplementedError


class RK4(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import RK4

        return RK4

    @staticmethod
    def get_Gusto_method():
        from gusto import RK4

        return RK4


class ImplicitMidpoint(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import ImplicitMidpointMethod

        return ImplicitMidpointMethod

    @staticmethod
    def get_Gusto_method():
        from gusto import ImplicitMidpoint

        return ImplicitMidpoint


class BackwardEuler(Method):

    @staticmethod
    def get_pySDC_method():
        from pySDC.implementations.sweeper_classes.Runge_Kutta import BackwardEuler

        return BackwardEuler

    @staticmethod
    def get_Gusto_method():
        from gusto import BackwardEuler

        return BackwardEuler


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
@pytest.mark.parametrize('method', [RK4, ImplicitMidpoint, BackwardEuler])
def test_pySDC_integrator_RK(use_transport_scheme, method, setup):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from firedrake import norm
    import numpy as np

    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(use_transport_scheme, method.imex, setup)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #

    level_params = dict()
    level_params['restol'] = -1

    step_params = dict()
    step_params['maxiter'] = 1

    sweeper_params = dict()

    problem_params = dict()

    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = method.get_pySDC_method()
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_gusto = get_gusto_stepper(
        eqns, method.get_Gusto_method()(domain, solver_parameters=solver_parameters), spatial_methods
    )
    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(
            description,
            controller_params,
            domain,
            solver_parameters=solver_parameters,
            imex=method.imex,
        ),
        spatial_methods,
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_gusto, stepper_pySDC]:
        get_initial_conditions(stepper, setup)

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    def run(stepper, n_steps):
        stepper.run(t=0, tmax=n_steps * float(domain.dt))

    for stepper in [stepper_gusto, stepper_pySDC]:
        run(stepper, 5)

    error = norm(stepper_gusto.fields('f') - stepper_pySDC.fields('f')) / norm(stepper_gusto.fields('f'))
    print(error)

    assert (
        error < solver_parameters['snes_rtol'] * 1e3
    ), f'pySDC and Gusto differ in method {method}! Got relative difference of {error}'


@pytest.mark.firedrake
@pytest.mark.parametrize('use_transport_scheme', [True, False])
@pytest.mark.parametrize('imex', [True, False])
def test_pySDC_integrator(use_transport_scheme, imex, setup):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from gusto import BackwardEuler, SDC
    from firedrake import norm
    import numpy as np

    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(use_transport_scheme, imex, setup)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #
    if imex:
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_cls
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_cls

    level_params = dict()
    level_params['restol'] = -1

    step_params = dict()
    step_params['maxiter'] = 3

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    problem_params = dict()

    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_cls
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup SDC in gusto
    # ------------------------------------------------------------------------ #

    SDC_params = {
        'base_scheme': BackwardEuler(domain, solver_parameters=solver_parameters),
        'M': sweeper_params['num_nodes'],
        'maxk': step_params['maxiter'],
        'quad_type': sweeper_params['quad_type'],
        'node_type': sweeper_params['node_type'],
        'qdelta_imp': sweeper_params['QI'],
        'qdelta_exp': sweeper_params['QE'],
        'formulation': 'Z2N',
        'initial_guess': 'copy',
        'nonlinear_solver_parameters': solver_parameters,
        'linear_solver_parameters': solver_parameters,
        'final_update': False,
    }

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_gusto = get_gusto_stepper(eqns, SDC(**SDC_params, domain=domain), spatial_methods)

    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(
            description,
            controller_params,
            domain,
            solver_parameters=solver_parameters,
            imex=imex,
        ),
        spatial_methods,
    )

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_gusto, stepper_pySDC]:
        get_initial_conditions(stepper, setup)

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    def run(stepper, n_steps):
        stepper.run(t=0, tmax=n_steps * float(domain.dt))

    for stepper in [stepper_gusto, stepper_pySDC]:
        run(stepper, 5)

    error = norm(stepper_gusto.fields('f') - stepper_pySDC.fields('f')) / norm(stepper_gusto.fields('f'))
    print(error)

    assert (
        error < solver_parameters['snes_rtol'] * 1e3
    ), f'pySDC and Gusto differ in SDC! Got relative difference of {error}'


@pytest.mark.firedrake
@pytest.mark.parametrize('dt_initial', [1e-5, 1e-1])
def test_pySDC_integrator_with_adaptivity(dt_initial, setup):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.convergence_controller_classes.spread_step_sizes import SpreadStepSizesBlockwiseNonMPI
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeRounding
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from pySDC.helpers.stats_helper import get_sorted
    from gusto import BackwardEuler, SDC
    from firedrake import norm, Constant
    import numpy as np

    use_transport_scheme = True
    imex = True

    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(use_transport_scheme, imex, setup)
    domain.dt = Constant(dt_initial)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #
    if imex:
        from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_cls
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_cls

    level_params = dict()
    level_params['restol'] = -1

    step_params = dict()
    step_params['maxiter'] = 3

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    problem_params = dict()

    convergence_controllers = {}
    convergence_controllers[Adaptivity] = {'e_tol': 1e-5, 'rel_error': True}
    convergence_controllers[SpreadStepSizesBlockwiseNonMPI] = {'overwrite_to_reach_Tend': False}
    convergence_controllers[StepSizeRounding] = {}

    controller_params = dict()
    controller_params['logger_level'] = 15
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_cls
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # ------------------------------------------------------------------------ #
    # Setup SDC in gusto
    # ------------------------------------------------------------------------ #

    SDC_params = {
        'base_scheme': BackwardEuler(domain, solver_parameters=solver_parameters),
        'M': sweeper_params['num_nodes'],
        'maxk': step_params['maxiter'],
        'quad_type': sweeper_params['quad_type'],
        'node_type': sweeper_params['node_type'],
        'qdelta_imp': sweeper_params['QI'],
        'qdelta_exp': sweeper_params['QE'],
        'formulation': 'Z2N',
        'initial_guess': 'copy',
        'nonlinear_solver_parameters': solver_parameters,
        'linear_solver_parameters': solver_parameters,
        'final_update': False,
    }

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(
            description,
            controller_params,
            domain,
            solver_parameters=solver_parameters,
            imex=imex,
        ),
        spatial_methods,
    )

    stepper_gusto = get_gusto_stepper(eqns, SDC(**SDC_params, domain=domain), spatial_methods)

    stepper_pySDC.scheme.timestepper = stepper_pySDC

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_gusto, stepper_pySDC]:
        get_initial_conditions(stepper, setup)

    # ------------------------------------------------------------------------ #
    # Run tests
    # ------------------------------------------------------------------------ #

    # run with pySDC first
    get_initial_conditions(stepper_pySDC, setup)
    stepper_pySDC.run(t=0, tmax=0.1)

    # retrieve step sizes
    stats = stepper_pySDC.scheme.stats
    dts_pySDC = get_sorted(stats, type='dt', recomputed=False)

    assert len(dts_pySDC) > 0, 'No step sizes were recorded in adaptivity test!'

    # run with Gusto using same step sizes
    get_initial_conditions(stepper_gusto, setup)
    old_dt = float(stepper_gusto.dt)

    for _dt in dts_pySDC:
        # update step size
        stepper_gusto.dt = Constant(_dt[1])

        stepper_gusto.scheme.Q *= _dt[1] / old_dt
        stepper_gusto.scheme.Qdelta_imp *= _dt[1] / old_dt
        stepper_gusto.scheme.Qdelta_exp *= _dt[1] / old_dt
        stepper_gusto.scheme.nodes *= _dt[1] / old_dt

        old_dt = _dt[1] * 1.0

        # run
        stepper_gusto.run(t=_dt[0], tmax=_dt[0] + _dt[1])

        # clear solver cache with old step size
        del stepper_gusto.scheme.solvers

    assert np.isclose(float(stepper_pySDC.t), float(stepper_gusto.t))

    print(dts_pySDC)

    error = norm(stepper_gusto.fields('f') - stepper_pySDC.fields('f')) / norm(stepper_gusto.fields('f'))
    print(error, norm(stepper_gusto.fields('f')))

    assert (
        error < np.finfo(float).eps * 1e2
    ), f'SDC does not match reference implementation with adaptive step size selection! Got relative difference of {error}'


@pytest.mark.firedrake
@pytest.mark.parametrize('n_steps', [1, 2, 4])
@pytest.mark.parametrize('useMPIController', [True, False])
def test_pySDC_integrator_MSSDC(n_steps, useMPIController, setup, submit=True, n_tasks=4):
    if submit and useMPIController:
        import os
        import subprocess

        assert n_steps <= n_tasks

        my_env = os.environ.copy()
        my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
        cwd = '.'
        cmd = f'mpiexec -np {n_tasks} python {__file__} --test=MSSDC --n_steps={n_steps}'.split()

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        for line in p.stdout:
            print(line)
        for line in p.stderr:
            print(line)
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
            p.returncode,
            n_steps,
        )
        return None

    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_cls
    from firedrake import norm, Constant, COMM_WORLD
    import numpy as np

    MSSDC_args = {}
    dirname = './tmp'
    if useMPIController:
        from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator

        controller_communicator = FiredrakeEnsembleCommunicator(COMM_WORLD, COMM_WORLD.size // n_steps)
        assert controller_communicator.size == n_steps
        MSSDC_args = {'useMPIController': True, 'controller_communicator': controller_communicator}
        dirname = f'./tmp_{controller_communicator.rank}'
        setup = tracer_setup(tmpdir=dirname, comm=controller_communicator.space_comm)
    else:
        MSSDC_args = {'useMPIController': False, 'n_steps': n_steps}

    method = BackwardEuler
    use_transport_scheme = True
    eqns, domain, spatial_methods, setup = get_gusto_advection_setup(use_transport_scheme, method.imex, setup)

    solver_parameters = {
        'snes_type': 'newtonls',
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_rtol': 1e-12,
        'snes_rtol': 1e-12,
        'ksp_atol': 1e-30,
        'snes_atol': 1e-30,
        'ksp_divtol': 1e30,
        'snes_divtol': 1e30,
        'snes_max_it': 99,
    }

    # ------------------------------------------------------------------------ #
    # Setup pySDC
    # ------------------------------------------------------------------------ #

    level_params = dict()
    level_params['restol'] = -1

    step_params = dict()
    step_params['maxiter'] = 1

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['num_nodes'] = 1
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['initial_guess'] = 'copy'

    problem_params = dict()

    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_cls
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # ------------------------------------------------------------------------ #
    # Setup time steppers
    # ------------------------------------------------------------------------ #

    tmax = float(domain.dt) * 2 * n_steps

    stepper_gusto = get_gusto_stepper(
        eqns,
        method.get_Gusto_method()(domain, solver_parameters=solver_parameters),
        spatial_methods,
        dirname=dirname,
    )

    domain.dt = Constant(domain.dt) * n_steps

    stepper_pySDC = get_gusto_stepper(
        eqns,
        pySDC_integrator(
            description,
            controller_params,
            domain,
            solver_parameters=solver_parameters,
            imex=method.imex,
            **MSSDC_args,
        ),
        spatial_methods,
        dirname=dirname,
    )

    # ------------------------------------------------------------------------ #
    # Get Initial conditions and run
    # ------------------------------------------------------------------------ #

    for stepper in [stepper_gusto, stepper_pySDC]:
        get_initial_conditions(stepper, setup)
        stepper.run(t=0, tmax=tmax)

    # ------------------------------------------------------------------------ #
    # Check results
    # ------------------------------------------------------------------------ #

    assert stepper_gusto.t == stepper_pySDC.t

    error = norm(stepper_gusto.fields('f') - stepper_pySDC.fields('f')) / norm(stepper_gusto.fields('f'))
    print(error)

    assert (
        error < solver_parameters['snes_rtol'] * 1e3
    ), f'pySDC and Gusto differ in method {method}! Got relative difference of {error}'


if __name__ == '__main__':
    from mpi4py import MPI
    from argparse import ArgumentParser

    if MPI.COMM_WORLD.size == 1:
        setup = tracer_setup()
    else:
        setup = None

    parser = ArgumentParser()
    parser.add_argument(
        '--test',
        help="which kind of test you want to run",
        type=str,
        default=None,
    )
    parser.add_argument(
        '--n_steps',
        help="number of steps",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if args.test == 'MSSDC':
        test_pySDC_integrator_MSSDC(n_steps=args.n_steps, useMPIController=True, setup=setup, submit=False)
    else:
        # test_generic_gusto_problem(setup)
        # test_pySDC_integrator_RK(False, RK4, setup)
        # test_pySDC_integrator(False, False, setup)
        # test_pySDC_integrator_with_adaptivity(1e-3, setup)
        test_pySDC_integrator_MSSDC(4, True, setup)
