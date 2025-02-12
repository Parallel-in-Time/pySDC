"""
Example for running pySDC together with Gusto. This test runs a shallow water equation and may take a considerable
amount of time. After you have run it, move on to step F_2, which includes a plotting script.

This is Test Case 5 (flow over a mountain) of Williamson et al, 1992:
``A standard test set for numerical approximations to the shallow water
equations in spherical geometry'', JCP.

This script is adapted from the Gusto example: https://github.com/firedrakeproject/gusto/blob/main/examples/shallow_water/williamson_5.py

The pySDC coupling works by setting up pySDC as a time integrator within Gusto.
To this end, you need to construct a pySDC description and controller parameters as usual and pass them when
constructing the pySDC time discretization.

After passing this to a Gusto timestepper, you have two choices:
    - Access the `.scheme.controller` variable of the timestepper, which is the pySDC controller and use pySDC for
      running
    - Use the Gusto timestepper for running
You may wonder why it is necessary to construct a Gusto timestepper if you don't want to use it. The reason is the
setup of spatial methods, such as upwinding. These are passed to the Gusto timestepper and modify the residual of the
equations during its instantiation. Once the residual is modified, we can choose whether to continue in Gusto or pySDC.

This script supports space-time parallelism, as well as running the Gusto SDC implementation or the pySDC-Gusto coupling.
Please run with `--help` to learn how to configure this script.
"""

import firedrake as fd
from pySDC.helpers.pySDC_as_gusto_time_discretization import pySDC_integrator
from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator
from gusto import SDC, BackwardEuler
from gusto.core.labels import implicit, time_derivative
from gusto.core.logging import logger, INFO

logger.setLevel(INFO)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from firedrake import SpatialCoordinate, as_vector, pi, sqrt, min_value, Function
from gusto import (
    Domain,
    IO,
    OutputParameters,
    DGUpwind,
    ShallowWaterParameters,
    ShallowWaterEquations,
    Sum,
    lonlatr_from_xyz,
    GeneralIcosahedralSphereMesh,
    ZonalComponent,
    MeridionalComponent,
    RelativeVorticity,
    Timestepper,
)

williamson_5_defaults = {
    'ncells_per_edge': 12,  # number of cells per icosahedron edge
    'dt': 900.0,
    'tmax': 50.0 * 24.0 * 60.0 * 60.0,  # 50 days
    'dumpfreq': 10,  # output every <dumpfreq> steps
    'dirname': 'williamson_5',  # results will go into ./results/<dirname>
    'time_parallelism': False,  # use parallel diagonal SDC or not
    'QI': 'MIN-SR-S',  # implicit preconditioner
    'M': '3',  # number of collocation nodes
    'kmax': '5',  # use fixed number of iteration up to this value
    'use_pySDC': True,  # whether to use pySDC for time integration
    'use_adaptivity': True,  # whether to use adaptive step size selection
    'Nlevels': 1,  # number of levels in SDC
    'logger_level': 15,  # pySDC logger level
}


def williamson_5(
    ncells_per_edge=williamson_5_defaults['ncells_per_edge'],
    dt=williamson_5_defaults['dt'],
    tmax=williamson_5_defaults['tmax'],
    dumpfreq=williamson_5_defaults['dumpfreq'],
    dirname=williamson_5_defaults['dirname'],
    time_parallelism=williamson_5_defaults['time_parallelism'],
    QI=williamson_5_defaults['QI'],
    M=williamson_5_defaults['M'],
    kmax=williamson_5_defaults['kmax'],
    use_pySDC=williamson_5_defaults['use_pySDC'],
    use_adaptivity=williamson_5_defaults['use_adaptivity'],
    Nlevels=williamson_5_defaults['Nlevels'],
    logger_level=williamson_5_defaults['logger_level'],
    mesh=None,
    _ML_is_setup=True,
):
    """
    Run the Williamson 5 test case.

    Args:
        ncells_per_edge (int): number of cells per icosahedron edge
        dt (float): Initial step size
        tmax (float): Time to integrate to
        dumpfreq (int): Output every <dumpfreq> time steps
        dirname (str): Output will go into ./results/<dirname>
        time_parallelism (bool): True for parallel SDC, False for serial
        M (int): Number of collocation nodes
        kmax (int): Max number of SDC iterations
        use_pySDC (bool): Use pySDC as Gusto time integrator or Gusto SDC implementation
        Nlevels (int): Number of SDC levels
        logger_level (int): Logger level
    """
    if not use_pySDC and use_adaptivity:
        raise NotImplementedError('Adaptive step size selection not yet implemented in Gusto')
    if not use_pySDC and Nlevels > 1:
        raise NotImplementedError('Multi-level SDC not yet implemented in Gusto')
    if time_parallelism and Nlevels > 1:
        raise NotImplementedError('Multi-level SDC does not work with MPI parallel sweeper yet')

    # ------------------------------------------------------------------------ #
    # Parameters for test case
    # ------------------------------------------------------------------------ #

    radius = 6371220.0  # planetary radius (m)
    mean_depth = 5960  # reference depth (m)
    g = 9.80616  # acceleration due to gravity (m/s^2)
    u_max = 20.0  # max amplitude of the zonal wind (m/s)
    mountain_height = 2000.0  # height of mountain (m)
    R0 = pi / 9.0  # radius of mountain (rad)
    lamda_c = -pi / 2.0  # longitudinal centre of mountain (rad)
    phi_c = pi / 6.0  # latitudinal centre of mountain (rad)

    # ------------------------------------------------------------------------ #
    # Our settings for this set up
    # ------------------------------------------------------------------------ #

    element_order = 1

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # parallelism
    if time_parallelism:
        ensemble_comm = FiredrakeEnsembleCommunicator(fd.COMM_WORLD, fd.COMM_WORLD.size // M)
        space_comm = ensemble_comm.space_comm
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class

        if ensemble_comm.time_comm.rank > 0:
            dirname = f'{dirname}-{ensemble_comm.time_comm.rank}'
    else:
        ensemble_comm = None
        space_comm = fd.COMM_WORLD
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

    # Domain
    mesh = GeneralIcosahedralSphereMesh(radius, ncells_per_edge, degree=2, comm=space_comm) if mesh is None else mesh
    if Nlevels > 1:
        hierarchy = fd.MeshHierarchy(mesh, Nlevels - 1)
        mesh = hierarchy[-1]
    domain = Domain(mesh, dt, 'BDM', element_order)
    x, y, z = SpatialCoordinate(mesh)
    lamda, phi, _ = lonlatr_from_xyz(x, y, z)

    # Equation: coriolis
    parameters = ShallowWaterParameters(H=mean_depth, g=g)
    Omega = parameters.Omega
    fexpr = 2 * Omega * z / radius

    # Equation: topography
    rsq = min_value(R0**2, (lamda - lamda_c) ** 2 + (phi - phi_c) ** 2)
    r = sqrt(rsq)
    tpexpr = mountain_height * (1 - r / R0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, topog_expr=tpexpr)

    eqns.label_terms(lambda t: not t.has_label(time_derivative), implicit)

    # I/O
    output = OutputParameters(
        dirname=dirname,
        dumplist_latlon=['D'],
        dumpfreq=dumpfreq,
        dump_vtus=True,
        dump_nc=True,
        dumplist=['D', 'topography'],
    )
    diagnostic_fields = [Sum('D', 'topography'), RelativeVorticity(), MeridionalComponent('u'), ZonalComponent('u')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # ------------------------------------------------------------------------ #
    # pySDC parameters: description and controller parameters
    # ------------------------------------------------------------------------ #

    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt
    level_params['residual_type'] = 'full_rel'

    step_params = dict()
    step_params['maxiter'] = kmax

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['num_nodes'] = M
    sweeper_params['QI'] = QI
    sweeper_params['QE'] = 'PIC'
    sweeper_params['comm'] = ensemble_comm
    sweeper_params['initial_guess'] = 'copy'

    problem_params = dict()

    convergence_controllers = {}
    if use_adaptivity:
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.spread_step_sizes import (
            SpreadStepSizesBlockwiseNonMPI,
        )

        convergence_controllers[Adaptivity] = {'e_tol': 1e-6, 'rel_error': True, 'dt_max': 1e4, 'dt_rel_min_slope': 0.5}
        # this is needed because the coupling runs on the controller level and this will almost always overwrite
        convergence_controllers[SpreadStepSizesBlockwiseNonMPI] = {'overwrite_to_reach_Tend': False}

    controller_params = dict()
    controller_params['logger_level'] = logger_level if fd.COMM_WORLD.rank == 0 else 30
    controller_params['mssdc_jac'] = False

    description = dict()
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    description['space_transfer_class'] = MeshToMeshFiredrakeHierarchy

    # ------------------------------------------------------------------------ #
    # petsc solver parameters
    # ------------------------------------------------------------------------ #

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
        'snes_max_it': 999,
        'ksp_max_it': 999,
    }

    # ------------------------------------------------------------------------ #
    # Set Gusto SDC parameters to match the pySDC ones
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
    # Setup time stepper
    # ------------------------------------------------------------------------ #

    if use_pySDC:
        method = pySDC_integrator(description, controller_params, domain=domain, solver_parameters=solver_parameters)
    else:
        method = SDC(**SDC_params, domain=domain)

    stepper = Timestepper(eqns, method, io, spatial_methods=transport_methods)

    # ------------------------------------------------------------------------ #
    # Setup multi-level SDC
    # ------------------------------------------------------------------------ #

    if not _ML_is_setup:
        return stepper

    if Nlevels > 1:
        steppers = [
            None,
        ] * (Nlevels)
        steppers[0] = stepper

        # get different steppers on the different levels
        # recall that the setup of the problems is only finished when the stepper is setup
        for i in range(1, Nlevels):
            steppers[i] = williamson_5(
                ncells_per_edge=ncells_per_edge,
                dt=dt,
                tmax=tmax,
                dumpfreq=dumpfreq,
                dirname=f'{dirname}_unused_{i}',
                time_parallelism=time_parallelism,
                QI=QI,
                M=M,
                kmax=kmax,
                use_pySDC=use_pySDC,
                use_adaptivity=use_adaptivity,
                Nlevels=1,
                mesh=hierarchy[-i - 1],  # mind that the finest level in pySDC is 0, but -1 in hierarchy
                logger_level=50,
                _ML_is_setup=False,
            )

        # update description and setup pySDC again with the discretizations from different steppers
        description['problem_params']['residual'] = [me.scheme.residual for me in steppers]
        description['problem_params']['equation'] = [me.scheme.equation for me in steppers]
        method = pySDC_integrator(description, controller_params, domain=domain, solver_parameters=solver_parameters)
        stepper = Timestepper(eqns, method, io, spatial_methods=transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    uexpr = as_vector([-u_max * y / radius, u_max * x / radius, 0.0])
    Dexpr = mean_depth - tpexpr - (radius * Omega * u_max + 0.5 * u_max**2) * (z / radius) ** 2 / g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(mean_depth)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    if use_pySDC and use_adaptivity:
        # we have to do this for adaptive time stepping, because it is a bit of a mess
        method.timestepper = stepper

    stepper.run(t=0, tmax=tmax)
    return stepper, mesh


# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ncells_per_edge',
        help="The number of cells per edge of icosahedron",
        type=int,
        default=williamson_5_defaults['ncells_per_edge'],
    )
    parser.add_argument('--dt', help="The time step in seconds.", type=float, default=williamson_5_defaults['dt'])
    parser.add_argument(
        "--tmax", help="The end time for the simulation in seconds.", type=float, default=williamson_5_defaults['tmax']
    )
    parser.add_argument(
        '--dumpfreq',
        help="The frequency at which to dump field output.",
        type=int,
        default=williamson_5_defaults['dumpfreq'],
    )
    parser.add_argument(
        '--dirname', help="The name of the directory to write to.", type=str, default=williamson_5_defaults['dirname']
    )
    parser.add_argument(
        '--time_parallelism',
        help="Whether to use parallel diagonal SDC or not.",
        type=str,
        default=williamson_5_defaults['time_parallelism'],
    )
    parser.add_argument('--kmax', help='SDC iteration count', type=int, default=williamson_5_defaults['kmax'])
    parser.add_argument('-M', help='SDC node count', type=int, default=williamson_5_defaults['M'])
    parser.add_argument(
        '--use_pySDC',
        help='whether to use pySDC or Gusto SDC implementation',
        type=str,
        default=williamson_5_defaults['use_pySDC'],
    )
    parser.add_argument(
        '--use_adaptivity',
        help='whether to use adaptive step size selection',
        type=str,
        default=williamson_5_defaults['use_adaptivity'],
    )
    parser.add_argument('--QI', help='Implicit preconditioner', type=str, default=williamson_5_defaults['QI'])
    parser.add_argument(
        '--Nlevels',
        help="Number of SDC levels.",
        type=int,
        default=williamson_5_defaults['Nlevels'],
    )
    args, unknown = parser.parse_known_args()

    options = vars(args)
    for key in ['use_pySDC', 'use_adaptivity', 'time_parallelism']:
        options[key] = options[key] not in ['False', 0, False, 'false']

    williamson_5(**options)
