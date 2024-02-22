from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic import fenics_vortex_2d, fenics_vortex_2d_mass
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

def setup_and_run(variant='mass'):

    num_procs = 1

    t0 = 0
    dt = 0.001
    Tend = 1 * dt

    # initialize level parameters
    level_params = dict()
    if variant == 'mass':
        level_params['restol'] = 5e-09 / 500
    elif variant == 'mass_inv':
        level_params['restol'] = 5e-09
    else:
        raise NotImplementedError('variant unknown')
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['finter'] = True

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = 'MIN-SR-S'
    sweeper_params['QE'] = 'PIC'

    problem_params = dict()
    problem_params['nu'] = 0.01
    problem_params['delta'] = 0.05
    problem_params['rho'] = 50
    problem_params['c_nvars'] = [(32, 32)]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_params'] = problem_params
    if variant == 'mass_inv':
        description['problem_class'] = fenics_vortex_2d
        description['sweeper_class'] = imex_1st_order
    elif variant == 'mass':
        description['problem_class'] = fenics_vortex_2d_mass
        description['sweeper_class'] = imex_1st_order_mass
    else:
        raise NotImplementedError('variant unknown')
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics
    description['space_transfer_params'] = space_transfer_params

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # # compute exact solution and compare
    # uex = P.u_exact(Tend)
    #
    # print('(classical) error at time %s: %s' % (Tend, abs(uex - uend) / abs(uex)))

if __name__ == "__main__":
    setup_and_run(variant='mass')
    setup_and_run(variant='mass_inv')