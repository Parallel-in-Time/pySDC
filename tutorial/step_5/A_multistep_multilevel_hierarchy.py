from pySDC.Step import step

from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh_dirichlet
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.5

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['do_LU'] = True

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1      # diffusion coefficient
    problem_params['freq'] = 4      # frequency for the test value
    problem_params['nvars'] = [31,15,7]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d_forced                    # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = rhs_imex_mesh                          # pass data type for f
    description['sweeper_class'] = imex_1st_order                   # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_dirichlet # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # instantiate controller
    controller = allinclusive_classic_nonMPI(num_procs=10, controller_params={}, description=description)

    # check number of levels
    for i in range(len(controller.MS)):
        print("Process %2i has %2i levels" %(i,len(controller.MS[i].levels)))

    assert all([len(S.levels) == 3 for S in controller.MS]), "ERROR: not all steps have the same number of levels"

if __name__ == "__main__":
    main()
