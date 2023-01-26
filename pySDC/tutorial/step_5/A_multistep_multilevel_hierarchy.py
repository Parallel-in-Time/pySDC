from pathlib import Path


from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to setup a full multi-step multi-level hierarchy
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [31, 15, 7]  # number of degrees of freedom for each level
    problem_params['bc'] = 'dirichlet-zero'  # boundary conditions

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_forced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass parameters for spatial transfer

    # instantiate controller
    controller = controller_nonMPI(num_procs=10, controller_params={}, description=description)

    # check number of levels
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_5_A_out.txt', 'w')
    for i in range(len(controller.MS)):
        out = "Process %2i has %2i levels" % (i, len(controller.MS[i].levels))
        f.write(out + '\n')
        print(out)
    f.close()

    assert all([len(S.levels) == 3 for S in controller.MS]), "ERROR: not all steps have the same number of levels"


if __name__ == "__main__":
    main()
