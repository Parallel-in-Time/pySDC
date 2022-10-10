from pathlib import Path

from pySDC.core.Step import step

from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to set up a full step hierarchy
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [5, 3]

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
    space_transfer_params['iorder'] = 2

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_unforced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_LU  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # now the description contains more or less everything we need to create a step with multiple levels
    S = step(description=description)

    # print out and check
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_4_B_out.txt', 'w')
    for l in range(len(S.levels)):
        L = S.levels[l]
        out = 'Level %2i: nvars = %4i -- nnodes = %2i' % (l, L.prob.params.nvars[0], L.sweep.coll.num_nodes)
        f.write(out + '\n')
        print(out)
        assert L.prob.params.nvars[0] == problem_params['nvars'][min(l, len(problem_params['nvars']) - 1)], (
            "ERROR: number of DOFs is not correct on this level, got %s" % L.prob.params.nvars
        )
        assert L.sweep.coll.num_nodes == sweeper_params['num_nodes'][min(l, len(sweeper_params['num_nodes']) - 1)], (
            "ERROR: number of nodes is not correct on this level, got %s" % L.sweep.coll.num_nodes
        )
    f.close()


if __name__ == "__main__":
    main()
