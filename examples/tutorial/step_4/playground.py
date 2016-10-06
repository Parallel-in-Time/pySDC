import numpy as np

from numpy.polynomial.polynomial import polyval

from pySDC.Step import step
import pySDC.Plugins.transfer_helper as th
from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.generic_LU import generic_LU
from implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_dirichlet

def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = [0.1, 0.2]

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [9, 7, 5, 3]

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [31, 15, 7]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d                           # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = mesh                                   # pass data type for f
    description['sweeper_class'] = generic_LU                       # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_1d_dirichlet # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # now the description contains more or less everything we need to create a step with multiple levels
    S = step(description=description)

    # print out and check
    for l in range(len(S.levels)):
        L = S.levels[l]
        print('Level %2i: nvars = %4i -- nnodes = %2i -- dt = %4.2f' %(l, L.prob.params.nvars, L.sweep.coll.num_nodes, L.dt))
        assert L.prob.params.nvars == problem_params['nvars'][min(l,len(problem_params['nvars'])-1)], \
            "ERROR: number of DOFs is not correct on this level, got %s" %L.prob.params.nvars
        assert L.sweep.coll.num_nodes == sweeper_params['num_nodes'][min(l,len(sweeper_params['num_nodes'])-1)], \
            "ERROR: number of nodes is not correct on this level, got %s" %L.sweep.coll.num_nodes
        assert L.dt == level_params['dt'][min(l,len(level_params['dt'])-1)], \
            "ERROR: dt is not correct on this level, got %s" %L.dt


    for l in range(1,len(S.levels)):
        Lf = S.levels[l-1]
        Lg = S.levels[l]
        # print(Lf.sweep.coll.nodes, Lg.sweep.coll.nodes)
        fine_grid = np.concatenate(([0], Lf.sweep.coll.nodes))
        coarse_grid = np.concatenate(([0], Lg.sweep.coll.nodes))

        ufine = Lf.prob.dtype_u(Lf.sweep.coll.num_nodes+1)
        ucoarse = Lg.prob.dtype_u(Lg.sweep.coll.num_nodes+1)

        for order in range(2,Lg.sweep.coll.num_nodes+2):

            Pcoll = th.interpolation_matrix_1d(fine_grid, coarse_grid, k=order)
            Rcoll = th.restriction_matrix_1d(fine_grid, coarse_grid, k=order)

            for polyorder in range(1,order+2):
                coeff = np.random.rand(polyorder)
                ufine.values = polyval(fine_grid,coeff)
                ucoarse.values = polyval(coarse_grid,coeff)

                uinter = Pcoll.dot(ucoarse.values)
                urestr = Rcoll.dot(ufine.values)

                err_inter = np.linalg.norm(uinter-ufine.values, np.inf)
                err_restr = np.linalg.norm(urestr-ucoarse.values, np.inf)
                if polyorder <= order:
                    assert err_inter < 2E-15, "ERROR: Q-interpolation order is not reached, got %s" %err_inter
                    assert err_restr < 2E-15, "ERROR: Q-restriction order is not reached, got %s" % err_restr
                else:
                    assert err_inter > 2E-15, "ERROR: Q-interpolation order is higher than expected, got %s" % polyorder
                    assert err_restr > 2E-15, "ERROR: Q-restriction order is higher than expected, got %s" % polyorder


if __name__ == "__main__":
    main()
