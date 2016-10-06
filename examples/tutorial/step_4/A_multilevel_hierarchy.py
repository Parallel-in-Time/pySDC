import numpy as np
from collections import namedtuple

from pySDC.Step import step
from implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from implementations.transfer_classes.TransferMesh_1D_IMEX import mesh_to_mesh_1d_dirichlet


# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', ('nvars_fine', 'iorder'))

def main():
    """
    A simple test program to run IMEX SDC for a single time step
    """

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 3  # frequency for the test value


    # initialize transfer parameters
    transfer_params = {}
    transfer_params['rorder'] = 2

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_forced
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = {}
    description['step_params'] = {}
    description['transfer_class'] = mesh_to_mesh_1d_dirichlet

    iorder_list = [2,4,6,8]
    nvars_fine_list = [2**p-1 for p in range(5,9)]
    results = {}
    results['nvars_fine_list'] = nvars_fine_list
    results['iorder_list'] = iorder_list
    for iorder in iorder_list:

        transfer_params['iorder'] = iorder
        description['transfer_params'] = transfer_params

        for nvars_fine in nvars_fine_list:

            print('Working on iorder = %2i and nvars_fine = %4i...' %(iorder,nvars_fine))

            problem_params['nvars'] = [nvars_fine, int((nvars_fine+1)/2.0-1)]  # number of degrees of freedom
            description['problem_params'] = problem_params


            # now the description contains more or less everything we need to create a step
            S = step(description=description)

            T = mesh_to_mesh_1d_dirichlet(fine_level=S.levels[0], coarse_level=S.levels[1], params=transfer_params)

            Pfine = S.levels[0].prob
            Pcoarse = S.levels[1].prob

            xvalues_fine = np.array([(i + 1) * Pfine.dx for i in range(Pfine.nvars)])
            uexact_fine = Pfine.dtype_u(0)
            uexact_fine.values = np.sin(np.pi * Pfine.freq * xvalues_fine)

            xvalues_coarse = np.array([(i + 1) * Pcoarse.dx for i in range(Pcoarse.nvars)])
            uexact_coarse = Pfine.dtype_u(0)
            uexact_coarse.values = np.sin(np.pi * Pcoarse.freq * xvalues_coarse)

            uinter = T.prolong_space(uexact_coarse)

            err = abs(uinter-uexact_fine)

            id = ID(nvars_fine=nvars_fine, iorder=iorder)
            results[id] = err

    print('Running order checks...')
    orders = get_accuracy_orders(results)
    for p in range(len(orders)):
        assert abs(orders[p][1]-orders[p][2])/orders[p][1] < 0.15, 'ERROR: did not get expected orders for interpolation, got %s' %str(orders[p])
    print('...got what we expected!')

def get_accuracy_orders(results):
    """
    Routine to compute the order of accuracy in space

    Args:
        results: the dictionary containing the errors

    Returns:
        the list of orders
    """

    # retrieve the list of nvars from results
    assert 'nvars_fine_list' in results, 'ERROR: expecting the list of nvars in the results dictionary'
    assert 'iorder_list' in results, 'ERROR: expecting the list of iorders in the results dictionary'
    nvars_fine_list = sorted(results['nvars_fine_list'])
    iorder_list = sorted(results['iorder_list'])

    order = []

    for iorder in iorder_list:
        # loop over two consecutive errors/nvars pairs
        for i in range(1,len(nvars_fine_list)):

            # get ids
            id = ID(nvars_fine=nvars_fine_list[i], iorder=iorder)
            id_prev = ID(nvars_fine=nvars_fine_list[i-1], iorder=iorder)

            # compute order as log(prev_error/this_error)/log(this_nvars/old_nvars) <-- depends on the sorting of the list!
            computed_order = np.log(results[id_prev]/results[id])/np.log(nvars_fine_list[i]/nvars_fine_list[i-1])
            order.append((nvars_fine_list[i], iorder, computed_order))

    return order





if __name__ == "__main__":
    main()
