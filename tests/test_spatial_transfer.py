import numpy as np
from collections import namedtuple

from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from implementations.problem_classes.HeatEquation_2D_FD_periodic import heat2d_periodic
from implementations.datatype_classes.mesh import mesh
from implementations.transfer_classes.TransferMesh import mesh_to_mesh

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', ('nvars_fine', 'iorder'))


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
    # loop over list of interpolation orders
    for iorder in iorder_list:
        # loop over two consecutive errors/nvars pairs
        for i in range(1,len(nvars_fine_list)):

            # get ids
            id = ID(nvars_fine=nvars_fine_list[i], iorder=iorder)
            id_prev = ID(nvars_fine=nvars_fine_list[i-1], iorder=iorder)

            # compute order as log(prev_error/this_error)/log(this_nvars/old_nvars)
            if type(nvars_fine_list[i]) is tuple:
                nvars = nvars_fine_list[i][0]
                nvars_prev = nvars_fine_list[i-1][0]
            else:
                nvars = nvars_fine_list[i]
                nvars_prev = nvars_fine_list[i-1]
            computed_order = np.log(results[id_prev]/results[id])/np.log(nvars/nvars_prev)
            order.append((nvars_fine_list[i], iorder, computed_order))

    return order


def test_mesh_to_mesh_1d_dirichlet():
    """
    A simple test program to test dirichlet interpolation order in space
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 3  # frequency for the test value

    # initialize transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2

    iorder_list = [2,4,6,8]
    nvars_fine_list = [2**p-1 for p in range(5,9)]

    # set up dictionary to store results (plus lists)
    results = {}
    results['nvars_fine_list'] = nvars_fine_list
    results['iorder_list'] = iorder_list

    # loop over interpolation orders and number of DOFs
    for iorder in iorder_list:

        space_transfer_params['iorder'] = iorder

        for nvars_fine in nvars_fine_list:

            # instantiate fine problem
            problem_params['nvars'] = nvars_fine  # number of degrees of freedom
            Pfine = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate coarse problem
            problem_params['nvars'] = int((nvars_fine + 1) / 2.0 - 1)
            Pcoarse = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate spatial interpolation
            T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

            # set exact fine solution to compare with
            uexact_fine = Pfine.u_exact(t=0)

            # set exact coarse solution as source
            uexact_coarse = Pcoarse.u_exact(t=0)

            # do the interpolation/prolongation
            uinter = T.prolong(uexact_coarse)

            # compute error and store
            err = abs(uinter-uexact_fine)
            id = ID(nvars_fine=nvars_fine, iorder=iorder)
            results[id] = err

    orders = get_accuracy_orders(results)
    for p in range(len(orders)):
        # print(abs(orders[p][1]-orders[p][2])/orders[p][1])
        assert abs(orders[p][1]-orders[p][2])/orders[p][1] < 0.151, 'ERROR: did not get expected orders for interpolation, got %s' %str(orders[p])

def test_mesh_to_mesh_1d_periodic():
    """
    A simple test program to test periodic interpolation order in space
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['c'] = 0.1  # advection coefficient
    problem_params['freq'] = 4  # frequency for the test value

    # initialize transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['periodic'] = True

    iorder_list = [2,4,6,8]
    nvars_fine_list = [2**p for p in range(5,9)]

    # set up dictionary to store results (plus lists)
    results = {}
    results['nvars_fine_list'] = nvars_fine_list
    results['iorder_list'] = iorder_list

    # loop over interpolation orders and number of DOFs
    for iorder in iorder_list:

        space_transfer_params['iorder'] = iorder

        for nvars_fine in nvars_fine_list:

            # instantiate fine problem
            problem_params['nvars'] = nvars_fine  # number of degrees of freedom
            Pfine = advection1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate coarse problem
            problem_params['nvars'] = int(nvars_fine / 2)
            Pcoarse = advection1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate spatial interpolation
            T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

            # set exact fine solution to compare with
            uexact_fine = Pfine.u_exact(t=0)

            # set exact coarse solution as source
            uexact_coarse = Pcoarse.u_exact(t=0)

            # do the interpolation/prolongation
            uinter = T.prolong(uexact_coarse)

            # compute error and store
            err = abs(uinter-uexact_fine)
            id = ID(nvars_fine=nvars_fine, iorder=iorder)
            results[id] = err

    orders = get_accuracy_orders(results)

    print(orders)

    for p in range(len(orders)):
        # print(abs(orders[p][1]-orders[p][2])/orders[p][1])
        assert abs(orders[p][1]-orders[p][2])/orders[p][1] < 0.051, 'ERROR: did not get expected orders for interpolation, got %s' %str(orders[p])


def test_mesh_to_mesh_2d_periodic():
    """
        A simple test program to test periodic interpolation order in 2d
        """

    # initialize problem parameters
    problem_params = {}
    problem_params['c'] = 0.1  # advection coefficient
    problem_params['freq'] = 4  # frequency for the test value

    # initialize transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['periodic'] = True

    iorder_list = [2, 4, 6, 8]
    nvars_fine_list = [(2 ** p, 2 ** p) for p in range(5, 9)]

    # set up dictionary to store results (plus lists)
    results = {}
    results['nvars_fine_list'] = nvars_fine_list
    results['iorder_list'] = iorder_list

    # loop over interpolation orders and number of DOFs
    for iorder in iorder_list:

        space_transfer_params['iorder'] = iorder

        for nvars_fine in nvars_fine_list:
            # instantiate fine problem
            problem_params['nvars'] = nvars_fine  # number of degrees of freedom
            Pfine = heat2d_periodic(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate coarse problem
            problem_params['nvars'] = (int(nvars_fine[0] / 2), int(nvars_fine[1] / 2))
            Pcoarse = heat2d_periodic(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

            # instantiate spatial interpolation
            T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

            # set exact fine solution to compare with
            uexact_fine = Pfine.u_exact(t=0)

            # set exact coarse solution as source
            uexact_coarse = Pcoarse.u_exact(t=0)

            # do the interpolation/prolongation
            uinter = T.prolong(uexact_coarse)

            # compute error and store
            err = abs(uinter - uexact_fine)
            id = ID(nvars_fine=nvars_fine, iorder=iorder)
            results[id] = err

    orders = get_accuracy_orders(results)

    print(orders)

    for p in range(len(orders)):
        # print(abs(orders[p][1] - orders[p][2]) / orders[p][1])
        assert abs(orders[p][1] - orders[p][2]) / orders[p][
            1] < 0.115, 'ERROR: did not get expected orders for interpolation, got %s' % str(orders[p])

if __name__ == "__main__":
    test_mesh_to_mesh_1d_dirichlet()
    pass
