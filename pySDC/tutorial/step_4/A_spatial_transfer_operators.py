from collections import namedtuple

import numpy as np

from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.tutorial.step_1.B_spatial_accuracy_check import get_accuracy_order

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars_fine')


def main():
    """
    A simple test program to test interpolation order in space
    """

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 3  # frequency for the test value

    # initialize transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 4

    nvars_fine_list = [2 ** p - 1 for p in range(5, 10)]

    # set up dictionary to store results (plus lists)
    results = dict()
    results['nvars_list'] = nvars_fine_list

    for nvars_fine in nvars_fine_list:
        print('Working on nvars_fine = %4i...' % nvars_fine)

        # instantiate fine problem
        problem_params['nvars'] = nvars_fine  # number of degrees of freedom
        Pfine = heat1d(problem_params=problem_params, dtype_u=parallel_mesh, dtype_f=parallel_mesh)

        # instantiate coarse problem using half of the DOFs
        problem_params['nvars'] = int((nvars_fine + 1) / 2.0 - 1)
        Pcoarse = heat1d(problem_params=problem_params, dtype_u=parallel_mesh, dtype_f=parallel_mesh)

        # instantiate spatial interpolation
        T = mesh_to_mesh(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

        # set exact fine solution to compare with
        xvalues_fine = np.array([(i + 1) * Pfine.dx for i in range(Pfine.params.nvars)])
        uexact_fine = Pfine.dtype_u(Pfine.init)
        uexact_fine[:] = np.sin(np.pi * Pfine.params.freq * xvalues_fine)

        # set exact coarse solution as source
        xvalues_coarse = np.array([(i + 1) * Pcoarse.dx for i in range(Pcoarse.params.nvars)])
        uexact_coarse = Pfine.dtype_u(Pcoarse.init)
        uexact_coarse[:] = np.sin(np.pi * Pcoarse.params.freq * xvalues_coarse)

        # do the interpolation/prolongation
        uinter = T.prolong(uexact_coarse)

        # compute error and store
        id = ID(nvars_fine=nvars_fine)
        results[id] = abs(uinter - uexact_fine)

    # print out and check
    print('Running order checks...')
    orders = get_accuracy_order(results)
    f = open('step_4_A_out.txt', 'w')
    for p in range(len(orders)):
        out = 'Expected order %2i, got order %5.2f, deviation of %5.2f%%' \
              % (space_transfer_params['iorder'], orders[p],
                 100 * abs(space_transfer_params['iorder'] - orders[p]) / space_transfer_params['iorder'])
        f.write(out + '\n')
        print(out)
        assert abs(space_transfer_params['iorder'] - orders[p]) / space_transfer_params['iorder'] < 0.05, \
            'ERROR: did not get expected orders for interpolation, got %s' % str(orders[p])
    f.close()
    print('...got what we expected!')


if __name__ == "__main__":
    main()
