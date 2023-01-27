import pytest
import numpy as np


@pytest.mark.base
def test_main():
    from pySDC.projects.DAE.run.run_iteration_test import setup, run

    # get setup data
    description, controller_params, run_params = setup()
    # update run_params
    run_params['t0'] = 0.0
    run_params['tend'] = 0.1
    run_params['max_iter_list'] = [4, 5]
    run_params['qd_list'] = ['IE', 'LU']
    run_params['num_nodes_list'] = [3]
    conv_data = run(description, controller_params, run_params)

    # validate results
    for qd_type in run_params['qd_list']:
        for num_nodes in run_params['num_nodes_list']:
            for i, max_iter in enumerate(run_params['max_iter_list']):
                assert np.isclose(
                    conv_data[qd_type][num_nodes]['error'][i], test_dict[qd_type][num_nodes][max_iter], atol=1e-5
                ), f"ERROR: error bound not fulfilled.\n Got {conv_data[qd_type][num_nodes]['error'][i]}\n Expecting less than {test_dict[qd_type][num_nodes][max_iter]}"


# Dictionary of test values for use with:
#   max_iter_low = 4
#   max_iter_high = 6
#   qd_list = ['IE', 'LU']
#   num_nodes_list = [3]

test_dict = {'IE': {3: {4: 1.6e-7, 5: 6e-8}}, 'LU': {3: {4: 4.1e-10, 5: 3.8e-13}}}
