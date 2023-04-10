import pytest
import numpy as np


@pytest.mark.base
def test_main():
    from pySDC.projects.DAE.run.run_convergence_test import setup, run

    # get setup data
    description, controller_params, run_params = setup()
    # update run_params
    num_samples = 2
    run_params = dict()
    run_params['t0'] = 0.0
    run_params['tend'] = 1e-1
    run_params['dt_list'] = np.logspace(-2, -3, num=num_samples)
    run_params['qd_list'] = ['IE', 'LU']
    run_params['num_nodes_list'] = [3]
    conv_data = run(description, controller_params, run_params)

    # validate results
    for qd_type in run_params['qd_list']:
        for num_nodes in run_params['num_nodes_list']:
            for i, dt in enumerate(run_params['dt_list']):
                assert np.isclose(
                    conv_data[qd_type][num_nodes]['error'][i], test_dict[qd_type][num_nodes][dt], atol=1e-5
                ), f"ERROR: error bound not fulfilled.\n Got {conv_data[qd_type][num_nodes]['error'][i]}\n Expecting less than {test_dict[qd_type][num_nodes][dt]}"


# Dictionary of test values for use with:
#   num_samples = 2
#   qd_list = ['IE', 'LU']
#   num_nodes_list = [3]

test_dict = {'IE': {3: {1e-2: 1.4e-12, 1e-3: 2.0e-14}}, 'LU': {3: {1e-2: 1.4e-12, 1e-3: 2.2e-14}}}
