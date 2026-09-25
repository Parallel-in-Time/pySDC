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
                err = conv_data[qd_type][num_nodes]['error'][i]
                ref, kind = test_dict[qd_type][num_nodes][dt]
                if kind == 'value':
                    assert np.isclose(err, ref, rtol=0.1, atol=0), f"Got error {err}, expected {ref}"
                else:
                    assert err < ref, f"Got error {err}, expected less than {ref}"


# Errors of the differential variable at tend, measured with the current code. At dt=1e-2 the error is
# the collocation error (1.380e-12 for IE, 1.375e-12 for LU), compared to 10%. At dt=1e-3 it is
# round-off (1.6e-14, 2.0e-14), so it is only bounded from above, with ~10x headroom.
test_dict = {
    'IE': {3: {1e-2: (1.38e-12, 'value'), 1e-3: (2e-13, 'bound')}},
    'LU': {3: {1e-2: (1.375e-12, 'value'), 1e-3: (2e-13, 'bound')}},
}
