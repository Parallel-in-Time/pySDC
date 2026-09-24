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
                err = conv_data[qd_type][num_nodes]['error'][i]
                ref, kind = test_dict[qd_type][num_nodes][max_iter]
                if kind == 'value':
                    assert np.isclose(err, ref, rtol=0.1, atol=0), f"Got error {err}, expected {ref}"
                else:
                    assert err < ref, f"Got error {err}, expected less than {ref}"


# Errors of the differential variable at tend, measured with the current code. IE after 4/5 iterations
# (1.617e-9, 5.982e-10) is iteration error, deterministic, so it is compared to 10%. LU (4.33e-13,
# 4.06e-14) is close to round-off, so it is only bounded from above, with ~10x headroom.
test_dict = {
    'IE': {3: {4: (1.617e-9, 'value'), 5: (5.982e-10, 'value')}},
    'LU': {3: {4: (5e-12, 'bound'), 5: (5e-13, 'bound')}},
}
