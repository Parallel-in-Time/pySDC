import runpy

import pytest


# The parts are notebook-style scripts, so running one is the test. run_module executes them afresh every time,
# also when another tutorial has imported them already.
@pytest.mark.base
@pytest.mark.parametrize(
    'part',
    [
        'A_spatial_problem_setup',
        'B_spatial_accuracy_check',
        'C_collocation_problem_setup',
        'D_collocation_accuracy_check',
    ],
)
def test_part(part):
    import matplotlib.pyplot as plt

    try:
        runpy.run_module(f'pySDC.tutorial.step_1.{part}', run_name='__main__')
    finally:
        plt.close('all')  # the parts leave their figures open, as a notebook does
