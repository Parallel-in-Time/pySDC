import runpy

import pytest


# The parts are notebook-style scripts, so running one is the test. run_module executes them afresh every time.
@pytest.mark.base
@pytest.mark.parametrize(
    'part',
    ['A_spatial_transfer_operators', 'B_multilevel_hierarchy', 'C_SDC_vs_MLSDC', 'D_MLSDC_with_particles'],
)
def test_part(part):
    import matplotlib.pyplot as plt

    try:
        runpy.run_module(f'pySDC.tutorial.step_4.{part}', run_name='__main__')
    finally:
        plt.close('all')  # the parts leave their figures open, as a notebook does
