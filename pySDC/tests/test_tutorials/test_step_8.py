import runpy
import sys

import pytest


# The parts are notebook-style scripts, so running one is the test. run_module executes them afresh every time.
@pytest.mark.base
@pytest.mark.parametrize('part', ['A_adaptive_time_stepping', 'B_multistep_SDC', 'C_iteration_estimator'])
def test_part(part):
    import matplotlib.pyplot as plt

    # another test may have imported the part already, and runpy warns when it would run an imported module again
    sys.modules.pop(f'pySDC.tutorial.step_8.{part}', None)
    try:
        runpy.run_module(f'pySDC.tutorial.step_8.{part}', run_name='__main__')
    finally:
        plt.close('all')  # the parts leave their figures open, as a notebook does
