import runpy

import pytest


# The parts are notebook-style scripts, so running one is the test. run_module executes them afresh every time.
@pytest.mark.base
@pytest.mark.parametrize('part', ['A_step_data_structure', 'B_my_first_sweeper', 'C_using_pySDCs_frontend'])
def test_part(part):
    runpy.run_module(f'pySDC.tutorial.step_2.{part}', run_name='__main__')
