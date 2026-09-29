import runpy

import pytest


# The parts are notebook-style scripts, so running one is the test. run_module executes them afresh every time.
@pytest.mark.base
@pytest.mark.parametrize(
    'part', ['A_getting_statistics', 'B_adding_statistics', 'C_study_collocations', 'D_writing_solutions_to_file']
)
def test_part(part):
    import matplotlib.pyplot as plt

    try:
        runpy.run_module(f'pySDC.tutorial.step_3.{part}', run_name='__main__')
    finally:
        plt.close('all')  # the parts leave their figures open, as a notebook does
