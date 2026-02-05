import pytest
import subprocess
import os
import warnings


@pytest.mark.fenics
@pytest.mark.mpi4py
@pytest.mark.slow
def test_3rd_order():

    from pySDC.projects.StroemungsRaum.run_accuracy.run_accuracy_3rd_order_heat_equation_FEniCS import (
        run_accuracy,
        plot_accuracy,
    )

    results = run_accuracy()
    plot_accuracy(results)

    assert os.path.isfile(
        'data/heat_equation_3rd_order_time_FEniCS.png'
    ), 'ERROR: 3rd order convergence plot has not been created'


"""    
@pytest.mark.fenics
@pytest.mark.mpi4py
@pytest.mark.slow    
def test_5th_order():
    
    from pySDC.projects.StroemungsRaum.run_accuracy.run_accuracy_5th_order_heat_equation_FEniCS import run_accuracy, plot_accuracy
    
    results = run_accuracy()
    plot_accuracy(results)

    
    assert os.path.isfile('data/heat_equation_5th_Order_time_FEniCS.png'), 'ERROR: 5th order convergence plot has not been created'
"""
