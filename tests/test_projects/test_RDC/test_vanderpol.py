import os
from projects.RDC.vanderpol_error_test import plot_RDC_results

def test_vanderpol_visualization():

    assert os.path.isfile('projects/RDC/data/vdp_ref.npy'), 'ERROR: reference solution does not exist'
    plot_RDC_results(cwd='projects/RDC/')