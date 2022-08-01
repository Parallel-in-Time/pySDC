def test_RDC_flavors():
    from pySDC.projects.RDC.vanderpol_MLSDC_PFASST_test import run_RDC

    results = run_RDC(cwd='pySDC/projects/RDC/')

    for item in results:
        assert item[0] < 9e-06, 'Error too high, got %s' % item[0]
        assert item[1] < 6.5, 'Iterations too high, got %s' % item[1]
