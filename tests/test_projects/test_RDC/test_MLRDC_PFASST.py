from projects.RDC.vanderpol_MLSDC_PFASST_test import run_RDC

def test_RDC_flavors():
    results = run_RDC(cwd='projects/RDC/')

    for item in results:
        assert item[0] < 9E-06, 'Error too high, got %s' % item
        assert item[1] < 12, 'Iterations too high, got %s' % item