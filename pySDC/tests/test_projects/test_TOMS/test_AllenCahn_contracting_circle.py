import pytest
import dill
import os

results = {}

@pytest.mark.base
@pytest.mark.slow
@pytest.mark.order(1)
@pytest.mark.parametrize("variant", ['multi-implicit', 'semi-implicit', 'fully-implicit', 'semi-implicit_v2', 'multi-implicit_v2'])
@pytest.mark.parametrize("inexact", [False, True])
def test_AllenCahn_contracting_circle(variant, inexact):
    from pySDC.projects.TOMS.AllenCahn_contracting_circle import run_SDC_variant
    results[(variant, 'exact' if not inexact else 'inexact')] = run_SDC_variant(variant=variant, inexact=inexact)

@pytest.mark.base
@pytest.mark.order(2)
def test_show_results():
    from pySDC.projects.TOMS.AllenCahn_contracting_circle import show_results

    # dump result
    cwd = 'pySDC/projects/TOMS/'
    fname = 'data/results_SDC_variants_AllenCahn_1E-03'
    file = open(cwd + fname + '.pkl', 'wb')
    dill.dump(results, file)
    file.close()
    assert os.path.isfile(cwd + fname + '.pkl'), 'ERROR: dill did not create file'

    # visualize
    show_results(fname, cwd=cwd)

