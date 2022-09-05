import pytest


@pytest.mark.base
def test_matrixbased():
    from pySDC.projects.matrixPFASST.compare_to_matrixbased import main as A

    A()


@pytest.mark.base
def test_propagator():
    from pySDC.projects.matrixPFASST.compare_to_propagator import main as B

    B()
