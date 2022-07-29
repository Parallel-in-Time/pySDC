from pySDC.projects.matrixPFASST.compare_to_matrixbased import main as A
from pySDC.projects.matrixPFASST.compare_to_propagator import main as B


def test_matrixbased():
    A()


def test_propagator():
    B()
