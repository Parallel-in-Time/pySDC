def test_matrixbased():
    from pySDC.projects.matrixPFASST.compare_to_matrixbased import main as A

    A()


def test_propagator():
    from pySDC.projects.matrixPFASST.compare_to_propagator import main as B

    B()
