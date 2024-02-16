from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh


class DAEMesh(MultiComponentMesh):
    r"""
    Datatype for DAE problems. The solution of the problem can be splitted in the differential part
    and in an algebraic part.

    This data type can be used for the solution of the problem itself as well as for its derivative.
    """

    components = ['diff', 'alg']
