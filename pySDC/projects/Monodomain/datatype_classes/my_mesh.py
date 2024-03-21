from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh


class imexexp_mesh(MultiComponentMesh):
    components = ['impl', 'expl', 'exp']


def get_np_list(vec):
    return [vec[i] for i in range(vec.shape[0])]
