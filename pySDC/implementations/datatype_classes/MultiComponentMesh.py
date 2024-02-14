import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh


class MultiComponentMesh(mesh):
    r"""
    Base class for meshes with multiple components.

    Attributes
    ----------
    components : list
        Components for mesh datatype. They are stored as attributes.
    """

    components = []

    def __new__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None, *args, **kwargs):
        if isinstance(init, tuple) and isinstance(init[0], int):
            obj = super().__new__(cls, ((len(cls.components), init[0]), *init[1:]), *args, **kwargs)
        elif isinstance(init, tuple) and isinstance(init[0], tuple):
            obj = np.ndarray.__new__(
                cls, (len(cls.components), *init[0]), dtype=init[2], buffer=buffer, offset=offset, strides=strides, order=order
            )
            obj.fill(val)
            obj._comm = init[1]
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        for comp, i in zip(cls.components, range(len(cls.components))):
            obj.__dict__[comp] = obj[i]
        return obj

    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs).view(type(self))

        if type(self) == type(results) and self.flags['OWNDATA']:
            for comp, i in zip(self.components, range(len(self.components))):
                results.__dict__[comp] = results[i]
        return results
