from pySDC.core.errors import DataError


class MultiComponentContainer(object):
    r"""
    Datatype with multiple components, each one an object of its own.

    This is the counterpart of ``MultiComponentMeshMixin`` for backends whose data cannot simply grow an axis:
    a FEniCS ``Function`` lives on a ``FunctionSpace`` and a PETSc ``Vec`` on a ``DMDA``, so the components
    cannot be views into one contiguous array and have to be separate objects instead.

    To make a specific multi-component datatype, derive from this class, list the components as strings in
    ``components``, and say what a single component is in ``component_type``. An example:

    ```
    class rhs_fenics_mesh(MultiComponentContainer):
        components = ['impl', 'expl']
        component_type = fenics_mesh
    ```

    Instantiating such a datatype builds one component of ``component_type`` per name, either by copying the
    components of another instance or by passing ``init`` and ``val`` on to each of them. The arithmetic is
    applied component by component, so the component type is what decides which operands it accepts.
    """

    components = []
    component_type = None

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: either another instance of this datatype, or whatever ``component_type`` accepts
            val: value to initialize the components with, if they are not copied
        """
        if isinstance(init, type(self)):
            for name in self.components:
                setattr(self, name, self.component_type(getattr(init, name)))
        else:
            for name in self.components:
                setattr(self, name, self.component_type(init, val=val))

    def _apply(self, other, operation):
        """
        Apply ``operation`` to each component of this datatype and the matching one of ``other``.

        Args:
            other: another instance of this datatype
            operation (callable): takes the two components and returns the new one

        Returns:
            a new instance of this datatype
        """
        if not isinstance(other, type(self)):
            raise DataError(f'Type error: cannot combine {type(other)} with {type(self)}')

        me = type(self)(self)
        for name in self.components:
            setattr(me, name, operation(getattr(self, name), getattr(other, name)))
        return me

    def __add__(self, other):
        """
        Overloading the addition operator

        Args:
            other: datatype of the same type to be added
        Raises:
            DataError: if other is not of the same type
        Returns:
            sum of caller and other, component by component
        """
        return self._apply(other, lambda a, b: a + b)

    def __sub__(self, other):
        """
        Overloading the subtraction operator

        Args:
            other: datatype of the same type to be subtracted
        Raises:
            DataError: if other is not of the same type
        Returns:
            difference between caller and other, component by component
        """
        return self._apply(other, lambda a, b: a - b)

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator

        Args:
            other (float): factor
        Raises:
            DataError: if the component type does not accept the factor
        Returns:
            copy of the caller scaled by the factor
        """
        me = type(self)(self)
        for name in self.components:
            setattr(me, name, other * getattr(self, name))
        return me
