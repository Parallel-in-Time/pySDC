import numpy as np

from qmat.lagrange import LagrangeApproximation


class DenseOutput:
    """
    Initialize the dense output class with given data.

    Parameters
    ----------
    nodes : list of tuples
        List where each tuple contains:
        (time_step_end, array of nodes within the step)
    uValues : list of tuples
        List where each tuple contains:
        (time_step_end, list of mesh values at each stage)
    """

    def __init__(self, nodes, uValues):
        """Initialization routine"""

        # Extract the stage times and values
        self.nodes = [np.array(entry[1]) for entry in nodes]
        self.uValues = [me[1] for me in uValues]

    def _find_time_interval(self, t):
        r"""
        Find the interval :math:`[t_n, t_{n+1}]` that contains :math:`t`.
        
        Parameters
        ----------
        t : float
            The time at which to find the solution.
        
        Returns
        -------
        index : int
            Index n such that ``t[n] <= t < t[n+1]``.
        """

        for index, tau in enumerate(self.nodes):
            if tau[0] <= t <= tau[-1]:
                return index
        raise ValueError(f"t={t} is out of the range of the provided stage times.")


    def _interpolate(self, t, index):
        r"""
        Interpolate the solution at time :math:`t` for the interval corresponding to the index.

        Parameters:
        t : float
            The time at which to interpolate the solution.
        index : int
            The index of the time interval to use for interpolation.
        
        Returns:
        uValuesInterp : array-like
            The interpolated solution at time t.
        """

        nodes = self.nodes[index]
        uValues = self.uValues[index]

        # Check if t is exactly one of the nodes
        if t in nodes:
            exact_index = np.where(nodes == t)[0][0]
            return uValues[exact_index]

        # Assuming each mesh value has the same dimensionality, so we use the first value's shape to determine dimensions
        uFirstNode = uValues[0]
        components = [None] if not hasattr(uFirstNode, 'components') else getattr(uFirstNode, 'components')

        # Interpolate each dimension separately along each component
        ElementsInterp = []
        for component in components:
            uValuesComponents = [getattr(value, component) for value in uValues] if component is not None else uValues
            numSub = len(uValuesComponents[0])

            SubElementsInterp = []
            for subIndex in range(numSub):
                uValuesSub = [subElement[subIndex] for subElement in uValuesComponents]

                # Create an LagrangeApproximation object for this dimension
                sol = LagrangeApproximation(points=nodes, fValues=uValuesSub)

                # Interpolate the value at time t for this dimension
                SubElementsInterp.append(sol(t))
            ElementsInterp += SubElementsInterp

        ElementsInterp = np.asarray(ElementsInterp)
        uValuesInterp = self._recover_datatype(ElementsInterp, uFirstNode.shape, type(uFirstNode))
        return uValuesInterp

    def __call__(self, t):
        r"""
        Evaluate the dense output at time :math:`t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the solution.

        Returns
        -------
        y : array-like
            Solution at time :math:`t`.
        """

        index = self._find_time_interval(t)
        return self._interpolate(t, index)

    def _recover_datatype(self, mesh, shape, type):
        return mesh.reshape(shape).view(type)