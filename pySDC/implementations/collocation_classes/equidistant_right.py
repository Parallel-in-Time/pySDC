from warnings import warn

from pySDC.implementations.collocations import Collocation

warn("This import is deprecated and will be removed in future versions."
     "To use this type of collocation, "
     "please use the new generic Collocation class in "
     "pySDC.implementations.collocations, for example:\n"
     "coll = Collocation(num_nodes, tleft, tright, "
     "node_type='EQUID', quadType='RADAU-RIGHT')\n",
     DeprecationWarning, stacklevel=2)


class EquidistantNoLeft(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantNoLeft, self).__init__(
            num_nodes, tleft, tright,
            node_type='EQUID', quad_type='RADAU-RIGHT', useSpline=False)
