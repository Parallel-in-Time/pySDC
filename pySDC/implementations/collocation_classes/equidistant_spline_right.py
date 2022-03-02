from warnings import warn

from pySDC.implementations.collocations import Collocation

warn("This import is deprecated and will be removed in future versions."
     "To use this type of collocation, "
     "please use the new generic Collocation class in "
     "pySDC.implementations.collocations, for example:\n"
     "coll = Collocation(num_nodes, tleft, tright, "
     "node_type='EQUID', quadType='RADAU-RIGHT', useSpline=True)\n",
     DeprecationWarning, stacklevel=2)


class EquidistantSpline_Right(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantSpline_Right, self).__init__(
            num_nodes, tleft, tright,
            node_type='EQUID', quad_type='RADAU-RIGHT', useSpline=True)
