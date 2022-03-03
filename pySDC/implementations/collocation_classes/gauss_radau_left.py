from warnings import warn

from pySDC.implementations.collocations import Collocation

warn("This import is deprecated and will be removed in future versions."
     "To use this type of collocation, "
     "please use the new generic Collocation class in "
     "pySDC.implementations.collocations, for example:\n"
     "coll = Collocation(num_nodes, tleft, tright, "
     "node_type='LEGENDRE', quadType='RADAU-LEFT')\n",
     DeprecationWarning, stacklevel=2)


class CollGaussRadau_Left(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Left, self).__init__(
            num_nodes, tleft, tright,
            node_type='LEGENDRE', quad_type='RADAU-LEFT', useSpline=False)
