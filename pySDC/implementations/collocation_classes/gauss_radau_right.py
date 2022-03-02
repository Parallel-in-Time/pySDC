from functools import partial
from warnings import warn

from pySDC.implementations.collocations import Collocation

CollGaussRadau_Right = partial(
    Collocation, node_type='LEGENDRE', quad_type='RADAU-RIGHT', useSpline=False)

warn("This import is deprecated and will be removed in future versions."
     "To use this type of collocation, "
     "please use the new generic Collocation class in "
     "pySDC.implementations.collocations, for example:\n"
     "coll = Collocation(num_nodes, tleft, tright, "
     "node_type='LEGENDRE', quadType='RADAU-RIGHT')\n",
     DeprecationWarning, stacklevel=2)
