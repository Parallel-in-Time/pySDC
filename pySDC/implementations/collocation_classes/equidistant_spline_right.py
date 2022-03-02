from functools import partial
from warnings import warn

from pySDC.implementations.collocations import Collocation

EquidistantSpline_Right = partial(
    Collocation, node_type='EQUID', quad_type='RADAU-RIGHT', useSpline=True)

warn("This import is deprecated and will be removed in future versions."
     "To use this type of collocation, "
     "please use the new generic Collocation class in "
     "pySDC.implementations.collocations, for example:\n"
     "coll = Collocation(num_nodes, tleft, tright, "
     "node_type='EQUID', quadType='RADAU-RIGHT', useSpline=True)\n",
     DeprecationWarning, stacklevel=2)
