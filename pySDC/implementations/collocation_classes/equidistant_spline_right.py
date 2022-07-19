from pySDC.implementations.collocations import Collocation


class EquidistantSpline_Right(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantSpline_Right, self).__init__(
            num_nodes, tleft, tright,
            node_type='EQUID', quad_type='RADAU-RIGHT', useSpline=True)
