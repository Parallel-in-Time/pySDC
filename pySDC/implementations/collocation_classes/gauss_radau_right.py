from pySDC.implementations.collocations import Collocation


class CollGaussRadau_Right(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Right, self).__init__(
            num_nodes, tleft, tright,
            node_type='LEGENDRE', quad_type='RADAU-RIGHT', useSpline=False)
