from pySDC.core.Collocation import CollBase


class CollGaussRadau_Left(CollBase):
    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Left, self).__init__(
            num_nodes, tleft, tright, node_type='LEGENDRE', quad_type='RADAU-LEFT', useSpline=False
        )
