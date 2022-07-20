from pySDC.core import CollBase


class CollGaussRadau_Right(CollBase):

    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussRadau_Right, self).__init__(
            num_nodes, tleft, tright,
            node_type='LEGENDRE', quad_type='RADAU-RIGHT', useSpline=False)
