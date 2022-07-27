from pySDC.core import CollBase


class Equidistant(CollBase):

    def __init__(self, num_nodes, tleft, tright):
        super(Equidistant, self).__init__(
            num_nodes, tleft, tright,
            node_type='EQUID', quad_type='LOBATTO', useSpline=False)
