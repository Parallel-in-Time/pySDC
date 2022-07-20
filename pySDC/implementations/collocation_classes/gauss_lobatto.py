from pySDC.implementations.collocations import Collocation


class CollGaussLobatto(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLobatto, self).__init__(
            num_nodes, tleft, tright,
            node_type='LEGENDRE', quad_type='LOBATTO', useSpline=False)
