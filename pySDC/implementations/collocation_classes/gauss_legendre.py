from pySDC.implementations.collocations import Collocation


class CollGaussLegendre(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(CollGaussLegendre, self).__init__(
            num_nodes, tleft, tright,
            node_type='LEGENDRE', quad_type='GAUSS', useSpline=False)
