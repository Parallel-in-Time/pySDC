from pySDC.implementations.collocations import Collocation


class Equidistant(Collocation):

    def __init__(self, num_nodes, tleft, tright):
        super(Equidistant, self).__init__(
            num_nodes, tleft, tright,
            node_type='EQUID', quad_type='LOBATTO', useSpline=False)
