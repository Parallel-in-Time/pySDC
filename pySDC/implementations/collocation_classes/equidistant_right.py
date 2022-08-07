from pySDC.core.Collocation import CollBase


class EquidistantNoLeft(CollBase):
    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantNoLeft, self).__init__(
            num_nodes, tleft, tright, node_type='EQUID', quad_type='RADAU-RIGHT', useSpline=False
        )
