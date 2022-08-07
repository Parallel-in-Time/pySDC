from pySDC.core.Collocation import CollBase


class EquidistantInner(CollBase):
    def __init__(self, num_nodes, tleft, tright):
        super(EquidistantInner, self).__init__(
            num_nodes, tleft, tright, node_type='EQUID', quad_type='GAUSS', useSpline=False
        )
