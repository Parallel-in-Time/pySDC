
class ptype():
    """
    Prototype class for problems, just defines the attributes essential to get started

    Attributes:
        init: number of degrees-of-freedom (whatever this may represent)
        dtype_u: variable data type
        dtype_f: RHS data type
    """

    def __init__(self, init, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            init: number of degrees-of-freedom (whatever this may represent)
            dtype_u: variable data type
            dtype_f: RHS data type
        """
        
        self.init = init
        self.dtype_u = dtype_u
        self.dtype_f = dtype_f
