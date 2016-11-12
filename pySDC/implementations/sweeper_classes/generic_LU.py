from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class generic_LU(generic_implicit):
    """
    LU sweeper using LU decomposition of the Q matrix for the base integrator, special type of generic implicit sweeper

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        params['QI'] = 'LU'

        # call parent's initialization routine
        super(generic_LU, self).__init__(params)
