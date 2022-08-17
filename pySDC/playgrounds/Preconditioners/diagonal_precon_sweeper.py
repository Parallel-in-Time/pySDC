import numpy as np
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class DiagPrecon(generic_implicit):
    """
    Generic implicit sweeper with a diagonal preconditioner, which is build from entrie in params

    Attributes:
        diags: diagonal elements of the preconditiner
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        # call parent's initialization routine
        super(DiagPrecon, self).__init__(params)

        # get QI matrix
        self.QI = np.diag(params['diagonal_elements'])


class DiagPreconIMEX(imex_1st_order):
    """
    Generic IMEX sweeper with a diagonal preconditioner, which is build from entrie in params

    Attributes:
        diags: diagonal elements of the preconditiner
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        # call parent's initialization routine
        super(DiagPreconIMEX, self).__init__(params)

        self.QI = np.diag(params['diagonal_elements'])
        self.QE = np.diag(np.zeros(len(params['diagonal_elements'])))
