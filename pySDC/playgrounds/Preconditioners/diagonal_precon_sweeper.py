import numpy as np
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class DiagPrecon(generic_implicit):
    """
    Generic implicit sweeper with a diagonal preconditioner, which is build from entries in params

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
        QI = params.get('QI', None)
        if QI is None:
            params['QI'] = 'IE'
        super(DiagPrecon, self).__init__(params)

        # get QI matrix
        self.QI = np.diag(np.append([0], params['diagonal_elements']))
        self.QI[:, 0] = np.append([0], params['first_row'])


class DiagPreconIMEX(imex_1st_order):
    """
    Generic IMEX sweeper with a diagonal preconditioner, which is build from entries in params

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

        self.QI = np.diag(np.append([0], params['diagonal_elements']))
        self.QI[:, 0] = np.append([0], params['first_row'])
        self.QE = np.diag(np.zeros(len(params['diagonal_elements']) + 1))
