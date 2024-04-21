from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class SemiImplicit2(generic_implicit):
    r"""
    Custom sweeper class to implement SDC for solving semi-explicit DAEs for the form

    .. math::
        u' = f(u, z, t),

    .. math::
        0 = g(u, z, t)

    with :math:`u(t), u'(t) \in\mathbb{R}^{N_d}` the differential variables and their derivates, and algebraic variables :math:`z(t) \in\mathbb{R}^{N_a}`. The right-hand side functions are :math:`f(u, z, t) \in \mathbb{R}^{N_d}`,
    and :math:`g(u, z, t) \in \mathbb{R}^{N_a}`, where :math:`N = N_d + N_a` is the dimension of the whole
    system of DAEs.

    The :math:`\varepsilon`-embedding method [1]_ proposes the SDC scheme

    .. math::
        \mathbf{u}^{k+1} = \mathbf{u}_0 + \Delta t (\mathbf{Q}-\mathbf{Q}_\Delta)\otimes\mathbf{I}_{N_d}f(\mathbf{u}^k,\mathbf{z}^k, \boldsymbol{\tau}) + \Delta t \mathbf{Q}_\Delta\otimes\mathbf{I}_{N_d}f(\mathbf{u}^{k+1},\mathbf{z}^{k+1}, \boldsymbol{\tau})
    
    .. math::
        \mathbf{0} = g(\mathbf{u}^{k+1},\mathbf{z}^{k+1}, \boldsymbol{\tau})

    where
    
    - :math:`\tau=(\tau_1,..,\tau_M) in \mathbb{R}^M` the vector of collocation nodes,
    - :math:`\mathbf{u}_0 = (u_0,..,u_0) \in \mathbb{R}^{MN_d}` the vector of initial condition spread to each node,
    - spectral integration matrix :math:`\mathbf{Q} \in \mathbb{R}^{M \times M}`,
    - :math:`\mathbf{u}=(u_1,..,u_M) \in \mathbb{R}^{MN_d}` the vector of unknown differential variables
      :math:`u_m \approx u(\tau_m) \in \mathbb{R}^{N_d}`,
    - :math:`\mathbf{z}=(z_1,..,z_M) \in \mathbb{R}^{MN_a}` the vector of unknown algebraic variables
      :math:`z_m \approx z(\tau_m) \in \mathbb{R}^{N_a}`,
    - and identity matrix :math:`\mathbf{I}_{N_d} \in \mathbb{R}^{N_d \times N_d}`.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    QI : np.2darray
        Implicit Euler integration matrix.

    References
    ----------
    .. [1] E. Hairer, G. Wanner. Solving ordinary differential equations. II: Stiff and differential-algebraic problems. Springer Series in Computational Mathematics. 14. Berlin: Springer. xvi, 614 p. (1996).

    Note
    ----
    In order to get insights into numerical methods and their behavior for stiff ODEs and DAEs, the numerical method is first applied to the stiff ODE for :math:`0 < \varepsilon \ll 1`. The case :math:`\varepsilon = 0` then suggest a scheme to numerically solve the corresponding DAE as stiff limit. This is called the :math:`\varepsilon`-embedding method [1]_.

    This class implements the method resulting from the :math:`\varepsilon`-embedding method: Consider a system of ODEs of the form

    .. math::
        u' = f(u, z, t),

    .. math::
        \varepsilon z' = g(u, z, t)

    for :math:`0 < \varepsilon \ll 1`. Setting :math:`\varepsilon = 0` leads to a system of DAEs:
    
    .. math::
        u' = f(u, z, t),

    .. math::
        0 = g(u, z, t)
    """

    def __init__(self, params):
        """Initialization routine"""

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super().__init__(params)

        msg = f"Quadrature type {self.params.quad_type} is not implemented yet. Use 'RADAU-RIGHT' instead!"
        if self.coll.left_is_node:
            raise ParameterError(msg)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)