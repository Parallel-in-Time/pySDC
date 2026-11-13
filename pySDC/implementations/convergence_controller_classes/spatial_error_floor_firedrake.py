import firedrake as fd

from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.datatype_classes.firedrake_mesh import riesz_map

_transfer_manager = fd.TransferManager()


def coarse_space_error(u, coarse_space):
    r"""
    The part of `u` that a coarser space cannot represent, :math:`\|u - P R u\|_{L^2}`.

    Here :math:`R` is the :math:`L^2` projection onto `coarse_space` and :math:`P` the prolongation back. For a smooth
    solution, this is about the error of the coarse space, so dividing it by :math:`2^{p+1}` estimates the
    :math:`L^2` error of a degree :math:`p` space on a mesh refined once.

    Args:
        u (firedrake.Function): Function on a mesh of a Firedrake ``MeshHierarchy``
        coarse_space (firedrake.FunctionSpace): Space on the next coarser mesh of the same hierarchy

    Returns:
        float: :math:`\|u - P R u\|_{L^2}`
    """
    coarse_load = fd.Cofunction(coarse_space.dual())
    _transfer_manager.restrict(riesz_map(u.function_space())(u), coarse_load)
    back = fd.Function(u.function_space())
    _transfer_manager.prolong(riesz_map(coarse_space)(coarse_load), back)
    return fd.norm(u - back)


class SpatialErrorFloorFiredrake(ConvergenceController):
    r"""
    Stop iterating once the residual is small compared with the spatial discretization error, for Firedrake problems.

    Iterating further cannot make the solution more accurate than its spatial error allows. This controller estimates
    that error at the end of the step, :math:`e_x = \|u - P R u\| / 2^{p+1}` (see :func:`coarse_space_error`), and
    declares the step converged once (on the last collocation node, the end point for RADAU-RIGHT)

    .. math::
        \|r\| \le \gamma \, e_x \, \Delta t / T,

    in addition to the usual criteria such as ``restol``, which remains the floor where the spatial error is small.
    The factor :math:`\Delta t / T` turns the global spatial error into a budget per step, since what each step leaves
    of the iteration error adds up over the :math:`T / \Delta t` steps.

    The residual :math:`r` has to be measured in the :math:`L^2` norm for this comparison: for the mass-matrix
    sweepers it is a load vector, whose norm (:class:`~pySDC.implementations.datatype_classes.firedrake_mesh.firedrake_dual_mesh`)
    is that of the function it represents. For diffusive problems, it bounds the iteration error up to a modest
    constant, which is why :math:`\gamma = 1` is a safe default.

    The estimate assumes a smooth solution, so that the error of a degree :math:`p` space falls by :math:`2^{p+1}` when
    the mesh is refined once. It needs a coarse space on the next coarser mesh of a ``MeshHierarchy``.

    Parameters:
        coarse_space (firedrake.FunctionSpace): Space of the same kind on the next coarser mesh of the hierarchy
        Tend (float): End time of the run
        gamma (float, optional): Safety factor, 1 by default
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define the defaults of the parameters.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: The updated params dictionary
        """
        return {'control_order': 100, 'gamma': 1.0, **super().setup(controller, params, description, **kwargs)}

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Make sure the coarse space and the end time are given.

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        for key in ['coarse_space', 'Tend']:
            if key not in params:
                return False, f'{type(self).__name__} needs the parameter `{key}`'
        return True, ''

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Compare the residual on the finest level with the spatial error budget of the step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        L = S.levels[0]
        u = L.u[-1].functionspace
        degree = u.function_space().ufl_element().embedded_superdegree
        e_x = coarse_space_error(u, self.params.coarse_space) / 2 ** (degree + 1)
        if L.status.residual <= self.params.gamma * e_x * L.dt / self.params.Tend:
            self.debug(f'Residual {L.status.residual:.2e} below the spatial error budget of {e_x:.2e}', S)
            S.status.force_done = True
