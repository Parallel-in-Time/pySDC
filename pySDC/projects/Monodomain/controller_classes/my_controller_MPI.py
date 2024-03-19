from pySDC.implementations.controller_classes.controller_MPI import controller_MPI


class my_controller_MPI(controller_MPI):
    def __init__(self, controller_params, description, comm):
        super().__init__(controller_params, description, comm)

    def recv(self, target, source, tag=None, comm=None):
        """
        Wrapper around recv which updates the lmbda and yinf status
        Indeed, lmbda and yinf are evaluated at u[0], so if we
        receive a new u[0] we need to update lmbda and yinf too.
        """
        super().recv(target, source, tag=tag, comm=comm)
        target.sweep.update_lmbda_yinf_status(outdated=True)
