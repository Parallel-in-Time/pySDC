from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class my_controller_nonMPI(controller_nonMPI):
    def __init__(self, num_procs, controller_params, description):
        super().__init__(num_procs, controller_params, description)

    def recv_full(self, S, level=None, add_to_stats=False):
        """
        Wrapper around recv which updates the lmbda and yinf status
        Indeed, lmbda and yinf are evaluated at u[0], so if we
        receive a new u[0] we need to update lmbda and yinf too.
        """

        super().recv_full(S, level, add_to_stats)

        # when the if is true then the recv method is called in super().recv_full and
        # S.levels[level].u[0] is updated, therefore lmbda and yinf become outdated at this level
        if not S.status.prev_done and not S.status.first:
            S.levels[level].sweep.update_lmbda_yinf_status(outdated=True)
