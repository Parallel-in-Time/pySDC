from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class my_controller_nonMPI(controller_nonMPI):
    def __init__(self, num_procs, controller_params, description):
        super().__init__(num_procs, controller_params, description)

    def recv_full(self, S, level=None, add_to_stats=False):
        """
        Wrapper around recv_full which updates the lmbda and yinf status
        whenever u[0] is updated
        Indeed, if u[0] is updated, lmbda and yinf become outdated
        since they are evaluated on u[0]
        """

        super().recv_full(S, level, add_to_stats)

        # when the if is true then the recv method is called in super().recv_full and
        # target u[0] is updated, therefore lmbda and yinf become outdated
        if not S.status.prev_done and not S.status.first:
            S.levels[level].sweep.update_lmbda_yinf_status(outdated=True)
