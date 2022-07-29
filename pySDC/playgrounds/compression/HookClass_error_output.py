from pySDC.core.Hooks import hooks
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class error_output(hooks):
    """
    Hook class to add output of error
    """

    def __init__(self):
        super(error_output, self).__init__()
        self.uex = None

    # def post_iteration(self, step, level_number):
    #     """
    #     Default routine called after each iteration
    #     Args:
    #         step: the current step
    #         level_number: the current level number
    #     """
    #
    #     super(error_output, self).post_iteration(step, level_number)
    #
    #     # some abbreviations
    #     L = step.levels[level_number]
    #     P = L.prob
    #
    #     L.sweep.compute_end_point()
    #
    #     uex = P.u_exact(step.time + step.dt)
    #     err = abs(uex - L.uend)
    #
    #     self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
    #                       sweep=L.status.sweep, type='error_after_iter', value=err)

    def pre_step(self, step, level_number):
        """
        Default routine called before each step
        Args:
            step: the current step
            level_number: the current level number
        """
        super(error_output, self).pre_step(step, level_number)

        L = step.levels[level_number]

        description = step.params.description
        description['level_params']['restol'] = 1e-14
        description['problem_params']['direct_solver'] = True

        controller_params = step.params.controller_params
        del controller_params['hook_class']
        controller_params['use_iteration_estimator'] = False
        controller_params['logger_level'] = 90

        controller = controller_nonMPI(num_procs=1, description=description, controller_params=controller_params)
        self.uex, _ = controller.run(u0=L.u[0], t0=L.time, Tend=L.time + L.dt)

    def post_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        pde_err = abs(upde - L.uend)
        coll_err = abs(self.uex - L.uend)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='PDE_error_after_step',
            value=pde_err,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='coll_error_after_step',
            value=coll_err,
        )
