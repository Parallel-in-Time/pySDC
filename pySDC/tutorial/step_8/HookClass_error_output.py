from pySDC.core.Hooks import hooks
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger


class error_output(hooks):
    """
    Hook class to add output of error
    """

    def __init__(self):
        super(error_output, self).__init__()
        self.uex = None

    def pre_step(self, step, level_number):
        """
        Default routine called before each step
        Args:
            step: the current step
            level_number: the current level number
        """
        super(error_output, self).pre_step(step, level_number)

        L = step.levels[level_number]

        # This is a bit black magic: we are going to run pySDC within the hook to check the error against the "exact"
        # solution of the collocation problem
        description = step.params.description
        description['level_params']['restol'] = 1e-14
        if type(L.prob) != auzinger:
            description['problem_params']['solver_type'] = 'direct'

        controller_params = step.params.controller_params
        del controller_params['hook_class']  # get rid of the hook, otherwise this will be an endless recursion..
        controller_params['logger_level'] = 90
        controller_params['convergence_controllers'] = {}

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

        # compute and save errors
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
