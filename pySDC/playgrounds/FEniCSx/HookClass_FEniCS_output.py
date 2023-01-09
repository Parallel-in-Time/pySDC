# import dolfin as df

from pySDC.core.Hooks import hooks

# file = df.File('output1d/grayscott.pvd')  # dirty, but this has to be unique (and not per step or level)


class fenics_output(hooks):
    """
    Hook class to add output to FEniCS runs
    """

    # def pre_run(self, step, level_number):
    #     """
    #     Overwrite default routine called before time-loop starts
    #
    #     Args:
    #         step: the current step
    #         level_number: the current level number
    #     """
    #     super(fenics_output, self).pre_run(step, level_number)
    #
    #     # some abbreviations
    #     L = step.levels[level_number]
    #
    #     v = L.u[0].values
    #     v.rename('func', 'label')
    #
    #     file << v

    def post_iteration(self, step, level_number):

        super(fenics_output, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        uex = L.prob.u_exact(L.time + L.dt)

        err = abs(uex - L.u[-1]) / abs(uex)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error',
            value=err,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='residual',
            value=L.status.residual / abs(L.u[0]),
        )

    # def post_step(self, step, level_number):
    #     """
    #     Default routine called after each iteration
    #     Args:
    #         step: the current step
    #         level_number: the current level number
    #     """
    #
    #     super(fenics_output, self).post_step(step, level_number)
    #
    #     # some abbreviations
    #     L = step.levels[level_number]
    #
    #     # u1,u2 = df.split(L.uend.values)
    #     v = L.uend.values
    #     v.rename('func', 'label')
    #
    #     file << v
