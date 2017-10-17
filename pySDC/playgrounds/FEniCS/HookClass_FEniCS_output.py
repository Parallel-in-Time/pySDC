from __future__ import division
from pySDC.core.Hooks import hooks
import dolfin as df

file = df.File('output1d/grayscott.pvd')  # dirty, but this has to be unique (and not per step or level)


class fenics_output(hooks):
    """
    Hook class to add output to FEniCS runs
    """

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts

        Args:
            step: the current step
            level_number: the current level number
        """
        super(fenics_output, self).pre_run(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        v = L.u[0].values
        v.rename('func', 'label')

        file << v

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(fenics_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # u1,u2 = df.split(L.uend.values)
        v = L.uend.values
        v.rename('func', 'label')

        file << v
