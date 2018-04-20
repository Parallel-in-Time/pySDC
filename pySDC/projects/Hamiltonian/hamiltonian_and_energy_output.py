from __future__ import division
from pySDC.core.Hooks import hooks
import numpy as np


class hamiltonian_and_energy_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(hamiltonian_and_energy_output, self).__init__()
        self.ham_init = None
        self.energy_init = None

    def pre_run(self, step, level_number):
        # some abbreviations
        L = step.levels[0]
        P = L.prob
        super(hamiltonian_and_energy_output, self).pre_run(step, level_number)
        self.ham_init = P.eval_hamiltonian(L.u[0])
        self.energy_init = P.eval_mode_energy(L.u[0])

    def post_iteration(self, step, level_number):
        """
        Overwrite standard post iteration hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(hamiltonian_and_energy_output, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[0]
        P = L.prob

        L.sweep.compute_end_point()
        H = P.eval_hamiltonian(L.uend)
        E = P.eval_mode_energy(L.uend)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='hamiltonian', value=H)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='err_hamiltonian', value=abs(self.ham_init - H))

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='energy_iter', value=E)

        return None

    def post_step(self, step, level_number):
        """
        Overwrite standard post iteration hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(hamiltonian_and_energy_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]
        P = L.prob

        E = P.eval_mode_energy(L.uend)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='position', value=L.uend.pos.values)

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='energy_step', value=E)

        return None
