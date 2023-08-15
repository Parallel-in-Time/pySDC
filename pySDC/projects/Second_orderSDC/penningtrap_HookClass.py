from pySDC.core.Hooks import hooks
import numpy as np


class particles_output(hooks):
    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(particles_output, self).pre_run(step, level_number)

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(particles_output, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # self.bar_run.update(L.time)

        L.sweep.compute_end_point()
        part = L.uend
        N = L.prob.nparts

        # =============================================================================
        #

        try:
            L.prob.Harmonic_oscillator

            # add up kinetic and potntial contributions to total energy
            epot = 0
            ekin = 0
            name = str(L.sweep)
            for n in range(N):
                epot += 1 / 2.0 * np.dot(part.pos[:, n], part.pos[:, n])
                ekin += 1 / 2.0 * np.dot(part.vel[:, n], part.vel[:, n])
                Energy = epot + ekin
            uinit = L.u[0]
            H0 = 1 / 2 * (np.dot(uinit.vel[:].T, uinit.vel[:]) + np.dot(uinit.pos[:].T, uinit.pos[:]))
            Ham = abs(Energy - H0) / abs(H0)
            if 'RKN' in name:
                filename = 'data/Ham_RKN' + '{}.txt'.format(step.params.maxiter)
            else:
                filename = 'data/Ham_SDC' + '{}{}.txt'.format(L.sweep.coll.num_nodes, step.params.maxiter)

            if L.time == 0.0:
                file = open(filename, 'w')
                file.write(str('time') + " | " + str('Hamiltonian error') + '\n')
            else:
                file = open(filename, 'a')

            file.write(str(L.time) + " | " + str(Ham) + '\n')

            file.close()
        except AttributeError:
            pass
        # =============================================================================

        part_exact = L.prob.u_exact(L.time + L.dt)
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='position',
            value=part.pos,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='velocity',
            value=part.vel,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='position_exact',
            value=part_exact.pos,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='velocity_exact',
            value=part_exact.vel,
        )
