import numpy as np

from pySDC.core.Hooks import hooks


class particle_hook(hooks):
    def __init__(self):
        """
        Initialization of particles output
        """
        super(particle_hook, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(particle_hook, self).pre_run(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        part = L.u[0]
        N = L.prob.nparts
        w = np.array([1, 1, -2])

        # compute (slowly..) the potential at u0
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0, i):
                dist2 = np.linalg.norm(part.pos[:, i] - part.pos[:, j], 2) ** 2 + L.prob.sig**2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            for j in range(i + 1, N):
                dist2 = np.linalg.norm(part.pos[:, i] - part.pos[:, j], 2) ** 2 + L.prob.sig**2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            fpot[i] -= (
                L.prob.omega_E**2 * part.m[i] / part.q[i] / 2.0 * np.dot(w, part.pos[:, i] * part.pos[:, i])
            )

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n] * fpot[n]
            ekin += part.m[n] / 2.0 * np.dot(part.vel[:, n], part.vel[:, n])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='etot',
            value=epot + ekin,
        )

    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """

        super(particle_hook, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        L.sweep.compute_end_point()
        part = L.uend
        N = L.prob.nparts
        w = np.array([1, 1, -2])

        # compute (slowly..) the potential at uend
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0, i):
                dist2 = np.linalg.norm(part.pos[:, i] - part.pos[:, j], 2) ** 2 + L.prob.sig**2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            for j in range(i + 1, N):
                dist2 = np.linalg.norm(part.pos[:, i] - part.pos[:, j], 2) ** 2 + L.prob.sig**2
                fpot[i] += part.q[j] / np.sqrt(dist2)
            fpot[i] -= (
                L.prob.omega_E**2 * part.m[i] / part.q[i] / 2.0 * np.dot(w, part.pos[:, i] * part.pos[:, i])
            )

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n] * fpot[n]
            ekin += part.m[n] / 2.0 * np.dot(part.vel[:, n], part.vel[:, n])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='etot',
            value=epot + ekin,
        )

        return None
