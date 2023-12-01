import numpy as np
import os
from pathlib import Path
from pySDC.core.Hooks import hooks


class pde_hook(hooks):
    def __init__(self):
        super(pde_hook, self).__init__()

    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(pde_hook, self).pre_run(step, level_number)

        L = step.levels[level_number]
        P = L.prob
        if level_number == 0 and L.time == P.t0:
            P.write_solution(L.u[0], P.t0)
            self.t_val = np.array([P.t0])
            self.u_val = P.eval_on_points(L.u[0])

    def post_step(self, step, level_number):
        super(pde_hook, self).post_step(step, level_number)

        L = step.levels[level_number]
        if level_number == 0:
            P = L.prob
            P.write_solution(L.uend, L.time + L.dt)
            self.save_sol_at_points(L.uend, L.time + L.dt, P)

    def post_run(self, step, level_number):
        L = step.levels[level_number]
        if level_number == 0:
            P = L.prob
            self.write_sol_at_points(P)

    def save_sol_at_points(self, u, t, P):
        self.t_val = np.append(self.t_val, t)
        unew = P.eval_on_points(u)
        if P.comm.rank == 0 and unew is not None:
            self.u_val = np.append(self.u_val, unew, axis=1)

    def write_sol_at_points(self, P):
        if P.comm.rank == 0 and self.u_val is not None:
            os.makedirs(P.output_folder, exist_ok=True)
            np.save(P.output_folder / Path(P.output_file_name + "_t"), self.t_val)
            np.save(P.output_folder / Path(P.output_file_name + "_u"), self.u_val)
            np.save(P.output_folder / Path(P.output_file_name + "_p"), P.parabolic.eval_points)
