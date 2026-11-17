"""
Hooks for the flow around the cylinder: drag and lift of every step, and the solutions of every step, which go to a
Firedrake ``CheckpointFile`` (HDF5, with the mesh).
"""

import os
from pathlib import Path

import firedrake as fd

from pySDC.core.hooks import Hooks

#: where the solutions go unless the hook is given another folder
DEFAULT_PATH = f'{os.path.dirname(__file__)}/../data/navier_stokes/'


class LogLiftDrag(Hooks):
    """
    Record the drag and lift coefficients of the cylinder at the end of every step, as ``drag_post_step`` and
    ``lift_post_step``, with the time the step ends at (the time of ``uend``). Works with every problem class that has
    ``drag_lift(uend)``, i.e. the monolithic and the projection version of the cylinder flow.
    """

    def post_step(self, step, level_number):
        """
        Record drag and lift at the end of the step.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The current level number
        """
        super().post_step(step, level_number)
        L = step.levels[level_number]
        L.sweep.compute_end_point()
        drag, lift = L.prob.drag_lift(L.uend)
        for kind, value in [('drag', drag), ('lift', lift)]:
            self.add_to_stats(
                process=step.status.slot,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=f'{kind}_post_step',
                value=value,
            )


class LogWriteSolutions(Hooks):
    """
    Write velocity and pressure at the end of every step to ``Cylinder_solutions.h5``, as ``un`` and ``pn`` with the
    step number as index. The pressure is the one the projection scheme keeps in ``pn``. Read them back with
    ``CheckpointFile(..., 'r')``, ``load_mesh()`` and ``load_function(mesh, 'un', idx=step)``.

    Set the class attribute `path` to write somewhere else than the project's ``data/navier_stokes`` folder.
    """

    path = DEFAULT_PATH

    def pre_run(self, step, level_number):
        """
        Open the file and store the mesh in it.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The current level number
        """
        super().pre_run(step, level_number)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        P = step.levels[level_number].prob
        self.file = fd.CheckpointFile(f'{self.path}/Cylinder_solutions.h5', 'w', comm=P.mesh.comm)
        self.file.save_mesh(P.mesh)
        self.steps_written = 0

    def post_step(self, step, level_number):
        """
        Write the solutions at the end of the step.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The current level number
        """
        super().post_step(step, level_number)
        L = step.levels[level_number]
        L.sweep.compute_end_point()
        self.file.save_function(L.uend.functionspace, idx=self.steps_written, name='un')
        self.file.save_function(L.prob.pn, idx=self.steps_written, name='pn')
        self.steps_written += 1

    def post_run(self, step, level_number):
        """
        Close the file.

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The current level number
        """
        super().post_run(step, level_number)
        self.file.close()
