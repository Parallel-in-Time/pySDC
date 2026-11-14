"""
Plots of the flow around the cylinder from :mod:`pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_Firedrake`,
as those of the FEniCS version: velocity field, pressure, vorticity, velocity magnitude and streamlines, frame by
frame. Reads the ``CheckpointFile`` that
:class:`~pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_Firedrake.LogWriteSolutions` writes.

The streamlines are drawn from the velocity directly, where the FEniCS version contours a stream function from a
Poisson solve.
"""

import json
import os

import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.pyplot import quiver, streamplot, tripcolor

DEFAULT_PATH = f'{os.path.dirname(__file__)}/../data/navier_stokes/'


def plot_solutions(path=DEFAULT_PATH, animate=True):
    """
    Plot the stored solutions; saves the last frame as ``Cylinder_Firedrake.png`` in `path`.

    Args:
        path (str): Folder the run script and the hook wrote to
        animate (bool): Show every step, not only the last one
    """
    with open(f'{path}/Navier_Stokes_Firedrake_parameters.json') as f:
        parameters = json.load(f)
    nsteps = int(round((parameters['Tend'] - parameters['t0']) / parameters['dt'])) + 1

    with fd.CheckpointFile(f'{path}/Cylinder_solutions.h5', 'r') as file:
        mesh = file.load_mesh()
        Vs = fd.FunctionSpace(mesh, 'CG', parameters['order'])
        fig = plt.figure(figsize=(16, 13), layout='constrained')
        for step in range(nsteps) if animate else [nsteps - 1]:
            un = file.load_function(mesh, 'un', idx=step)
            pn = file.load_function(mesh, 'pn', idx=step)
            vorticity = fd.Function(Vs).interpolate(un[1].dx(0) - un[0].dx(1))
            magnitude = fd.Function(Vs).interpolate(fd.sqrt(fd.dot(un, un)))
            fig.clf()
            panels = [
                ('Velocity field', lambda ax: quiver(un, axes=ax, cmap='jet')),
                ('pressure field', lambda ax: tripcolor(pn, axes=ax, cmap='jet')),
                ('Vorticity', lambda ax: tripcolor(vorticity, axes=ax, vmin=-30, vmax=30, cmap='jet')),
                ('Magnitude', lambda ax: tripcolor(magnitude, axes=ax, cmap='jet')),
                ('Streamlines', lambda ax: streamplot(un, axes=ax, resolution=1 / 50)),
            ]
            for i, (title, draw) in enumerate(panels):
                ax = fig.add_subplot(5, 1, i + 1)
                drawn = draw(ax)
                if hasattr(drawn, 'get_array'):
                    fig.colorbar(drawn, ax=ax)
                ax.set(xlabel='Distance x', ylabel='Distance y', title=title, xlim=(0, 2.2), ylim=(0, 0.41))
            if animate:
                plt.pause(0.01)
        fig.savefig(f'{path}/Cylinder_Firedrake.png', bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    plot_solutions()
