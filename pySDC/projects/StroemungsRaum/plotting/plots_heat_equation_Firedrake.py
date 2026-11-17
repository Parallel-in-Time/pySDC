"""
Plots of :mod:`pySDC.projects.StroemungsRaum.run_heat_equation_Firedrake`, as those of the FEniCS version: the numerical
and exact solutions as surfaces with the cross-section along the diagonal, frame by frame, and contours of the final
step. Reads the ``CheckpointFile`` the run script writes.
"""

import json
import os

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from firedrake.pyplot import tricontour, tripcolor, trisurf

DEFAULT_PATH = f'{os.path.dirname(__file__)}/../data/heat_equation/'


def plot_solutions(path=DEFAULT_PATH, animate=True):
    """
    Plot the stored solutions; saves the last frame and the final contours as PNG files in `path`.

    Args:
        path (str): Folder the run script wrote to
        animate (bool): Show every step, not only the last one
    """
    with open(f'{path}/heat_equation_Firedrake_parameters.json') as f:
        parameters = json.load(f)
    nsteps = int(round(parameters['Tend'] / parameters['dt']))
    diagonal = np.linspace(0.001, 0.999, 51)

    with fd.CheckpointFile(f'{path}/heat_equation_Firedrake_Temperature.h5', 'r') as file:
        mesh = file.load_mesh()
        evaluate = fd.PointEvaluator(mesh, [[s, s] for s in diagonal]).evaluate
        fig = plt.figure(figsize=(8, 16))
        for step in range(nsteps) if animate else [nsteps - 1]:
            un = file.load_function(mesh, 'un', idx=step)
            ux = file.load_function(mesh, 'ux', idx=step)
            fig.clf()
            for position, u, title in [(221, un, 'Numerical Solution'), (222, ux, 'Analytical Solution')]:
                ax = fig.add_subplot(position, projection='3d')
                trisurf(u, axes=ax, cmap='jet')
                ax.set(xlabel='Distance x', ylabel='Distance y', title=title, zlim=(-1, 1))
            ax = fig.add_subplot(2, 2, (3, 4))
            ax.plot(np.sqrt(2) * diagonal, evaluate(un), 'b*-', label='Numerical solution')
            ax.plot(np.sqrt(2) * diagonal, evaluate(ux), 'r-', label='Analytical solution')
            ax.set(
                xlabel='Diagonal x=y', ylabel='Solution', title='Cross-section at the main diagonal', ylim=(-1.05, 1.05)
            )
            ax.legend()
            if animate:
                plt.pause(0.1)
        fig.savefig(f'{path}/heat_equation_Firedrake_Results.png', bbox_inches='tight')
        plt.close(fig)

    fig = plt.figure(figsize=(8, 11))
    for position, u, title, contour in [
        (221, un, 'Numerical Solution', False),
        (222, ux, 'Analytical Solution', False),
        (223, un, 'Numerical Solution', True),
        (224, ux, 'Analytical Solution', True),
    ]:
        ax = fig.add_subplot(position)
        if contour:
            tricontour(u, axes=ax, levels=30, cmap='jet')
            ax.axis('scaled')
        else:
            tripcolor(u, axes=ax, cmap='jet')
        ax.set(xlabel='Distance x', ylabel='Distance y', title=title)
    fig.savefig(f'{path}/heat_equation_Firedrake_Contours.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plot_solutions()
    plt.show()
