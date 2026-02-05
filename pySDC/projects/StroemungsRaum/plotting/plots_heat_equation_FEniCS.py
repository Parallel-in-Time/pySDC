import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json


def plot_solutions():

    # Get the data directory
    path = "data/"

    # Open XDMF file for visualization output
    xdmffile_u = df.XDMFFile(path + 'heat_equation_FEniCS_Temperature.xdmf')

    # load parameters
    parameters = json.load(open(path + "heat_equation_FEniCS_parameters.json", 'r'))

    # Get the simulation parameters
    dt = parameters['dt']
    Tend = parameters['Tend']
    c_nvars = parameters['c_nvars']
    family = parameters['family']
    order = parameters['order']

    # Compute the number of time steps
    nsteps = int(Tend / dt)

    # set mesh
    mesh = df.UnitSquareMesh(c_nvars[0], c_nvars[0])

    # define function space for future reference
    V = df.FunctionSpace(mesh, family, order[0])

    # Define exact and numerical solutions
    un = df.Function(V)
    ux = df.Function(V)

    # Compute cross sections of the results
    tol = 0.001  # Avoid hitting the outside of the domain
    S = np.linspace(0 + tol, 1 - tol, 51)

    # Open figure for plots
    fig = plt.figure(figsize=(8, 16))

    # Time-stepping
    t = 0
    for s in range(nsteps):
        # Update current time
        t += dt

        # Read the velocity field u from the XDMF file
        xdmffile_u.read_checkpoint(un, 'un', s)
        xdmffile_u.read_checkpoint(ux, 'ux', s)

        csn = []
        csx = []

        for i in range(len(S)):
            csn.append(un(S[i], S[i]))
            csx.append(ux(S[i], S[i]))

        # plt.title('Time t='+str(time))
        # plt.axis('off')

        ax = fig.add_subplot(221, projection='3d')
        df.plot(un, cmap=cm.jet)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Numerical Solution')
        ax.set_zlim(-1, 1)
        plt.draw()
        # plt.show()

        ax = fig.add_subplot(222, projection='3d')
        df.plot(ux, cmap=cm.jet)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Analytical Solution')
        ax.set_zlim(-1, 1)
        plt.draw()

        # plot diagonal cross-section of the numerical and the analytical solutions
        plt.subplot(2, 2, (3, 4))
        ax = plt.gca()
        plt.plot(np.sqrt(2) * S, csn, 'b*-', label='Numerical solution')
        plt.plot(np.sqrt(2) * S, csx, 'r-', label='Analytical solution')
        ax.set_xlabel('Diagonal x=y')
        ax.set_ylabel('Solution')
        ax.set_title('Cross-section at the main diagonal')
        # plt.xlim([0,1])
        plt.ylim([-1.05, 1.05])
        plt.legend()

        plt.pause(1)
        if t == Tend:
            plt.savefig(path + '/heat_equation_FEniCS_Results', bbox_inches='tight')
            # plt.show()
        plt.clf()
    # plot contourlines plots
    fig = plt.figure(figsize=(8, 11))

    ax = fig.add_subplot(221)
    df.plot(un, cmap=cm.jet)
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('SDC-FEniCS Solution')
    plt.draw()

    ax = fig.add_subplot(222)
    df.plot(ux, cmap=cm.jet)
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Analytical Solution')
    plt.draw()

    ax = fig.add_subplot(223)
    df.plot(un, mode="contour", levels=30, cmap=plt.cm.jet)
    plt.axis('scaled')
    plt.xlabel('Distance x')
    plt.ylabel('Distance y')
    plt.title('SDC-FEniCS Solution')
    plt.draw()

    ax = fig.add_subplot(224)
    df.plot(ux, mode="contour", levels=30, cmap=plt.cm.jet)
    plt.axis('scaled')
    plt.xlabel('Distance x')
    plt.ylabel('Distance y')
    plt.title('Analytical Solution')
    plt.draw()

    plt.savefig(path + '/heat_equation_FEniCS_Contours', bbox_inches='tight')


if __name__ == '__main__':
    plot_solutions()
    plt.show()
