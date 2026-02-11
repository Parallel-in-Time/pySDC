import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json


def plot_solutions():  # pragma: no cover

    # get the data directory
    import os

    path = f"{os.path.dirname(__file__)}/../data/heat_equation/"

    # Open XDMF file for visualization output
    xdmffile_u = df.XDMFFile(path + 'heat_equation_FEniCS_Temperature.xdmf')

    # load parameters
    with open(path + "heat_equation_FEniCS_parameters.json", 'r') as f:
        parameters = json.load(f)

    # Get the simulation parameters
    dt = parameters['dt']
    Tend = parameters['Tend']
    c_nvars = parameters['c_nvars']
    family = parameters['family']
    order = parameters['order']

    # Compute the number of time steps (robust to floating-point rounding)
    nsteps = int(round(Tend / dt))

    # set mesh
    mesh = df.UnitSquareMesh(c_nvars, c_nvars)

    # define function space for future reference
    V = df.FunctionSpace(mesh, family, order)

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

        ax = fig.add_subplot(221, projection='3d')
        df.plot(un, cmap=cm.jet)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Numerical Solution')
        ax.set_zlim(-1, 1)
        plt.draw()

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

        plt.pause(0.1)
        if s == nsteps - 1:
            plt.savefig(path + '/heat_equation_FEniCS_Results.png', bbox_inches='tight')
        plt.clf()

    fig = plt.figure(figsize=(8, 11))

    ax = fig.add_subplot(221)
    df.plot(un, cmap=cm.jet)
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Numerical Solution')
    plt.draw()

    ax = fig.add_subplot(222)
    df.plot(ux, cmap=cm.jet)
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Analytical Solution')
    plt.draw()

    ax = fig.add_subplot(223)
    df.plot(un, mode="contour", levels=30, cmap=plt.cm.jet)
    ax.axis('scaled')
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Numerical Solution')
    plt.draw()

    ax = fig.add_subplot(224)
    df.plot(ux, mode="contour", levels=30, cmap=plt.cm.jet)
    ax.axis('scaled')
    ax.set_xlabel('Distance x')
    ax.set_ylabel('Distance y')
    ax.set_title('Analytical Solution')
    plt.draw()

    plt.savefig(path + '/heat_equation_FEniCS_Contours.png', bbox_inches='tight')
    xdmffile_u.close()


if __name__ == '__main__':
    plot_solutions()
    plt.show()
