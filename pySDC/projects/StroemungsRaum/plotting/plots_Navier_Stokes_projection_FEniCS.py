import os
import dolfin as df
import json
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def plot_solutions():  # pragma: no cover

    # Open XDMF files for reading the stored solutions
    path = f"{os.path.dirname(__file__)}/../data/navier_stokes/"
    xdmffile_u = df.XDMFFile(path + 'Cylinder_velocity.xdmf')
    xdmffile_p = df.XDMFFile(path + 'Cylinder_pressure.xdmf')

    # load parameters
    parameters = json.load(open(path + "Navier_Stokes_FEniCS_parameters.json", 'r'))

    # get parameters
    dt = parameters['dt']
    Tend = parameters['Tend']
    family = parameters['family']
    order = parameters['order']
    t0 = parameters['t0']
    nsteps = int(Tend / dt)

    # load mesh
    meshpath = f"{os.path.dirname(__file__)}/../meshs/"
    mesh = df.Mesh(meshpath + 'cylinder.xml')

    # define function spaces for velocity and physical quantities
    V = df.VectorFunctionSpace(mesh, family, order)
    Q = df.FunctionSpace(mesh, family, order - 1)
    Vs = df.FunctionSpace(mesh, family, order)

    # define functions for velocity and pressure
    un = df.Function(V)
    pn = df.Function(Q)

    # stiffness  matrix for streamlines computation
    u = df.TrialFunction(Vs)
    v = df.TestFunction(Vs)
    a = df.dot(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    S = df.assemble(a)
    Str = df.Function(Vs)

    # open figure for plots
    fig = plt.figure(1, figsize=(16, 13))

    # initialize time variable
    t = t0

    for s in range(nsteps):

        # Update current time
        t += dt

        # read the velocity and pressure solutions at the current time step
        xdmffile_u.read_checkpoint(un, 'un', s)
        xdmffile_p.read_checkpoint(pn, 'pn', s)

        # compute the vorticity field
        ux, uy = un.split(deepcopy=True)
        Vort = uy.dx(0) - ux.dx(1)

        # compute the streamlines
        l = Vort * v * df.dx
        L = df.assemble(l)
        df.solve(S, Str.vector(), L)

        # compute the magnitude of the velocity field
        u_magn = df.sqrt(df.dot(un, un))
        u_magn = df.project(u_magn, Vs)

        # subplot for velocity
        ax = fig.add_subplot(511)
        c = df.plot(un, cmap='jet')
        plt.colorbar(c)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Velocity field')
        ax.set_xlim(-0.01, 2.22)
        ax.set_ylim(-0.005, 0.415)
        plt.draw()

        # subplot for pressure
        ax = fig.add_subplot(512)
        c = df.plot(pn, mode='color', cmap='jet')
        plt.colorbar(c)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('pressure field')
        ax.set_xlim(0, 2.20)
        ax.set_ylim(0, 0.41)
        plt.draw()

        # subplot for vorticity
        ax = fig.add_subplot(513)
        c = df.plot(Vort, mode='color', vmin=-30, vmax=30, cmap='jet')
        plt.colorbar(c)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Vorticity')
        ax.set_xlim(0, 2.20)
        ax.set_ylim(0, 0.41)
        plt.draw()

        # subplot for velocity magnitude
        ax = fig.add_subplot(514)
        c = df.plot(u_magn, mode='color', cmap='jet')
        plt.colorbar(c)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Magnitude')
        ax.set_xlim(0, 2.20)
        ax.set_ylim(0, 0.41)
        plt.draw()

        ax = fig.add_subplot(515)
        c = df.plot(Str, mode='contour', levels=50)
        plt.colorbar(c)
        ax.set_xlabel('Distance x')
        ax.set_ylabel('Distance y')
        ax.set_title('Magnitude = Streamlines')
        ax.set_xlim(0, 2.20)
        ax.set_ylim(0, 0.41)
        plt.draw()

        plt.pause(0.01)
        plt.clf()

    plt.close(fig)


if __name__ == '__main__':
    plot_solutions()
