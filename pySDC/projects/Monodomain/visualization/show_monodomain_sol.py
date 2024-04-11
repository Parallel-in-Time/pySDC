import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os

# Script for displaying the solution of the monodomain equation

executed_file_dir = os.path.dirname(os.path.realpath(__file__))
output_root = executed_file_dir + "/../../../../data/Monodomain/results_tmp/"
domain_name = "cube_1D"
refinements = 2
ionic_model = "TTP"
file_name = "monodomain"
file_path = Path(output_root + domain_name + "/" + "ref_" + str(refinements) + "/" + ionic_model + "/" + file_name)


# no need to modifiy below this line
# ------------------------------------------------------------------------------
if not file_path.with_suffix(".npy").is_file():
    print(f"File {str(file_path)} does not exist")
    exit()

with open(str(file_path) + "_txyz.npy", "rb") as file:
    t = np.load(file, allow_pickle=True)
n_dt = t.size
print(f"t_end = {t[-1]}, n_dt = {n_dt}")

V = []
with open(file_path.with_suffix(".npy"), "rb") as file:
    for _ in range(n_dt):
        V.append(np.load(file, allow_pickle=True))

Vmin = np.min(V[0].flatten())
Vmax = np.max(V[0].flatten())
for Vi in V:
    Vmin = min(Vmin, np.min(Vi.flatten()))
    Vmax = max(Vmax, np.max(Vi.flatten()))

Vmin = 1.1 * max(Vmin, -100)
Vmax = 1.1 * min(Vmax, 200)
print(f"Vmin = {Vmin}, Vmax = {Vmax}")

dim = len(V[0].shape)

with open(str(file_path) + "_txyz.npy", "rb") as file:
    t = np.load(file, allow_pickle=True)
    xyz = []
    for _ in range(dim):
        xyz.append(np.load(file, allow_pickle=True))

if dim == 1:
    fig, ax = plt.subplots()
    ax.set(ylim=[Vmin, Vmax], xlabel="x [mm]", ylabel="V [mV]")
    line = ax.plot(xyz[0], V[0])[0]
elif dim == 2:
    fig, ax = plt.subplots()
    ax.set(xlabel="x [mm]", ylabel="y [mm]")
    ax.set_aspect(aspect="equal")
    line = ax.pcolormesh(xyz[0], xyz[1], V[0], cmap=plt.cm.jet, vmin=Vmin, vmax=Vmax)
    fig.colorbar(line)
elif dim == 3:
    Z, Y, X = np.meshgrid(xyz[2].ravel(), xyz[1].ravel(), xyz[0].ravel(), indexing="ij")
    kw = {"vmin": Vmin, "vmax": Vmax, "levels": np.linspace(Vmin, Vmax, 10)}
    # fig, ax = plt.subplots(projection="3d")
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    A = ax.contourf(V[0][:, :, 0], Y[:, :, 0], Z[:, :, 0], zdir="x", offset=0, **kw)
    B = ax.contourf(X[:, 0, :], V[0][:, 0, :], Z[:, 0, :], zdir="y", offset=0, **kw)
    C = ax.contourf(X[0, :, :], Y[0, :, :], V[0][0, :, :], zdir="z", offset=0, **kw)
    # D = ax.contourf(V[0][:, :, -1], Y[:, :, -1], Z[:, :, -1], zdir="x", offset=X.max(), **kw)

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    ax.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    fig.colorbar(A, ax=ax, fraction=0.02, pad=0.1, label="V [mV]")


def plot_V(k):
    ax.set_title(f"V at t = {t[k]:.3f} [ms]")
    if dim == 1:
        line.set_ydata(V[k])
        return line
    elif dim == 2:
        line.set_array(V[k].flatten())
        return line
    elif dim == 3:
        A = ax.contourf(V[k][:, :, 0], Y[:, :, 0], Z[:, :, 0], zdir="x", offset=0, **kw)
        B = ax.contourf(X[:, 0, :], V[k][:, 0, :], Z[:, 0, :], zdir="y", offset=0, **kw)
        C = ax.contourf(X[0, :, :], Y[0, :, :], V[k][0, :, :], zdir="z", offset=0, **kw)
        # D = ax.contourf(V[k][:, :, -1], Y[:, :, -1], Z[:, :, -1], zdir="x", offset=X.max(), **kw)
        return A, B, C


anim = animation.FuncAnimation(fig=fig, func=plot_V, interval=1, frames=n_dt, repeat=False)
plt.show()
# anim.save(file_path.with_suffix(".gif"))
