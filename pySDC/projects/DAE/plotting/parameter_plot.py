import matplotlib.pyplot as plt
import numpy as np


data = np.load(r"/Users/heisenberg/Workspace/pySDC/data/dae_data.npy")

fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot(data[:, 1], data[:, 2])
ax.plot(data[:, 0], data[:, 1], label="x")
ax.plot(data[:, 0], data[:, 2], label="y")
# title='Convergence plot two stage implicit Runge-Kutta with Gauss nodes'
ax.set(xlabel='t', ylabel='size')
ax.grid(visible=True)
# fig.tight_layout()
plt.legend()
plt.show()
# plt.savefig('../results/problematic_good.png')