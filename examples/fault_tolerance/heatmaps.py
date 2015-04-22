import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

rc('text', usetex=True)
# rc("font", **{"sans-serif": ["Arial"], "size": 30})
rc('font', family='serif',size=30)
rc('legend', fontsize='small')
rc('xtick', labelsize='small')
rc('ytick', labelsize='small')


infile = np.load('results_hf_spread.npz')
data = infile['iter_count']
print(data)

infile = np.load('results_hf_interp_all.npz')
data = infile['iter_count']
print(data)

infile = np.load('results_hf_predict_3.npz')
data = infile['iter_count']
print(data)

infile = np.load('results_hf_predict_full.npz')
data = infile['iter_count']
print(data)


exit()

# ft_iter = infile['ft_iter']
# ft_step = infile['ft_step']

ft_iter = range(1,11)
ft_step = range(0,16)

xsize = len(ft_step)
ysize = len(ft_iter)

x = np.linspace(ft_step[0],ft_step[-1]+1,xsize)
# y = np.linspace(ft_iter[0],ft_iter[-1]+1,ysize)
y = np.linspace(0,11,ysize)
[X,Y] = np.meshgrid(x,y)


fig, ax = plt.subplots(figsize=(10,8))

plt.pcolor(X, Y, data, vmin=9, vmax=16)
plt.colorbar()
plt.tight_layout()

# ax.set_xticks(np.arange(xsize)+0.5, minor=False)
# ax.set_yticks(np.arange(ysize)+0.5, minor=False)
# ax.set_xticklabels(ft_step, minor=False)
# ax.set_yticklabels(ft_iter, minor=False)




plt.show()