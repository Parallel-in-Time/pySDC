import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import os

# rc('text', usetex=True)
rc("font", **{"sans-serif": ["Arial"], "size": 30})
# rc('font', family='serif',size=30)
rc('legend', fontsize='small')
rc('xtick', labelsize='small')
rc('ytick', labelsize='small')


list = [('HEAT','HEAT_steps_vs_iteration_hf_NOFAULT.npz','NOFAULT','k'),
        ('HEAT','HEAT_steps_vs_iteration_hf_SPREAD.npz','SPREAD','red'),
        ('HEAT','HEAT_steps_vs_iteration_hf_INTERP.npz','INTERP','orange'),
        ('HEAT','HEAT_steps_vs_iteration_hf_SPREAD_PREDICT.npz','SPREAD_PREDICT','blue'),
        ('HEAT','HEAT_steps_vs_iteration_hf_INTERP_PREDICT.npz','INTERP_PREDICT','green')]

maxres = -1
minres = -10
maxiter = 0
maxsteps = 0

for setup,file,strategy,color in list:

    infile = np.load(file)
    residual = infile['residual']
    maxiter = max(maxiter,len(residual[:,0]))
    maxsteps = max(maxsteps,len(residual[0,:]))


# for setup,file,strategy,color in list:
#
#     residual = np.zeros((maxiter,maxsteps))
#     residual[:] = -99
#
#     infile = np.load(file)
#     input = infile['residual']
#     step = infile['ft_step']
#     iter = infile['ft_iter']
#
#     residual[0:len(input[:,0]),0:len(input[0,:])] = input
#
#     fig, ax = plt.subplots(figsize=(15,10))
#
#     cmap = plt.get_cmap('Reds')
#     plt.pcolor(residual.T, cmap=cmap, vmin=minres, vmax=maxres)
#
#     plt.axis([0,maxiter,0,maxsteps])
#
#     cax = plt.colorbar()
#     cax.set_label('log10(residual)')
#
#     ax.set_xlabel('iteration')
#     ax.set_ylabel('step')
#
#     ax.set_xticks(np.arange(maxiter)+0.5, minor=False)
#     ax.set_yticks(np.arange(maxsteps)+0.5, minor=False)
#     ax.set_xticklabels(np.arange(maxiter)+1, minor=False)
#     ax.set_yticklabels(np.arange(maxsteps), minor=False)
#
#     plt.tight_layout()
#
#     if strategy is not 'NOFAULT':
#         plt.text(step-1+0.5,iter+0.5,'xxx',horizontalalignment='center',verticalalignment='center')
#
#     fname = setup+'_steps_vs_iteration_hf_'+str(step)+'x'+str(iter)+'_'+strategy+'.png'
#     plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')



fig, ax = plt.subplots(figsize=(15,10))
xvals = range(1,maxiter+1)

for setup,file,strategy,color in list:

    infile = np.load(file)
    residual = infile['residual']
    step = infile['ft_step']
    iter = infile['ft_iter']-1

    yvals = residual[:,step]
    print(strategy,yvals)

    plt.plot(xvals[0:iter],yvals[0:iter],'x-',color=color,label=strategy)
    plt.plot(xvals[iter:len(yvals)],yvals[iter:],'x-',color=color)
    plt.legend()

plt.xlim(1-0.5,maxiter+0.5)
plt.ylim(minres-0.5,maxres+0.5)

plt.xlabel('iteration')
plt.ylabel('residual')

plt.xticks(range(1,maxiter+1))
plt.yticks(range(minres,maxres+1))

plt.tight_layout()

plt.show()