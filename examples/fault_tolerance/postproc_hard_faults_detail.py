import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt


# rc('text', usetex=True)
rc("font", **{"sans-serif": ["Arial"], "size": 30})
rc('font', family='serif',size=30)
rc('legend', fontsize='small')
rc('xtick', labelsize='small')
rc('ytick', labelsize='small')


# setup = 'HEAT'
setup = 'ADVECTION'

list = [(setup+'_steps_vs_iteration_hf_NOFAULT.npz','NOFAULT','no fault','k','^'),
        (setup+'_steps_vs_iteration_hf_SPREAD.npz','SPREAD','1-sided','red','v'),
        (setup+'_steps_vs_iteration_hf_INTERP.npz','INTERP','2-sided','orange','o'),
        (setup+'_steps_vs_iteration_hf_SPREAD_PREDICT.npz','SPREAD_PREDICT','1-sided + corr','blue','s'),
        (setup+'_steps_vs_iteration_hf_INTERP_PREDICT.npz','INTERP_PREDICT','2-sided + corr','green','d')]

maxres = -1
minres = -11
maxiter = 0
maxsteps = 0

for file,strategy,label,color,marker in list:

    infile = np.load(file)
    residual = infile['residual']
    maxiter = max(maxiter,len(residual[:,0]))
    maxsteps = max(maxsteps,len(residual[0,:]))


for file,strategy,label,color,marker in list:

    residual = np.zeros((maxiter,maxsteps))
    residual[:] = -99

    infile = np.load(file)
    input = infile['residual']
    step = infile['ft_step']
    iter = infile['ft_iter']

    residual[0:len(input[:,0]),0:len(input[0,:])] = input

    fig, ax = plt.subplots(figsize=(15,10))

    cmap = plt.get_cmap('Reds')
    plt.pcolor(residual.T, cmap=cmap, vmin=minres, vmax=maxres)

    plt.axis([0,maxiter,0,maxsteps])

    cax = plt.colorbar()
    cax.set_label('log10(residual)')

    ax.set_xlabel('iteration')
    ax.set_ylabel('step')

    ax.set_xticks(np.arange(maxiter)+0.5, minor=False)
    ax.set_yticks(np.arange(maxsteps)+0.5, minor=False)
    ax.set_xticklabels(np.arange(maxiter)+1, minor=False)
    ax.set_yticklabels(np.arange(maxsteps), minor=False)

    ax.tick_params(pad=8)
    plt.tight_layout()

    if strategy is not 'NOFAULT':
        plt.text(step-1+0.5,iter+0.5,'xxx',horizontalalignment='center',verticalalignment='center')

    fname = setup+'_steps_vs_iteration_hf_'+str(step)+'x'+str(iter)+'_'+strategy+'.png'
    plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')



fig, ax = plt.subplots(figsize=(15,10))
maxiter = 0
lw = 2
ms = 10


for file,strategy,label,color,marker in list:

    infile = np.load(file)
    residual = infile['residual']
    step = infile['ft_step']
    iter = infile['ft_iter']-1
    yvals = residual[residual[:,step]>-99,step]

    maxiter = max(maxiter,len(yvals))
    xvals = range(1,len(yvals)+1)
    # print(strategy,yvals)

    plt.plot(xvals[0:iter],yvals[0:iter],color=color,linewidth=lw,linestyle='-',markersize=ms,marker=marker,
                                         markeredgecolor='k',markerfacecolor=color,label=label)
    plt.plot(xvals[iter:len(yvals)],yvals[iter:],color=color,linewidth=lw,linestyle='-',markersize=ms,marker=marker,
                                                 markeredgecolor='k',markerfacecolor=color)

xvals = range(1,maxiter+1)
plt.plot(xvals,[-9 for i in range(maxiter)],'k--')
plt.annotate('tolerance',xy=(1,-9.4),fontsize=24)

left = 6.15
bottom = -12
width = 0.7
height = 12
right = left+width
top = bottom + height
rect = plt.Rectangle(xy=(left,bottom),width=width,height=height,color='lightgrey')
plt.text(0.5*(left+right),0.5*(bottom+top),'node failure',horizontalalignment='center',
        verticalalignment='center',rotation=90, color='k',fontsize=24)
fig.gca().add_artist(rect)

plt.xlim(1-0.25,maxiter+0.25)
plt.ylim(minres-0.25,maxres+0.25)

plt.xlabel('iteration')
plt.ylabel('log10(residual)')

plt.legend(numpoints=1)

plt.xticks(range(1,maxiter+1))
plt.yticks(range(minres,maxres+1))

ax.tick_params(pad=8)

plt.tight_layout()

fname = setup+'_residuals_allstrategies.png'
plt.savefig(fname, rasterized=True, transparent=True, bbox_inches='tight')

# plt.show()