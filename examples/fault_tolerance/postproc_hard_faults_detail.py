import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os

axis_font = {'fontname':'Arial', 'size':'8', 'family':'serif'}
fs = 8

setup = 'HEAT'
# setup = 'ADVECTION'

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

    rcParams['figure.figsize'] = 3.0, 2.5
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('Reds')
    pcol = plt.pcolor(residual.T, cmap=cmap, vmin=minres, vmax=maxres)
    pcol.set_edgecolor('face')

    plt.axis([0,maxiter,0,maxsteps])

    cax = plt.colorbar(pcol)
    cax.set_label('log10(residual)', **axis_font)
    cax.ax.tick_params(labelsize=fs)

    plt.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_xlabel('iteration', labelpad=1, **axis_font)
    ax.set_ylabel('step', labelpad=1, **axis_font)

    ax.set_xticks(np.arange(maxiter)+0.5, minor=False)
    ax.set_yticks(np.arange(maxsteps)+0.5, minor=False)
    ax.set_xticklabels(np.arange(maxiter)+1, minor=False)
    ax.set_yticklabels(np.arange(maxsteps), minor=False)
    
    # Set every second label to invisible
    for label in ax.xaxis.get_ticklabels()[::2]:
      label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
      label.set_visible(False)

    ax.tick_params(pad=2)
    plt.tight_layout()

    if strategy is not 'NOFAULT':
        plt.text(step-1+0.5,iter+0.5,'x',horizontalalignment='center',verticalalignment='center')

    fname = setup+'_steps_vs_iteration_hf_'+str(step)+'x'+str(iter)+'_'+strategy+'.pdf'
    plt.savefig(fname, bbox_inches='tight')
    os.system('pdfcrop '+fname+' '+fname)

#
#
#
rcParams['figure.figsize'] = 6.0, 3.0
fig, ax = plt.subplots()
maxiter = 0
lw = 2
ms = 8


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
plt.annotate('tolerance',xy=(1,-9.4),fontsize=fs)

left = 6.15
bottom = -12
width = 0.7
height = 12
right = left+width
top = bottom + height
rect = plt.Rectangle(xy=(left,bottom),width=width,height=height,color='lightgrey')
plt.text(0.5*(left+right),0.5*(bottom+top),'node failure',horizontalalignment='center',
        verticalalignment='center',rotation=90, color='k',fontsize=fs)
fig.gca().add_artist(rect)

plt.xlim(1-0.25,maxiter+0.25)
plt.ylim(minres-0.25,maxres+0.25)

plt.xlabel('iteration', **axis_font)
plt.ylabel('log10(residual)', **axis_font)
ax.xaxis.labelpad = 0
ax.yaxis.labelpad = 0
plt.tick_params(axis='both', which='major', labelsize=fs)

plt.legend(numpoints=1, fontsize=fs)

plt.xticks(range(1,maxiter+1))
plt.yticks(range(minres,maxres+1))

ax.tick_params(pad=2)

plt.tight_layout()

fname = setup+'_residuals_allstrategies.pdf'
plt.savefig(fname, bbox_inches='tight')
os.system('pdfcrop '+fname+' '+fname)

# plt.show()