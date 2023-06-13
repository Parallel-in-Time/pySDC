# import matplotlib

# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.Boris.penningtrap_HookClass import convergence_data

class Convergence(object):
    """
    Initialize necessary params for the convergence plots
    """
    def __init__(self,controller_params, description, time_iter=3, K_iter=[1, 2, 3],Tend=2, axes=[1], cwd=''):
        self.controller_params=controller_params
        self.description=description
        self.time_iter=time_iter
        self.K_iter=K_iter
        self.Tend=Tend
        self.axes=axes
        self.cwd=cwd
        self.quad_type=self.description['sweeper_params']['quad_type']
        self.num_nodes=self.description['sweeper_params']['num_nodes']

    # run local convergence rate and plot the graph
    @property
    def run_local_error(self):
        self.error_type='Local'
        self.compute_local_error_data()
        self.find_approximate_order()
        self.plot_convergence()

    # run global convergence rate and plot the graph
    @property
    def run_global_error(self):
        self.error_type='Global'
        self.compute_global_error_data()
        self.find_approximate_order(filename='data/Global-conv-data.txt')
        self.plot_convergence()


    """
    Plot convergence order plots for the position and velocity
    """
    def plot_convergence(self):

        fs=16
        [N, time_data, error_data, order_data, convline]=self.organize_data(filename='data/{}-conv-data.txt'.format(self.error_type))

        color = ['r', 'brown', 'g', 'blue']
        shape = ['o', 'd', 's', 'x']
        rcParams['figure.figsize']=7.44, 6.74
        rcParams['pgf.rcfonts'] = False
        rcParams['xtick.labelsize']=fs
        rcParams['ytick.labelsize']=fs
        rcParams['mathtext.fontset']='cm'
        plt.rc('font', size=16)
        fig1, ax=  plt.subplots(1, len(self.axes))

        for count, value in enumerate(self.axes):
            if len(self.axes)==1:
                axes=ax
            else:
                axes=ax[count]
            for ii in range(0, N):
                axes.loglog(time_data[ii,:], convline['pos'][value, ii, :], color='black', markersize=fs-2, linewidth=3)
                axes.loglog(time_data[ii,:], error_data['pos'][value, ii, :], ' ', color=color[ii], marker=shape[ii], markersize=fs-2, label='k={}'.format(int(self.K_iter[ii])))
                if value==2:
                    axes.text(time_data[ii,1], 0.3*convline['pos'][value,ii, 1],r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii,0,1]),size=fs+2,)
                else:
                    axes.text(time_data[ii,1], 0.3*convline['pos'][value, ii,1],r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['pos'][ii,0, 0]),size=fs+2,)

                if self.error_type=='Local':
                    axes.set_ylabel('$\Delta x^{\mathrm{(abs)}}_{%d}$'%(value+1), fontsize=fs+5)
                else:
                    axes.set_ylabel('$\Delta x^{\mathrm{(rel)}}_{%d}$'%(value+1), fontsize=fs+5)
                axes.set_title('{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes), fontsize=fs+5)
                axes.set_xlabel('$\omega_{B} \cdot \Delta t$', fontsize=fs+5)
                axes.legend(loc='best')
                fig1.tight_layout()
        fig1.savefig('figures/{}_conv_plot_pos{}.pdf'.format(self.error_type, value+1))

        fig2, ax=  plt.subplots(1, len(self.axes))
        for count, value in enumerate(self.axes):
            if len(self.axes)==1:
                axes=ax
            else:
                axes=ax[count]
            for ii in range(0, N):
                axes.loglog(time_data[ii,:], convline['vel'][value, ii,:], color='black', markersize=fs-2, linewidth=3)
                axes.loglog(time_data[ii,:], error_data['vel'][value, ii,:], ' ', color=color[ii], marker=shape[ii], markersize=fs-2, label='k={}'.format(int(self.K_iter[ii])) )

                if value==2:
                    axes.text(time_data[ii,1], 0.3*convline['vel'][value, ii,1],r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0,1]),size=fs+2)
                else:
                    axes.text(time_data[ii,1], 0.3*convline['vel'][value, ii,1],r"$\mathcal{O}(\Delta t^{%d})$" % (order_data['vel'][ii, 0,0]),size=fs+2)

                if self.error_type=='Local':
                    axes.set_ylabel('$\Delta v^{\mathrm{(abs)}}_{%d}$'%(value+1), fontsize=fs+5)
                else:
                    axes.set_ylabel('$\Delta v^{\mathrm{(rel)}}_{%d}$'%(value+1), fontsize=fs+5)
                axes.set_title('{} order of convergence, $M={}$'.format(self.error_type, self.num_nodes), fontsize=fs+5)
                axes.set_xlabel('$\omega_{B} \cdot \Delta t$', fontsize=fs+5)
                axes.legend(loc='best')
                fig2.tight_layout()
        fig2.savefig('figures/{}_conv_plot_vel{}.pdf'.format(self.error_type, value+1))
        plt.show()

    """
    Compute local convergece data and save it
    """
    def compute_local_error_data(self):
        step_params=dict()
        dt_val=self.description['level_params']['dt']

        for order in self.K_iter:
            # define storage for the local error
            error={'pos': np.zeros([1,3]), 'vel': np.zeros([1,3])}

            step_params['maxiter']=order
            self.description['step_params']=step_params

            if order ==1:
                file=open(self.cwd +'data/Local-conv-data.txt', 'w')
            else:
                file= open(self.cwd + 'data/Local-conv-data.txt', 'a')

            for ii in range(0, self.time_iter):

                dt=dt_val/2**ii

                self.description['level_params']['dt']=dt
                self.description['level_params']=self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(num_procs=1, controller_params=self.controller_params, description=self.description)


                # set time parameters
                t0 = 0.0
                Tend= dt

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                # compute exact solution and compare
                uex = P.u_exact(Tend)

                #find order of quadrature rule
                coll_order=controller.MS[0].levels[0].sweep.coll.order


                # find order of convergence for the postion and velocity
                order_pos=self.local_order_pos(order, coll_order)
                order_vel=self.local_order_vel(order, coll_order)
                # evaluate error
                error['pos']=np.abs((uex-uend).pos).T
                error['vel']=np.abs((uex-uend).vel).T
                dt_omega=dt*self.description['problem_params']['omega_B']
                file.write(str(dt_omega) + " * " + str(order_pos) + " * " +str(error['pos']) + " * " + str(order_vel) + " * " + str(error['vel']) + '\n')

            file.close()

    """

    Compute global convergence data and save it data folder

    """
    def compute_global_error_data(self):
        convergence_data.Tend=self.Tend
        self.controller_params['hook_class']=convergence_data
        step_params=dict()
        dt_val=self.description['level_params']['dt']

        values=['position', 'velocity']

        error = dict()

        for order in self.K_iter:
            # define storage for the global error
            error['position'] = np.zeros([3, self.time_iter])
            error['velocity']= np.zeros([3, self.time_iter])
            u_val = dict()
            uex_val = dict()

            step_params['maxiter']=order
            self.description['step_params']=step_params

            if order ==1:
                file=open(self.cwd +'data/Global-conv-data.txt', 'w')
            else:
                file= open(self.cwd + 'data/Global-conv-data.txt', 'a')

            for ii in range(0, self.time_iter):

                dt=dt_val/2**ii

                self.description['level_params']['dt']=dt
                self.description['level_params']=self.description['level_params']

                # instantiate the controller (no controller parameters used here)
                controller = controller_nonMPI(num_procs=1, controller_params=self.controller_params, description=self.description)


                # set time parameters
                t0 = 0.0
                # Tend = dt
                Tend= self.Tend

                # get initial values on finest level
                P = controller.MS[0].levels[0].prob
                uinit = P.u_init()

                # call main function to get things done...
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                # extract values from stats
                extract_stats = filter_stats(stats, type="error")
                sortedlist_stats = sort_stats(extract_stats, sortby="time")

                sortedlist_stats[0][1]["position_ex"] = P.u_exact(Tend).pos
                sortedlist_stats[0][1]["velocity_ex"] = P.u_exact(Tend).vel
                # sort values and compute error
                for mm, nn in enumerate(values):
                    data = sortedlist_stats[0][1][nn].values()
                    u_val[nn] = np.array(list(data))
                    u_val[nn] = u_val[nn].reshape(np.shape(u_val[nn])[0], np.shape(u_val[nn])[1])

                    data = sortedlist_stats[0][1][nn + "_exact"].values()
                    uex_val[nn] = np.array(list(data))
                    uex_val[nn] = uex_val[nn].reshape(np.shape(uex_val[nn])[0], np.shape(uex_val[nn])[1])

                    error[nn][:, ii] = self.relative_error(uex_val[nn], u_val[nn])
                coll_order=controller.MS[0].levels[0].sweep.coll.order
                global_order=self.global_order(order, coll_order)
                dt_omega=dt*self.description['problem_params']['omega_B']

                file.write(str(dt_omega) + " * " + str(global_order) + " * " +str(error['position'][:,ii]) + " * " + str(global_order) + " * " + str(error['velocity'][:,ii]) + '\n')
            file.close()

    # find expected local convergence order for position
    def local_order_pos(self, order_K, order_quad):
        if self.quad_type=='GAUSS' or self.quad_type=='RADAU-RIGHT':
            return np.array([np.min([order_K+2, order_quad]), np.min([2*order_K+3, order_quad])])
        elif self.quad_type=='LOBATTO' or self.quad_type=='RADAU-LEFT':
            return np.array([np.min([order_K+2, order_quad]), np.min([2*order_K+3, order_quad])])
        else:
            raise NotImplementedError('order of convergence explicitly not implemented ')

    # find expected local convergence order for velocity
    def local_order_vel(self, order_K, order_quad):
        if self.quad_type=='GAUSS' or self.quad_type=='RADAU-RIGHT':
            return np.array([np.min([order_K+1, order_quad]), np.min([2*order_K+2, order_quad])])
        elif self.quad_type=='LOBATTO' or self.quad_type=='RADAU-LEFT':
            return np.array([np.min([order_K+1, order_quad]), np.min([2*order_K+2, order_quad])])
        else:
            raise NotImplementedError('order of convergence explicitly not implemented ')

    # find expected global convergence order
    def global_order(self, order_K, order_quad):
        if self.quad_type=='GAUSS' or self.quad_type=='RADAU-RIGHT':
            return np.array([np.min([order_K, order_quad]), np.min([2*order_K, order_quad])])
        elif self.quad_type=='LOBATTO' or self.quad_type=='RADAU-LEFT':
            return np.array([np.min([order_K, order_quad]), np.min([2*order_K, order_quad])])+2
        else:
            raise NotImplementedError('order of convergence explicitly not implemented ')

    # compute relative error
    def relative_error(self, u_ex, u):
        return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0)/ np.linalg.norm(u_ex, np.inf, 0)

    """

    Seperate data to plot for the graph

    """
    def organize_data(self, filename='data/Local-conv-data.txt', time_iter=None):
        """
        Organize data according to plot
        Args:
            filename (string): data to find approximate order
        """
        if time_iter==None:
            time_iter=self.time_iter

        time=np.array([])
        order={'pos':np.array([]).reshape([0,2]), 'vel': np.array([]).reshape([0,2])}
        error={'pos':np.array([]).reshape([0,3]), 'vel': np.array([]).reshape([0,3])}

        file=open(self.cwd + filename, 'r')

        while True:
            line =file.readline()
            if not line:
                break

            items=str.split(line, " * ", 5)
            time=np.append(time, float(items[0]))
            order['pos']=np.vstack((order['pos'], np.matrix(items[1]).A[0]))
            order['vel']=np.vstack((order['vel'], np.matrix(items[3]).A[0]))
            error['pos'] = np.vstack((error['pos'], np.matrix(items[2]).A[0]))
            error['vel'] = np.vstack((error['vel'], np.matrix(items[4]).A[0]))



        N=int(np.size(time)/time_iter)

        error_data = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}
        order_data = {'pos': np.zeros([N,time_iter,  2]), 'vel': np.zeros([N,time_iter,  2])}
        time_data = np.zeros([N, time_iter])
        convline = {'pos': np.zeros([3, N, time_iter]), 'vel': np.zeros([3, N, time_iter])}

        time_data=time.reshape([N, time_iter])

        order_data['pos'][:,:,0]=order['pos'][:, 0].reshape([N, time_iter])
        order_data['pos'][:,:,1]=order['pos'][:, 1].reshape([N, time_iter])
        order_data['vel'][:, :, 0]=order['vel'][:, 0].reshape([N, time_iter])
        order_data['vel'][:, :, 1]=order['vel'][:, 1].reshape([N, time_iter])

        for ii in range(0, 3):

            error_data['pos'][ii, :, :]=error['pos'][:,ii].reshape([N, time_iter])
            error_data['vel'][ii, :, :]=error['vel'][:,ii].reshape([N, time_iter])


        for jj in range(0, 3):
            if jj==2:
                convline['pos'][jj,:,:]= ((time_data/time_data[0,0]).T**order_data['pos'][:,jj,1]).T*error_data['pos'][jj,:,0][:,None]
                convline['vel'][jj,:,:]= ((time_data/time_data[0,0]).T**order_data['vel'][:,jj,1]).T*error_data['vel'][jj,:,0][:,None]
            else:
                convline['pos'][jj,:,:]= ((time_data/time_data[0,0]).T**order_data['pos'][:,jj,0]).T*error_data['pos'][jj,:,0][:,None]
                convline['vel'][jj,:,:]= ((time_data/time_data[0,0]).T**order_data['vel'][:,jj,0]).T*error_data['vel'][jj,:,0][:,None]

        return [N, time_data, error_data, order_data, convline]

    # find approximate order
    def find_approximate_order(self, filename='data/Local-conv-data.txt'):

        [N, time_data, error_data, order_data, convline]=self.organize_data(self.cwd + filename)
        approx_order={'pos': np.zeros([1,N]), 'vel':np.zeros([1,N])}

        for jj in range(0,3):
            if jj==0:
                file=open(self.cwd + 'data/{}_order_vs_approxorder.txt'.format('Local'), 'w')
            else:
                file = open(self.cwd + 'data/{}_order_vs_approxorder.txt'.format('Local'), 'a')

            for ii in range(0, N):

                approx_order['pos'][0,ii]=np.polyfit(np.log(time_data[ii,:]), np.log(error_data['pos'][jj,ii,: ]),1)[0].real
                approx_order['vel'][0,ii]=np.polyfit(np.log(time_data[ii,:]), np.log(error_data['vel'][jj,ii, :]),1)[0].real
            if jj==2:
                file.write(str(order_data['pos'][:, jj, 1]) + ' * ' + str(approx_order['pos'])+ ' * ' + str(order_data['vel'][:,jj, 1]) +' * ' +str(approx_order['vel']) + '\n')
            else:
                file.write(str(order_data['pos'][:, jj, 0]) + ' * ' + str(approx_order['pos'])+ ' * ' + str(order_data['vel'][:,jj, 0]) +' * ' +str(approx_order['vel']) + '\n')
