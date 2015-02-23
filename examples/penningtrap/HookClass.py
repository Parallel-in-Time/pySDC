from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class particles_output(hooks):

    def __init__(self):
        """
        Initialization of particles output
        """
        super(particles_output,self).__init__()

        # add figure object for further use
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim3d([-20,20])
        self.ax.set_ylim3d([-20,20])
        self.ax.set_zlim3d([-20,20])
        plt.ion()
        self.sframe = None


    def dump_pre(self,status):
        """
        Overwrite standard dump at the beginning

        Args:
            status: status object per step
        """
        super(particles_output,self).dump_pre(status)

        # some abbreviations
        L = self.level
        part = L.u[0]
        N = L.prob.nparts
        w = np.array([1,1,-2])

        # compute (slowly..) the potential at u0
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0,i):
                dist2 = np.linalg.norm(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3],2)**2+L.prob.sig**2
                fpot[i] += part.q[j]/np.sqrt(dist2)
            for j in range(i+1,N):
                dist2 = np.linalg.norm(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3],2)**2+L.prob.sig**2
                fpot[i] += part.q[j]/np.sqrt(dist2)
            fpot[i] -= L.prob.omega_E**2 * part.m[i]/part.q[i]/2.0 \
                                         * np.dot(w,part.pos.values[3*i:3*i+3]*part.pos.values[3*i:3*i+3])

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n]*fpot[n]
            ekin += part.m[n]/2.0*np.dot(part.vel.values[3*n:3*n+3],part.vel.values[3*n:3*n+3])

        print('Energy (pot/kin/tot): %12.4f / %12.4f / %12.4f' %(epot,ekin,epot+ekin))

        stats.add_to_stats(type='etot', value=epot+ekin)



    def dump_step(self,status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(particles_output,self).dump_step(status)

        # some abbreviations
        L = self.level
        part = L.uend
        N = L.prob.nparts
        w = np.array([1,1,-2])

        # compute (slowly..) the potential at uend
        fpot = np.zeros(N)
        for i in range(N):
            # inner loop, omit ith particle
            for j in range(0,i):
                dist2 = np.linalg.norm(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3],2)**2+L.prob.sig**2
                fpot[i] += part.q[j]/np.sqrt(dist2)
            for j in range(i+1,N):
                dist2 = np.linalg.norm(part.pos.values[3*i:3*i+3]-part.pos.values[3*j:3*j+3],2)**2+L.prob.sig**2
                fpot[i] += part.q[j]/np.sqrt(dist2)
            fpot[i] -= L.prob.omega_E**2 * part.m[i]/part.q[i]/2.0 \
                                         * np.dot(w,part.pos.values[3*i:3*i+3]*part.pos.values[3*i:3*i+3])

        # add up kinetic and potntial contributions to total energy
        epot = 0
        ekin = 0
        for n in range(N):
            epot += part.q[n]*fpot[n]
            ekin += part.m[n]/2.0*np.dot(part.vel.values[3*n:3*n+3],part.vel.values[3*n:3*n+3])

        print('Energy (pot/kin/tot) at step %i: %12.4f / %12.4f / %12.4f' %(status.step,epot,ekin,epot+ekin))

        stats.add_to_stats(step=status.step, time=status.time, type='etot', value=epot+ekin)

        # print('plotting particles...')


        # oldcol = self.sframe
        # # self.sframe = self.ax.scatter(L.uend.pos.values[0],L.uend.pos.values[1],L.uend.pos.values[2])
        self.sframe = self.ax.scatter(L.uend.pos.values[0::3],L.uend.pos.values[1::3],L.uend.pos.values[2::3])
        # # Remove old line collection before drawing
        # if oldcol is not None:
        #     self.ax.collections.remove(oldcol)
        # plt.pause(0.001)

        return None
