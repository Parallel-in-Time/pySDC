from __future__ import division
from pySDC.Hooks import hooks
from pySDC.Stats import stats
import dolfin as df

file = df.File('/Users/robert/Scratch/output1d/grayscott.pvd') # dirty, but this has to be unique (and not per step or level)

class fenics_output(hooks):

    def __init__(self):
        super(fenics_output,self).__init__()
        pass

    def dump_pre(self,status):
        """
        Overwrite standard dump at the beginning

        Args:
            status: status object per step
        """
        super(fenics_output,self).dump_pre(status)

        L = self.level
        v = L.u[0].values
        v.rename('func','label')

        file << v



    def dump_step(self,status):
        """
        Overwrite standard dump per step

        Args:
            status: status object per step
        """
        super(fenics_output,self).dump_step(status)

        L = self.level
        # u1,u2 = df.split(L.uend.values)
        v = L.uend.values
        v.rename('func','label')

        file << v


        return None
