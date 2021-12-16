from pySDC.core.Errors import TransferError
from pySDC.core.SpaceTransfer import space_transfer
from pySDC.playgrounds.pmesh.PMESH_datatype_NEW import pmesh_datatype, rhs_imex_pmesh
import time

class pmesh_to_pmesh(space_transfer):
    """
    Custon base_transfer class, implements Transfer.py

    This implementation can restrict and prolong between PMESH datatypes meshes with FFT for periodic boundaries

    """

    def __init__(self, fine_prob, coarse_prob, params):
        """
        Initialization routine

        Args:
            fine_prob: fine problem
            coarse_prob: coarse problem
            params: parameters for the transfer operators
        """
        # invoke super initialization
        super(pmesh_to_pmesh, self).__init__(fine_prob, coarse_prob, params)
        self.tmp_F = self.fine_prob.pm.create(type='complex')

    def restrict(self, F):
        """
        Restriction implementation

        Args:
            F: the fine level data (easier to access than via the fine attribute)
        """
        t0 = time.perf_counter()
        if isinstance(F, pmesh_datatype):
            G = self.coarse_prob.dtype_u(self.coarse_prob.init)
            # convert numpy array to RealField
            tmp_F = self.fine_prob.pm.create(type='complex', value=F.values)
            # tmp_G = self.coarse_prob.pm.create(type='complex')
            # resample fine to coarse
            # tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            # tmp_F.resample(tmp_G)
            tmp_F = tmp_F.c2r(out=Ellipsis)
            tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            tmp_G = tmp_G.r2c(out=Ellipsis)
            # copy values to data structure
            G.values = tmp_G.value
        elif isinstance(F, rhs_imex_pmesh):
            G = self.coarse_prob.dtype_f(self.coarse_prob.init)
            # convert numpy array to RealField
            tmp_F = self.fine_prob.pm.create(type='complex', value=F.impl.values)
            # tmp_G = self.coarse_prob.pm.create(type='complex')
            # resample fine to coarse
            # tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            # tmp_F.resample(tmp_G)
            tmp_F = tmp_F.c2r(out=Ellipsis)
            tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            tmp_G = tmp_G.r2c(out=Ellipsis)
            # copy values to data structure
            G.impl.values[:] = tmp_G.value
            # convert numpy array to RealField
            tmp_F = self.fine_prob.pm.create(type='complex', value=F.expl.values)
            # tmp_G = self.coarse_prob.pm.create(type='complex')
            # resample fine to coarse
            # tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            # tmp_F.resample(tmp_G)
            tmp_F = tmp_F.c2r(out=Ellipsis)
            tmp_G = self.coarse_prob.pm.upsample(tmp_F, keep_mean=True)
            tmp_G = tmp_G.r2c(out=Ellipsis)
            # copy values to data structure
            G.expl.values[:] = tmp_G.value
        else:
            raise TransferError('Unknown data type, got %s' % type(F))
        t1 = time.perf_counter()
        print(f'Space restrict: {t1-t0}')
        return G

    def prolong(self, G):
        """
        Prolongation implementation

        Args:
            G: the coarse level data (easier to access than via the coarse attribute)
        """
        t0 = time.perf_counter()
        if isinstance(G, pmesh_datatype):
            F = self.fine_prob.dtype_u(self.fine_prob.init)
            # convert numpy array to RealField
            # tmp_F = self.fine_prob.pm.create(type='complex')
            tmp_G = self.coarse_prob.pm.create(type='complex', value=G.values)
            # resample coarse to fine
            tmp_G.resample(self.tmp_F)
            # copy values to data structure
            F.values = self.tmp_F.value
        elif isinstance(G, rhs_imex_pmesh):
            F = self.fine_prob.dtype_f(self.fine_prob.init)
            # convert numpy array to RealField
            # tmp_F = self.fine_prob.pm.create(type='complex')
            tmp_G = self.coarse_prob.pm.create(type='complex', value=G.impl.values + G.expl.values)
            # resample coarse to fine
            tmp_G.resample(self.tmp_F)
            # copy values to data structure
            F.impl.values = self.tmp_F.value / 2
            # convert numpy array to RealField
            # tmp_F = self.fine_prob.pm.create(type='complex')
            # tmp_G = self.coarse_prob.pm.create(type='complex', value=G.expl.values)
            # # resample coarse to fine
            # tmp_G.resample(self.tmp_F)
            # copy values to data structure
            F.expl.values = self.tmp_F.value / 2
        else:
            raise TransferError('Unknown data type, got %s' % type(G))
        t1 = time.perf_counter()
        print(f'Space interpolate: {t1 - t0}')
        return F
