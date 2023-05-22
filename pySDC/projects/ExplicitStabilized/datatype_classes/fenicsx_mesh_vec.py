from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh import fenicsx_mesh
import numpy as np
from pySDC.core.Errors import DataError

class fenicsx_mesh_vec(object):
    """
    Vector of FEniCSx Function data type 
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init, fenicsx_mesh_vec):
            self.val_list = [fenicsx_mesh(init_k) for init_k in init.val_list]
        else:
            self.val_list = [fenicsx_mesh(init,val) for _ in range(size)]
        self.size = len(self.val_list)

    def __getitem__(self,key):
        return self.val_list[key]
    
    def __setitem__(self,key):
        return self.val_list[key]
    
    def copy(self,other):
        for i in range(self.size):
            self.val_list[i].copy(other[i])    
    
    def __add__(self, other):
        me = fenicsx_mesh_vec(self)
        me += other
        return me
    
    def __sub__(self, other):
        me = fenicsx_mesh_vec(self)
        me -= other
        return me
    
    def __rmul__(self, other):
        me = fenicsx_mesh_vec(self)
        me *= other
        return me
    
    def __iadd__(self, other):
        for i in range(self.size):
            self.val_list[i] += other.val_list[i]
        return self

    def __isub__(self, other):
        for i in range(self.size):
            self.val_list[i] -= other.val_list[i]
        return self

    def __imul__(self, other):
        for i in range(self.size):
            self.val_list[i] *= other
        return self
    
    def __abs__(self):
        l2_norm = 0.
        for val in self.val_list:
            l2_norm += abs(val)**2
        return np.sqrt(l2_norm/self.size)
    
    def axpy(self,a,x):
        for i in range(self.size):
            self[i].axpy(a,x[i])

    def aypx(self,a,x):
        for i in range(self.size):
            self[i].aypx(a,x[i])

    def axpby(self,a,b,x):
        for i in range(self.size):
            self[i].axpby(a,b,x[i])

    def zero(self):
        for i in range(self.size):
            with self[i].values.vector.localForm() as loc_self:
                loc_self.set(0.)
    
    def ghostUpdate(self,addv,mode):
        for i in range(self.size):
            self[i].values.vector.ghostUpdate(addv, mode)

    def interpolate(self,other):
        if isinstance(other,fenicsx_mesh_vec):
            if self.size==other.size:
                for i in range(self.size):
                    self.val_list[i].values.interpolate(other.val_list[i].values)
            else:
                raise DataError("Size error: interpolating vectors have different sizes.")
        else:
            raise DataError("Type error: cannot interpolate %s to %s" % (type(other), type(self)))
        
    def sub(self,i):
        return self.val_list[i].values

    @property
    def n_loc_dofs(self):
        return self.val_list[0].values.vector.getSizes()[0]
    
    @property
    def n_ghost_dofs(self):
        return self.val_list[0].values.x.array.size-self.n_loc_dofs
    
    def iadd_sub(self,other,indices):
        for i in indices:
            self.val_list[i] += other.val_list[i]

    def isub_sub(self,other,indices):
        for i in indices:
            self.val_list[i] -= other.val_list[i]

    def imul_sub(self,other,indices):
        for i in indices:
            self.val_list[i] *= other

    def axpby_sub(self,a,b,x,indices):
        for i in indices:
            self.val_list[i].axpby(a,b,x.val_list[i])

    def axpy_sub(self,a,x,indices):
        for i in indices:
            self.val_list[i].axpy(a,x.val_list[i])

    # def aypx_sub(self,a,x,indices):
    #     for i in indices:
    #         self.val_list[i].aypx(a,x.val_list[i])

    def copy_sub(self,other,indices):
        for i in indices:
            self.val_list[i].copy(other.val_list[i])

    def zero_sub(self,indices):
        for i in indices:
            self.val_list[i].zero()

    def swap_sub(self,other,indices):
        for i in indices:
            self.val_list[i], other.val_list[i] = other.val_list[i], self.val_list[i]
    


class rhs_fenicsx_mesh_vec(object):
    """
    Vector of rhs FEniCSx Function data type 
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init,rhs_fenicsx_mesh_vec):
            self.expl = fenicsx_mesh_vec(init.expl)
            self.impl = fenicsx_mesh_vec(init.impl)
            self.size = len(self.expl.val_list)
        else:
            self.expl = fenicsx_mesh_vec(init,val,size)
            self.impl = fenicsx_mesh_vec(init,val,size)
            self.size = size

class exp_rhs_fenicsx_mesh_vec(object):
    """
    Vector of rhs FEniCSx Function data type 
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init,exp_rhs_fenicsx_mesh_vec):
            self.expl = fenicsx_mesh_vec(init.expl)
            self.impl = fenicsx_mesh_vec(init.impl)
            self.exp = fenicsx_mesh_vec(init.exp)
            self.size = len(self.expl.val_list)
        else:
            self.expl = fenicsx_mesh_vec(init,val,size)
            self.impl = fenicsx_mesh_vec(init,val,size)
            self.exp = fenicsx_mesh_vec(init,val,size)
            self.size = size