import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

from pySDC.core.Errors import DataError

class fenicsx_mesh(object):
    """
    FEniCSx Function data type with arbitrary dimensions

    Attributes:
        values: contains the fenicsx Function
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Attribute:
            values: a dolfinx.fem.Function

        Args:
            init: can either be another fenicsx_mesh object to be copied, a fem.Function to be copied into values 
                  or a fem.FunctionSpace (with a constant value val to be assigned to the fem.Function)
            val: initial value (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        self.loc_to_glob_map = None

        # if init is another fenicsx_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.values = init.values.copy()
            self.loc_to_glob_map = init.loc_to_glob_map            
        elif isinstance(init, df.fem.Function):
            self.values = init.copy()            
        elif isinstance(init, df.fem.FunctionSpace):
            self.values = df.fem.Function(init)
            if isinstance(val,str) and val=="random":
                self.values.x.array[:] = np.random.random(self.values.x.array.shape)[:]
            elif isinstance(val,float):
                self.values.vector.set(val)
            elif isinstance(val,PETSc.Vec):
                self.values.vector.setArray(val)
            else:
                raise DataError('something went wrong during %s initialization' % type(init))
        else:
            raise DataError('something went wrong during %s initialization' % type(init))      
        
        self.size = len(self.values.split())

        if self.loc_to_glob_map is None:
            self.init_loc_to_glob_map()
        

    def init_loc_to_glob_map(self):
        self.loc_to_glob_map = []
        V = self.values.function_space
        for i in range(self.size):
            Vi, mapi = V.sub(i).collapse()
            self.loc_to_glob_map.append(mapi)

    def copy(self,other):
        if isinstance(other, type(self)):
            self.values.vector.setArray(other.values.vector)
        else:
            raise DataError("Type error: cannot copy %s to %s" % (type(other), type(self)))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (fenicsx_mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenicsx_mesh: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a + b changes a as well!
            # me = fenicsx_mesh(other)
            # loc_me = me.values.vector.localForm().__enter__()
            # loc_other = other.values.vector.localForm().__enter__()
            # loc_self = self.values.vector.localForm().__enter__()
            # loc_me.waxpy(1.,loc_other,loc_self)
            # me.values.vector.localForm().__exit__()
            # other.values.vector.localForm().__exit__()
            # self.values.vector.localForm().__exit__()
            # return me
            me = fenicsx_mesh(self)
            me.values.vector.axpy(1.,other.values.vector)
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))
        
    def __iadd__(self, other):
       
        if isinstance(other, type(self)):            
            self.values.vector.axpy(1.,other.values.vector)
            return self
        else:
            raise DataError("Type error: cannot iadd %s to %s" % (type(other), type(self)))
        
    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other (fenics_mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenics_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a - b changes a as well!            
            me = fenicsx_mesh(self)
            me.values.vector.axpy(-1.,other.values.vector)
            return me
        else:
            raise DataError("Type error: cannot sub %s from %s" % (type(other), type(self)))

    def __isub__(self, other):
       
        if isinstance(other, type(self)):            
            self.values.vector.axpy(-1.,other.values.vector)
            return self
        else:
            raise DataError("Type error: cannot isub %s to %s" % (type(other), type(self)))
        
    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            fenics_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            me = fenicsx_mesh(self)
            me.values.vector.scale(other)            
            return me
        elif isinstance(other,fenicsx_mesh):
            mult = fenicsx_mesh(init=self.values.function_space,val=0.)    
            for i in range(self.size):
                mult.values.sub(i).interpolate(fem.Expression(self.values.sub(i)*other.values.sub(i),self.values.function_space.sub(i).element.interpolation_points()))     
            return mult
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))
        
    def __imul__(self, other):
        """
        Overloading the inplace right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a float
        Returns:
            fenicsx_mesh: original values scaled by factor
        """

        if isinstance(other, float):            
            self.values.vector.scale(other)         
            return self
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: L2 norm of self.values (a fem.Function)
        """

        # take absolute values of the mesh values
        # self.values.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # self.sq_norm_form = df.fem.form(self.values**2 * ufl.dx)
        # norm_L2 = np.sqrt(self.values.function_space.mesh.comm.allreduce(df.fem.assemble_scalar(self.sq_norm_form), op=MPI.SUM))

        self.values.vector.normBegin()
        norm_L2 = self.values.vector.normEnd()
        norm_L2 /= np.sqrt(self.values.vector.getSize())

        # this also works for one processor, maybe not for more
        # norm_L2 = self.values.x.norm()/np.sqrt(self.values.x.array.size)

        return norm_L2
    
    def dot(self,other):
        self.values.vector.dotBegin(other.values.vector)
        res = self.values.vector.dotEnd(other.values.vector)
        return res

    def axpy(self, a, x):
        """
        Performs self.values = a*x.values+self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.axpy(a,x.values.vector)                                  
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))
        
    def aypx(self, a, x):
        """
        Performs self.values = x.values+a*self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.aypx(a,x.values.vector)                                  
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))
        
    def axpby(self, a, b, x):
        """
        Performs self.values = a*x.values+b*self.values
        """

        if isinstance(x, type(self)):
            self.values.vector.axpby(a,b,x.values.vector)                                  
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(x), type(self)))

    def zero(self):
        """
        Set to zero.
        """
        with self.values.vector.localForm() as loc_self:
                loc_self.set(0.)

    def ghostUpdate(self,addv,mode):        
        self.values.vector.ghostUpdate(addv, mode)

    def interpolate(self,other):
        if isinstance(other,fenicsx_mesh):
            for i in range(self.size):
                self.values.sub(i).interpolate(other.values.sub(i))
        else:
            raise DataError("Type error: cannot interpolate %s to %s" % (type(other), type(self)))
        
    def sub(self,i):
        return self.values.sub(i)

    @property
    def n_loc_dofs(self):
        return self.values.vector.getSizes()[0]
    
    @property
    def n_ghost_dofs(self):
        return self.values.x.array.size-self.n_loc_dofs
    
    def iadd_sub(self,other,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] += other.values.x.array[self.loc_to_glob_map[i]]

    def isub_sub(self,other,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] -= other.values.x.array[self.loc_to_glob_map[i]]

    def imul_sub(self,other,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] *= other

    def axpby_sub(self,a,b,x,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] = a*x.values.x.array[self.loc_to_glob_map[i]]\
                                                            +b*self.values.x.array[self.loc_to_glob_map[i]]

    def axpy_sub(self,a,x,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] += a*x.values.x.array[self.loc_to_glob_map[i]]

    def copy_sub(self,other,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] = other.values.x.array[self.loc_to_glob_map[i]]

    def zero_sub(self,indices):
        for i in indices:
            self.values.x.array[self.loc_to_glob_map[i]] = 0.

class rhs_fenicsx_mesh(object):
    """
    RHS data type for fenicsx_meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (fenicsx_mesh): implicit part
        expl (fenicsx_mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = fenicsx_mesh(init.impl)
            self.expl = fenicsx_mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, df.fem.FunctionSpace):
            self.impl = fenicsx_mesh(init, val=val)
            self.expl = fenicsx_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    # def __add__(self, other):

    #     if isinstance(other, rhs_fenicsx_mesh):
    #         me = rhs_fenicsx_mesh(self)
    #         me += other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
    
    # def __sub__(self, other):

    #     if isinstance(other, rhs_fenicsx_mesh):
    #         me = rhs_fenicsx_mesh(self)
    #         me -= other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __rmul__(self, other):

    #     if isinstance(other, float):
    #         me = rhs_fenicsx_mesh(self)
    #         me *= other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __iadd__(self, other):

    #     if isinstance(other, rhs_fenicsx_mesh):
    #         self.expl += other.expl
    #         self.impl += other.impl
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __isub__(self, other):

    #     if isinstance(other, rhs_fenicsx_mesh):
    #         self.expl -= other.expl
    #         self.impl -= other.impl
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __imul__(self, other):

    #     if isinstance(other, float):
    #         self.expl *= other
    #         self.impl *= other
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __abs__(self):
    #     return abs(self.expl+self.impl)
    
class exp_rhs_fenicsx_mesh(object):
    """
    RHS data type for fenicsx_meshes with implicit, explicit and exponential components

    This data type can be used to have RHS with 3 components (here implicit and explicit and exponential)

    Attributes:
        impl (fenicsx_mesh): implicit part
        expl (fenicsx_mesh): explicit part
        exp  (fenicsx_mesh): exponential part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = fenicsx_mesh(init.impl)
            self.expl = fenicsx_mesh(init.expl)
            self.exp = fenicsx_mesh(init.exp)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, df.fem.FunctionSpace):
            self.impl = fenicsx_mesh(init, val=val)
            self.expl = fenicsx_mesh(init, val=val)
            self.exp = fenicsx_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    # def __add__(self, other):

    #     if isinstance(other, exp_rhs_fenicsx_mesh):
    #         me = exp_rhs_fenicsx_mesh(self)
    #         me += other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
    
    # def __sub__(self, other):

    #     if isinstance(other, exp_rhs_fenicsx_mesh):
    #         me = exp_rhs_fenicsx_mesh(self)
    #         me -= other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __rmul__(self, other):

    #     if isinstance(other, float):
    #         me = exp_rhs_fenicsx_mesh(self)
    #         me *= other
    #         return me
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __iadd__(self, other):

    #     if isinstance(other, exp_rhs_fenicsx_mesh):
    #         self.expl += other.expl
    #         self.impl += other.impl
    #         self.exp += other.exp
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __isub__(self, other):

    #     if isinstance(other, exp_rhs_fenicsx_mesh):
    #         self.expl -= other.expl
    #         self.impl -= other.impl
    #         self.exp -= other.exp
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __imul__(self, other):

    #     if isinstance(other, float):
    #         self.expl *= other
    #         self.impl *= other
    #         self.exp *= other
    #         return self
    #     else:
    #         raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
        
    # def __abs__(self):
    #     return abs(self.expl+self.impl+self.exp)