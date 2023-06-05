import numpy as np
from pySDC.core.Problem import ptype
# from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError
import math
try:
    # TODO : mpi4py cannot be imported before dolfin when using fenics mesh
    # see https://github.com/Parallel-in-Time/pySDC/pull/285#discussion_r1145850590
    # This should be dealt with at some point
    from mpi4py import MPI
except ImportError:
    MPI = None

class my_mesh(np.ndarray):

    def __new__(cls, init, val=0.0, offset=0, buffer=None, strides=None, order=None):
        
        if isinstance(init, my_mesh):
            obj = np.ndarray.__new__(
                cls, shape=init.shape, dtype=init.dtype, buffer=buffer, offset=offset, strides=strides, order=order
            )
            obj[:] = init[:]
            # obj._comm = init._comm
        elif isinstance(init, int):
            obj = np.ndarray.__new__(
                cls, init, dtype=np.dtype('float64'), buffer=buffer, offset=offset, strides=strides, order=order
            )
            obj.fill(val)
            # obj._comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    def copy(self,other):
        self[:] = other[:]

    def __abs__(self):
        return np.linalg.norm(self)

class exp_imex_mesh(object):

    def __init__(self, init, val=0.0):

        if isinstance(init, type(self)):
            self.impl = my_mesh(init.impl)
            self.expl = my_mesh(init.expl)
            self.exp = my_mesh(init.exp)
        elif isinstance(init, int):
            self.impl = my_mesh(init, val=val)
            self.expl = my_mesh(init, val=val)
            self.exp = my_mesh(init, val=val)
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

class debug_exp_runge_kutta_1(ptype):

    dtype_u = my_mesh
    dtype_f = exp_imex_mesh

    def __init__(self,**problem_params):
        self.know_exact = True
        self.nvars = 2
        self.init = self.nvars

        super(debug_exp_runge_kutta_1, self).__init__(self.init)
        self._makeAttributeAndRegister(*problem_params.keys(),localVars=problem_params,readOnly=True)

        self.lmbda_cte = [-5.,-5.]
        self.lmbda = lambda t:  [self.lmbda_cte[0]*np.ones_like(t), self.lmbda_cte[1]*np.ones_like(t)]
        self.f = lambda t:  [np.cos(t), np.cos(t)]
        self.yinf = self.dtype_u(self.init)
        self.yinf[:] = np.array([5.,5.])
        self.y0 = self.dtype_u(self.init)
        self.y0[:] = np.array([1.,1.])

        self.t0 = 0.
        self.Tend = 1.

    def initial_value(self):
        return self.y0
    
    def u_exact(self,t):
        def eval_rhs(t,u):
            f = self.eval_f(u,t,True,True,True)
            return f.impl+f.expl+f.exp
        # u_ex = self.generate_scipy_reference_solution(eval_rhs,t,self.y0,self.t0)        
        u_ex = self.y0
        N = int(1e3)
        dt = (t-self.t0)/(N-1)
        t = np.linspace(self.t0,t,N)
        for i in range(N):
            u_ex += dt*eval_rhs(t[i],u_ex)        

        return u_ex

    def compute_errors(self,uh,t):
        u_ex = self.u_exact(t)
        errors_L2 = np.abs(uh-u_ex)
        norms_L2_sol = np.abs(u_ex)
        rel_errors_L2 = errors_L2/norms_L2_sol
        
        print(f"L2-errors: {errors_L2}")
        print(f"Relative L2-errors: {rel_errors_L2}")
        print(f'u_ex = {u_ex}')
        print(f'u_h = {uh}')

    def get_size(self):        
        return self.nvars
    
    def solve_system(self, rhs, factor, u0, t, u_sol = None):                

        if u_sol is None:
            u_sol = self.dtype_u(self.init)                

        l = self.lmbda(t+factor)
        u_sol[0] = rhs[0]-factor*l[0]*self.yinf[0]
        u_sol[0] = u_sol[0]/(1.-factor*l[0])
        u_sol[1] = rhs[1]

        return u_sol
    
    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None):
    
        if fh is None:
            fh = self.dtype_f(self.init,val=0.)    

        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl = self.eval_f_stiff(u,t,fh.impl)

        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:                                                
            fh.expl = self.eval_f_nonstiff(u,t,fh.expl) 
                
        # evaluate exponential (super stiff) part M^-1*f_exp(u,t)
        if eval_exp:                                                
            fh.exp = self.eval_f_exp(u,t,fh.exp) 
                            
        return fh
    
    def eval_f_nonstiff(self,u,t,fh_nonstiff):

        ft = self.f(t)
        fh_nonstiff[0] = ft[0]
        fh_nonstiff[1] = 0.

        return fh_nonstiff

    def eval_f_stiff(self,u,t,fh_stiff):

        l = self.lmbda(t)
        fh_stiff[0] = l[0]*(u[0]-self.yinf[0])
        fh_stiff[1] = 0.

        return fh_stiff
    
    def eval_f_exp(self,u,t,fh_exp):
        
        l = self.lmbda(t)
        ft = self.f(t)
        fh_exp[0] = 0.
        fh_exp[1] = l[1]*(u[1]-self.yinf[1]) + ft[1]

        return fh_exp
    
    def phi_eval(self, u, dt, t, k, phi = None):

        if phi is None:
            phi = self.dtype_u(self.init,val=0.)    

        l = self.lmbda(t)
        z = dt*l[1]
        phi_k = math.exp(z) 
        fac_k = 1.
        for i in range(1,k+1):
            fac_k = fac_k*k
            phi_k = (phi_k-1./fac_k)/z

        phi[0] = 1.
        phi[1] = phi_k

        return phi
    
    def lmbda_eval(self, u, t, lmbda = None):

        if lmbda is None:
            lmbda = self.dtype_u(self.init,val=0.)    

        lmbda = self.lmbda(t)
        
        return lmbda
    
    def phi_one_f_eval(self, u, dt, t, u_sol = None):

        if u_sol is None:
            u_sol = self.dtype_u(self.init,val=0.)    

        l = self.lmbda(t)
        z = dt*l[1]
        phi_one = (math.exp(z) - 1.)/z

        u_sol[0] = 0.
        u_sol[1] = phi_one*l[1]*(u[1]-self.yinf[1])

        return u_sol
    
    @property
    def exact(self):

        class tmp:
            def __init__(self,lmbda):
                self.lmbda = lmbda

            def rho_nonstiff(self,y,t,fy):
                return 0.
            
            def rho_stiff(self,y,t,fy):
                return abs(self.lmbda)
            
        return tmp(self.lmbda_cte[0])
    

class debug_exp_runge_kutta_2(ptype):

    dtype_u = my_mesh
    dtype_f = exp_imex_mesh

    def __init__(self,**problem_params):
        self.know_exact = True
        self.nvars = 1
        self.init = self.nvars

        super(debug_exp_runge_kutta_2, self).__init__(self.init)
        self._makeAttributeAndRegister(*problem_params.keys(),localVars=problem_params,readOnly=True)

        self.t0 = 0.
        self.Tend = 0.2
        self.y_ex = lambda t: np.sin(2.*np.pi*t)
        self.lmbda_cte = -5.
        self.lmbda = lambda t:  self.lmbda_cte
        self.f = lambda y,t:  self.lmbda(t)*(y-self.y_ex(t)) + 2.*np.pi*np.cos(2.*np.pi*t) + self.lmbda(t)*(np.cos(y)**2-np.cos(self.y_ex(t))**2)
        self.y0 = self.dtype_u(self.init)
        self.y0[:] = self.y_ex(self.t0)

    def initial_value(self):
        return self.y0
    
    def u_exact(self,t):
        y_ex = self.dtype_u(self.init)
        y_ex[:] = self.y_ex(t)    

        return y_ex

    def compute_errors(self,uh,t):
        u_ex = self.u_exact(t)
        errors_L2 = np.abs(uh-u_ex)
        norms_L2_sol = np.abs(u_ex)
        rel_errors_L2 = errors_L2/norms_L2_sol
        
        print(f"L2-errors: {errors_L2}")
        print(f"Relative L2-errors: {rel_errors_L2}")
        print(f'u_ex = {u_ex}')
        print(f'u_h = {uh}')

    def get_size(self):        
        return self.nvars
    
    def solve_system(self, rhs, factor, u0, t, u_sol = None):                

        return rhs
    
    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None):
    
        if fh is None:
            fh = self.dtype_f(self.init,val=0.)    

        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl = self.eval_f_stiff(u,t,fh.impl)

        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:                                                
            fh.expl = self.eval_f_nonstiff(u,t,fh.expl) 
                
        # evaluate exponential (super stiff) part M^-1*f_exp(u,t)
        if eval_exp:                                                
            fh.exp = self.eval_f_exp(u,t,fh.exp) 
                            
        return fh
    
    def eval_f_nonstiff(self,u,t,fh_nonstiff):

        fh_nonstiff[0] = 0.

        return fh_nonstiff

    def eval_f_stiff(self,u,t,fh_stiff):

        fh_stiff[0] = 0.

        return fh_stiff
    
    def eval_f_exp(self,u,t,fh_exp):
        
        ft = self.f(u,t)
        fh_exp[0] = ft[0]

        return fh_exp
    
    def phi_eval(self, u, dt, t, k, phi = None):

        if phi is None:
            phi = self.dtype_u(self.init,val=0.)    

        l = self.lmbda(t)
        z = dt*l
        phi_k = math.exp(z) 
        fac_k = 1.
        for i in range(1,k+1):            
            phi_k = (phi_k-1./fac_k)/z
            fac_k = fac_k*i
        
        phi[0] = phi_k

        return phi
    
    def lmbda_eval(self, u, t, lmbda = None):

        if lmbda is None:
            lmbda = self.dtype_u(self.init,val=0.)    

        lmbda[0] = self.lmbda(t)
        
        return lmbda
    
    def phi_one_f_eval(self, u, dt, t, u_sol = None):

        if u_sol is None:
            u_sol = self.dtype_u(self.init,val=0.)    

        l = self.lmbda(t)
        z = dt*l
        phi_one = (math.exp(z) - 1.)/z

        u_sol[0] = phi_one*self.eval_f(u,t,eval_impl=False,eval_expl=False,eval_exp=True).exp[0]

        return u_sol
    
    @property
    def exact(self):

        class tmp:
            def __init__(self,lmbda):
                self.lmbda = lmbda

            def rho_nonstiff(self,y,t,fy):
                return 0.
            
            def rho_stiff(self,y,t,fy):
                return abs(self.lmbda)
            
        return tmp(self.lmbda_cte[0])