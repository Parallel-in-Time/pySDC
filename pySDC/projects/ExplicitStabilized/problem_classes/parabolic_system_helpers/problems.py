import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from pySDC.core.Errors import ParameterError

class parabolic_system_problem():
    # REDO doc
    # class defining a problem u_t - div(grad(u)) = f(t)
    # subclasses must either define self.sol_expr (ufl expression for the exact solution)
    # or self.uD and self.u0, two ufl expressions for the Dirichlet data and the initial value
    # self.rhs_expr must be defined in subclasses, it implements f(t).
    # if the exact solution is unknown (i.e. self.sol_expr not implemented), then set self.know_exact=False in subclasses
    
    def __init__(self,dim,n_elems,dom_size=[[0.,0.,0.],[1.,1.,1.]]):
        self.dom_size = dom_size
        self.define_domain(dim,n_elems)
        self.dim = self.domain.topology.dim

        # defined in subclasses
        self.size = None # number of system variables
        self.sp_ind = None # indeces of equations with non zero diffusion

        self.eval_points = np.array([])

        self.know_exact = False
        self.cte_Dirichlet = False # indicates if the Dirichlet boundary condition is constant in time
        self.rhs_or_N_bnd_dep_t = True # indicates if f(t) or the Neumann boundary data depend on t. 
        self.bnd_cond = 'N' # can be D, N or Mixed

        if self.bnd_cond not in ['D','N','Mixed']:
            raise Exception('Boundary condition must be either D,N or Mixed.')
        
    @property
    def uD_expr(self):
        return self.sol
    
    @property
    def g_expr(self):
        return self.g

    @property
    def u0_expr(self):
        self.t.value=self.t0
        return self.sol
    
    @property
    def sol_expr(self):
        return self.sol
    
    @property
    def rhs_expr(self):
        return self.rhs
    
    @property
    def rhs_stiff_expr(self):
        return self.rhs_stiff
        
    @property
    def rhs_nonstiff_expr(self):
        return self.rhs_nonstiff
    
    @property
    def rhs_expl_expr(self):
        return self.rhs_expl
    
    @property
    def lmbda_expr(self):
        return self.lmbda
    
    @property
    def yinf_expr(self):
        return self.yinf
    
    def define_domain_dependent_variables(self,domain,V,dtype_u):
        self.domain = domain
        self.V = V
        self.x = ufl.SpatialCoordinate(self.domain)
        self.n = ufl.FacetNormal(self.domain)
        self.t = fem.Constant(self.domain,0.)
        self.dt = fem.Constant(self.domain,0.)
        self.dtype_u = dtype_u
        self.uh = self.dtype_u(init=self.V,val=0.)            

    def update_time(self,t):
        if self.rhs_or_N_bnd_dep_t:
            self.t.value = t

    def update_dt(self,dt):        
        self.dt.value = dt

    def define_domain(self,dim,n_elems):
        dom_size = self.dom_size
        d = np.asarray(dom_size[1])-np.asarray(dom_size[0])
        max_d = np.max(d)
        n = [n_elems]*dim
        for i in range(len(n)):
            n[i] = int(np.ceil(n[i]*d[i]/max_d))
            
        if dim==1:
            self.domain = mesh.create_interval(comm=MPI.COMM_WORLD, nx=n_elems, points=[dom_size[0][0],dom_size[1][0]])            
        elif dim==2:            
            self.domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, n=n, cell_type=mesh.CellType.triangle, points=[dom_size[0][:dim],dom_size[1][:dim]])
        elif dim==3:
            self.domain = mesh.create_box(comm=MPI.COMM_WORLD, n=n, cell_type=mesh.CellType.tetrahedron, points=[dom_size[0][:dim],dom_size[1][:dim]])
        else:
            raise ParameterError(f"need dim=1,2,3 to instantiate problem, got dim={dim}")
    
    def Dirichlet_boundary(self,x):
        if self.dim == 1:
            if self.bnd_cond=='D':
                return np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0]))
            elif self.bnd_cond=='Mixed':
                return np.isclose(x[0], self.dom_size[0][0])
            else:
                return [False]*len(x[0])
        elif self.dim==2:
            if self.bnd_cond=='D':
                return np.logical_or(np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0])),\
                                     np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1])))
            elif self.bnd_cond=='Mixed':
                return np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0]))
            else:
                return [False]*len(x[0])
        else:
            if self.bnd_cond=='D':
                return np.logical_or(np.logical_or(np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0])),\
                                                   np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1]))),\
                                    np.logical_or(np.isclose(x[2], self.dom_size[0][2]), np.isclose(x[2],self.dom_size[1][2])))
            elif self.bnd_cond=='Mixed':
                return np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0]))
            else:
                return [False]*len(x[0])
    
    def Neumann_boundary(self,x):        
        if self.dim == 1:
            if self.bnd_cond=='D':
                return [False]*len(x[0])
            elif self.bnd_cond=='Mixed':
                return np.isclose(x[0], self.dom_size[1][0])
            else:
                return np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0]))
        elif self.dim==2:
            if self.bnd_cond=='D':
                return [False]*len(x[0])
            elif self.bnd_cond=='Mixed':
                return np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1]))
            else:
                return np.logical_or(np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0])),\
                                     np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1])))
        else:
            if self.bnd_cond=='D':
                return [False]*len(x[0])
            elif self.bnd_cond=='Mixed':
                return np.logical_or(np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1])),\
                                     np.logical_or(np.isclose(x[2], self.dom_size[0][2]), np.isclose(x[2],self.dom_size[1][2])))
            else:
                return np.logical_or(np.logical_or(np.logical_or(np.isclose(x[0], self.dom_size[0][0]), np.isclose(x[0],self.dom_size[1][0])),\
                                                   np.logical_or(np.isclose(x[1], self.dom_size[0][1]), np.isclose(x[1],self.dom_size[1][1]))),\
                                    np.logical_or(np.isclose(x[2], self.dom_size[0][2]), np.isclose(x[2],self.dom_size[1][2])))


    def define_standard_splittings(self):

        # Here we define different splittings of the rhs into stiff, nonstiff and exponential terms
        self.rhs_stiff = dict()
        self.rhs_nonstiff = dict()
        self.rhs_stiff_args = dict()
        self.rhs_nonstiff_args = dict()
        self.rhs_exp_args = dict()
        self.lmbda = dict()
        self.yinf = dict()
        self.diagonal_stiff = dict()
        self.diagonal_nonstiff = dict()

        # here we consider only a stiff and a nonstiff part
        self.rhs_nonstiff['stiff_nonstiff'] = [self.rhs[0], self.rhs[1]] #this is random splitting (i.e. not really in stiff-nonstiff) just for debugging
        self.rhs_stiff['stiff_nonstiff'] = [None, None]              
        self.lmbda['stiff_nonstiff'] = [None, fem.Constant(self.domain,-1.)]
        self.yinf['stiff_nonstiff'] = [None, None]
        self.rhs_nonstiff_args['stiff_nonstiff'] = [0,1] 
        self.rhs_stiff_args['stiff_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['stiff_nonstiff'] = [] 
        self.diagonal_nonstiff['stiff_nonstiff'] = True
        self.diagonal_stiff['stiff_nonstiff'] = False
        
        # here we add (artificially) an exponential term and remove the stiff term
        self.rhs_nonstiff['exp_nonstiff'] = [self.rhs[0], self.rhs[1] + self.sol[1]]
        self.rhs_stiff['exp_nonstiff'] = [None, None]      
        self.lmbda['exp_nonstiff'] = [None, fem.Constant(self.domain,-1.)]
        self.yinf['exp_nonstiff'] = [None, 0.]
        self.rhs_nonstiff_args['exp_nonstiff'] = [0,1] 
        self.rhs_stiff_args['exp_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['exp_nonstiff'] = [1] 
        self.diagonal_nonstiff['exp_nonstiff'] = True
        self.diagonal_stiff['exp_nonstiff'] = False

        # here we consider the three terms
        self.rhs_nonstiff['exp_stiff_nonstiff'] = [self.rhs[0], self.sol[1]]
        self.rhs_stiff['exp_stiff_nonstiff'] = [None, self.rhs[1]]      
        self.lmbda['exp_stiff_nonstiff'] = [None, fem.Constant(self.domain,-1.)]
        self.yinf['exp_stiff_nonstiff'] = [None, 0.]
        self.rhs_nonstiff_args['exp_stiff_nonstiff'] = [0,1] 
        self.rhs_stiff_args['exp_stiff_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['exp_stiff_nonstiff'] = [1] 
        self.diagonal_nonstiff['exp_stiff_nonstiff'] = True
        self.diagonal_stiff['exp_stiff_nonstiff'] = False


class linlin_solution(parabolic_system_problem):
    def __init__(self,dim,n_elems):
        dom_size=[[0.,0.,0.],[1.,1.,1.]]
        super(linlin_solution,self).__init__(dim,n_elems,dom_size)

        self.size = 2
        self.sp_ind = [0,1]

        self.alpha = 3.
        self.beta = 1.
        self.t0 = 0.0
        self.Tend = 1.
        self.diff = [0.5,0.5]        

        self.know_exact = True
        self.cte_Dirichlet = False 
        self.rhs_or_N_bnd_dep_t = True 
        self.bnd_cond = 'N' 

    def define_domain_dependent_variables(self,domain,V,dtype_u):
        super().define_domain_dependent_variables(domain,V,dtype_u)
        self.cte = fem.Constant(self.domain,self.beta)
        if self.dim==1:
            sol = 1. + self.x[0] + self.beta * self.t
        else:            
            sol = 1. + self.x[0] + self.alpha * self.x[1] + self.beta * self.t
        self.sol = [sol,sol]
        self.g = [-ufl.dot(self.diff[i]*ufl.grad(self.sol[i]),self.n) for i in range(2)]
        
        self.rhs = [self.cte,self.cte]        
        
        self.define_standard_splittings()
        

class coscoscos(parabolic_system_problem):
    def __init__(self,dim,n_elems,dom_size=[[0.,0.,0.],[1.,1.,1.]]):        
        super(coscoscos,self).__init__(dim,n_elems,dom_size)

        self.size = 2
        self.sp_ind = [0,1]

        self.fx = 1.
        self.fy = 1.
        self.fz = 1.
        self.ft = 1.
        self.t0 = 0.
        self.Tend = 1.
        self.diff = [0.5,0.5]

        self.know_exact = True
        self.cte_Dirichlet = False # indicates if the Dirichlet boundary condition is constant in time
        self.rhs_or_N_bnd_dep_t = True # indicates if f(t) or the Neumann boundary data depend on t. 
        self.bnd_cond = 'N' # can be D, N or Mixed

    def define_domain_dependent_variables(self,domain,V,dtype_u):
        super().define_domain_dependent_variables(domain,V,dtype_u)
        if self.dim==1:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.ft*self.t)     
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[i]*ufl.pi**2*self.fx**2*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0]) for i in range(self.size)]
        elif self.dim==2:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.ft*self.t)       
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[i]*ufl.pi**2*(self.fx**2+self.fy**2)*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1]) for i in range(self.size)]
        else:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.fz*self.x[2])*ufl.cos(ufl.pi*self.ft*self.t)      
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[i]*ufl.pi**2*(self.fx**2+self.fy**2+self.fz**2)*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.fz*self.x[2]) for i in range(self.size)]
        self.sol = [sol,sol]
        self.rhs = [self.rhs[i]+self.f(self.sol[i])-self.f(self.uh.sub(i)) for i in range(self.size)]  

        self.g = [-ufl.dot(self.diff[i]*ufl.grad(self.sol[i]),self.n) for i in range(self.size)]

        self.define_standard_splittings()
    
    def f(self,u):
        return ufl.cos(u)
    
# Similar to above but the second equation is purely ODE
class coscoscos_pdeode(parabolic_system_problem):
    def __init__(self,dim,n_elems,dom_size=[[0.,0.,0.],[1.,1.,1.]]):        
        super(coscoscos_pdeode,self).__init__(dim,n_elems,dom_size)

        self.size = 2
        self.sp_ind = [0]

        self.fx = 1.
        self.fy = 1.
        self.fz = 1.
        self.ft = 1.
        self.t0 = 0.
        self.Tend = 1.
        self.diff = [0.1,None]

        self.know_exact = True
        self.cte_Dirichlet = False # indicates if the Dirichlet boundary condition is constant in time
        self.rhs_or_N_bnd_dep_t = True # indicates if f(t) or the Neumann boundary data depend on t. 
        self.bnd_cond = 'N' # can be D, N or Mixed        
        
    def define_domain_dependent_variables(self,domain,V,dtype_u):
        super().define_domain_dependent_variables(domain,V,dtype_u)
        if self.dim==1:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.ft*self.t)     
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[0]*ufl.pi**2*self.fx**2*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0]), \
                        (-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])]
        elif self.dim==2:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.ft*self.t)       
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[0]*ufl.pi**2*(self.fx**2+self.fy**2)*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1]),\
                        (-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])]
        else:
            sol = ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.fz*self.x[2])*ufl.cos(ufl.pi*self.ft*self.t)      
            self.rhs = [(-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t)+self.diff[0]*ufl.pi**2*(self.fx**2+self.fy**2+self.fz**2)*ufl.cos(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.fz*self.x[2]), \
                        (-ufl.pi*self.ft*ufl.sin(ufl.pi*self.ft*self.t))*ufl.cos(ufl.pi*self.fx*self.x[0])*ufl.cos(ufl.pi*self.fy*self.x[1])*ufl.cos(ufl.pi*self.fz*self.x[2])]
        self.sol = [sol,sol]
        self.g = [-ufl.dot(self.diff[0]*ufl.grad(self.sol[0]),self.n), None]
        self.rhs = [self.rhs[i]+self.f(self.sol[i])-self.f(self.uh.sub(i)) for i in range(self.size)]        
        
        self.define_standard_splittings()

    def f(self,u):
        return ufl.cos(u)
    
    def define_standard_splittings(self):
        super().define_standard_splittings()

        self.rhs_stiff_args['stiff_nonstiff'] = [0]
        self.rhs_stiff_args['exp_nonstiff'] = [0]
        self.rhs_stiff_args['exp_stiff_nonstiff'] = [0,1]

        # here we consider the three terms
        self.rhs_nonstiff['exp_nonstiff_dir_sum'] = [self.rhs[0], None]
        self.rhs_stiff['exp_nonstiff_dir_sum'] = [None, None]      
        self.lmbda['exp_nonstiff_dir_sum'] = [None, fem.Constant(self.domain,-1.)]
        self.yinf['exp_nonstiff_dir_sum'] = [None, self.rhs[1]+self.sol[1]]
        self.rhs_nonstiff_args['exp_nonstiff_dir_sum'] = [0] 
        self.rhs_stiff_args['exp_nonstiff_dir_sum'] = [0]
        self.rhs_exp_args['exp_nonstiff_dir_sum'] = [1] 
        self.diagonal_nonstiff['exp_nonstiff_dir_sum'] = True
        self.diagonal_stiff['exp_nonstiff_dir_sum'] = False

        # here we consider the three terms
        self.rhs_nonstiff['exp_stiff_nonstiff_dir_sum'] = [self.rhs[0], None]
        self.rhs_stiff['exp_stiff_nonstiff_dir_sum'] = [None, None]      
        self.lmbda['exp_stiff_nonstiff_dir_sum'] = [None, fem.Constant(self.domain,-1.)]
        self.yinf['exp_stiff_nonstiff_dir_sum'] = [None, self.rhs[1]+self.sol[1]]
        self.rhs_nonstiff_args['exp_stiff_nonstiff_dir_sum'] = [0] 
        self.rhs_stiff_args['exp_stiff_nonstiff_dir_sum'] = [0]
        self.rhs_exp_args['exp_stiff_nonstiff_dir_sum'] = [1] 
        self.diagonal_nonstiff['exp_stiff_nonstiff_dir_sum'] = True
        self.diagonal_stiff['exp_stiff_nonstiff_dir_sum'] = False

class brusselator(parabolic_system_problem):
    def __init__(self,dim,n_elems):
        dom_size=[[0.,0.,0.],[1.,1.,1.]]
        super(brusselator,self).__init__(dim,n_elems,dom_size)

        self.size = 2
        self.sp_ind = [0,1]

        self.alpha = 0.1
        self.c1 = 1.
        self.c2 = 3.4
        self.c3 = 4.4
        self.t0 = 0.0
        self.Tend = 1.
        self.diff = [self.alpha,self.alpha]

        self.know_exact = False
        self.cte_Dirichlet = False 
        self.rhs_or_N_bnd_dep_t = True 
        self.bnd_cond = 'N' 

    def define_domain_dependent_variables(self,domain,V,dtype_u):
        super().define_domain_dependent_variables(domain,V,dtype_u)    
        self.zero = fem.Constant(self.domain,0.)
        self.g = [self.zero]*self.size        
        self.u0 = [22.*self.x[1]*ufl.elem_pow(1.-self.x[1],1.5),27.*self.x[0]*ufl.elem_pow(1.-self.x[0],1.5)]        
        self.rhs = [1.0+self.uh.sub(0)**2*self.uh.sub(1)-4.4*self.uh.sub(0)+5.*ufl.exp(-((self.x[0]-0.3)**2+(self.x[1]-0.6)**2)/0.005), 3.4*self.uh.sub(0)-self.uh.sub(0)**2*self.uh.sub(1)]
        self.define_splittings()

    def define_splittings(self):
        
        # Here we define different splittings of the rhs into stiff, nonstiff and exponential terms
        self.rhs_stiff = dict()
        self.rhs_nonstiff = dict()
        self.yinf = dict()
        self.lmbda = dict()
        self.rhs_nonstiff_args = dict()
        self.rhs_stiff_args = dict()
        self.rhs_exp_args = dict()
        self.diagonal_stiff = dict()
        self.diagonal_nonstiff = dict()

        # here we consider only a stiff and a nonstiff part
        self.rhs_nonstiff['stiff_nonstiff'] = self.rhs
        self.rhs_stiff['stiff_nonstiff'] = [None,None]      
        self.rhs_stiff_args['stiff_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.lmbda['stiff_nonstiff'] = [None,None]
        self.yinf['stiff_nonstiff'] = [None,None]
        self.rhs_nonstiff_args['stiff_nonstiff'] = [0,1] 
        self.rhs_stiff_args['stiff_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['stiff_nonstiff'] = [] 
        self.diagonal_nonstiff['stiff_nonstiff'] = False
        self.diagonal_stiff['stiff_nonstiff'] = False

        self.rhs_nonstiff['exp_nonstiff'] = [1.0+self.uh.sub(0)**2*self.uh.sub(1)+5.*ufl.exp(-((self.x[0]-0.3)**2+(self.x[1]-0.6)**2)/0.005), 3.4*self.uh.sub(0)-self.uh.sub(0)**2*self.uh.sub(1)]
        self.rhs_stiff['exp_nonstiff'] = [None,None]      
        self.lmbda['exp_nonstiff'] = [-4.4,None]  
        self.yinf['exp_nonstiff'] = [0.,None]
        self.rhs_nonstiff_args['exp_nonstiff'] = [0,1] 
        self.rhs_stiff_args['exp_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['exp_nonstiff'] = [0] 
        self.diagonal_nonstiff['exp_nonstiff'] = False
        self.diagonal_stiff['exp_nonstiff'] = False

        # here we consider the three terms
        self.rhs_nonstiff['exp_stiff_nonstiff'] = [1.0+self.uh.sub(0)**2*self.uh.sub(1)+5.*ufl.exp(-((self.x[0]-0.3)**2+(self.x[1]-0.6)**2)/0.005), 3.4*self.uh.sub(0)-self.uh.sub(0)**2*self.uh.sub(1)]
        self.rhs_stiff['exp_stiff_nonstiff'] = [None,None]      
        self.lmbda['exp_stiff_nonstiff'] = [-4.4,None]  
        self.yinf['exp_stiff_nonstiff'] = [0.,None]
        self.rhs_nonstiff_args['exp_stiff_nonstiff'] = [0,1] 
        self.rhs_stiff_args['exp_stiff_nonstiff'] = [0,1]  # do not forget the laplacian!! it acts on both variables
        self.rhs_exp_args['exp_stiff_nonstiff'] = [0] 
        self.diagonal_nonstiff['exp_stiff_nonstiff'] = False
        self.diagonal_stiff['exp_stiff_nonstiff'] = False

    @property
    def u0_expr(self):        
        return self.u0
