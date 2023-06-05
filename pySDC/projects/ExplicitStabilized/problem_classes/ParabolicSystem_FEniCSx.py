import numpy as np
from dolfinx import mesh, fem, io, nls, log, geometry
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh import fenicsx_mesh, rhs_fenicsx_mesh, exp_rhs_fenicsx_mesh
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

class parabolic_system(ptype):

    dtype_u = fenicsx_mesh
    dtype_f = fenicsx_mesh

    def __init__(self, **problem_params):
        
        # these parameters will be used later, so assert their existence
        essential_keys = ['n_elems', 'dim', 'family', 'order', 'refinements','exact_solution_class','enable_output']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)       
        if problem_params['enable_output']:
            essential_keys = ['output_folder','output_file_name']
            for key in essential_keys:
                if key not in problem_params:
                    msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                    raise ParameterError(msg)               

        n_elems = problem_params['n_elems']
        dim = problem_params['dim']
        n_ref = problem_params['refinements']
        
        self.exact = problem_params['exact_solution_class'](dim,n_elems)  
        self.t0 = self.exact.t0 # Start time
        self.Tend = self.exact.Tend  # End time  
        self.know_exact = self.exact.know_exact      

        self.domain = self.exact.domain
        self.refine_domain(n_ref)

        FE = ufl.FiniteElement(problem_params['family'], self.domain.ufl_cell(), problem_params['order'])
        self.V = fem.FunctionSpace(self.domain, ufl.MixedElement([FE]*self.exact.size))
        self.exact.define_domain_dependent_variables(self.domain,self.V,self.dtype_u)

        self.set_solver_options(problem_params)

        # invoke super init
        super(parabolic_system, self).__init__(self.V)
        self._makeAttributeAndRegister(*problem_params.keys(),localVars=problem_params,readOnly=True)

        if self.mass_lumping and (self.family!='CG' or self.order>1):
            raise Exception("You have specified mass_lumping=True but for order>1 or family!='CG'.")
        
        if self.mass_lumping and self.exact.bnd_cond!='N':
            raise Exception('Cannot apply lifting of the mass lumped form. Try without mass lumping or without Dirichlet boundary conditions.')

        self.define_boundary_conditions()  
        self.define_variational_forms()
        self.assemble_vec_mat()
        self.define_mass_solver()

        # save in xdmf format, which requires saving the mesh only once
        if self.enable_output:
            self.xdmf = io.XDMFFile(self.domain.comm, self.output_folder+self.output_file_name+".xdmf", "w")
            self.xdmf.write_mesh(self.domain)
        
    def refine_domain(self,n_ref):
        for _ in range(n_ref):
            self.domain.topology.create_entities(1)
            self.domain = mesh.refine(self.domain)

    def set_solver_options(self,params):
        # we suppose that the problem is symmetric
        # first set the default options
        if params['dim']<=2:
            def_solver_ksp = 'preonly'
            def_solver_pc = 'cholesky'
        else:
            def_solver_ksp = 'cg'
            def_solver_pc = 'hypre'
       
        if 'solver_ksp' not in params:
            params['solver_ksp'] = def_solver_ksp
        if 'solver_pc' not in params:
            params['solver_pc'] = def_solver_pc

    def define_boundary_conditions(self):    
        self.uD = fem.Function(self.V)
        if self.exact.bnd_cond!='N':
            self.get_DirBC(self.uD,self.t0)
        
        self.find_Dirichlet_dofs()

        self.bc = []
        for i in self.exact.sp_ind:
            self.bc.append(fem.dirichletbc(self.uD.sub(i), self.dofs_Dbc[i], self.V.sub(i)))

    def get_DirBC(self,u_D,t):        
        self.exact.t.value = t        
        for i in self.exact.sp_ind:            
            u_D.sub(i).interpolate(fem.Expression(self.exact.uD_expr[i],self.V.sub(i).element.interpolation_points()))   # find way to interpolate only on dirichlet nodes self.dofs_Dbc

    def find_Dirichlet_dofs(self):
        boundaries = [(1, self.exact.Dirichlet_boundary),
                      (2, self.exact.Neumann_boundary)]
        facet_indices, facet_markers = [], []
        fdim = self.domain.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = mesh.locate_entities(self.domain, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(self.domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=facet_tag)

        facets = facet_tag.find(1)
        self.dofs_Dbc = [None]*self.exact.size
        for i in self.exact.sp_ind:
            self.dofs_Dbc[i] = fem.locate_dofs_topological((self.V.sub(i),self.V.sub(i)), fdim, facets)            

    def define_variational_forms(self):
        
        u, v = ufl.TrialFunctions(self.V), ufl.TestFunctions(self.V)

        self.mass = 0.
        for i in range(self.exact.size):
            self.mass  += u[i] * v[i] * ufl.dx

        self.diff = 0.
        self.F_bnd = 0.
        
        if self.family=='CG':
            for i in self.exact.sp_ind:
                self.diff += ufl.dot(self.exact.diff[i]*ufl.grad(u[i]), ufl.grad(v[i])) * ufl.dx 
                self.F_bnd += - self.exact.g_expr[i]*v[i]*self.ds(2)                 
        elif self.family=='DG':            
            n = ufl.FacetNormal(self.domain)
            h = ufl.FacetArea(self.domain)
            dim = self.dim
            p = self.order
            beta0 = 1./(dim-1)
            if dim==2:
                eta = fem.Constant(self.domain, ScalarType(1.5))*p*(p+1)
            elif dim==3:
                eta = fem.Constant(self.domain, ScalarType(15.))*p*(p+2)*(h**0.5)            
            
            for i in self.exact.sp_ind:
                diff_tens = self.exact.diff[i]
                delta = ufl.dot(n,diff_tens*n)
                om_p = delta('-')/(delta('+')+delta('-'))
                om_m = delta('+')/(delta('+')+delta('-'))
                gamma = 2.*delta('+')*delta('-')/(delta('-')+delta('+'))                
                def avg_o(w):                                
                    return om_p*w('+')+om_m*w('-')
                
                self.diff += ufl.inner(diff_tens*ufl.grad(u[i]), ufl.grad(v[i]))*ufl.dx \
                            - ufl.inner(avg_o(diff_tens*ufl.grad(v[i])), ufl.jump(u[i], n))*ufl.dS \
                            - ufl.inner(ufl.jump(v[i], n), avg_o(diff_tens*ufl.grad(u[i])))*ufl.dS \
                            + eta/(h**beta0)*gamma*ufl.inner(ufl.jump(v[i], n), ufl.jump(u[i], n))*ufl.dS \
                            - ufl.inner(diff_tens*ufl.grad(v[i]), u[i]*n)*self.ds(1) \
                            - ufl.inner(v[i]*n, diff_tens*ufl.grad(u[i]))*self.ds(1) \
                            + (eta/(h**beta0))*delta*v[i]*u[i]*self.ds(1)
                self.F_bnd += - ufl.inner(diff_tens*ufl.grad(v[i]), self.exact.uD_expr[i]*n)*self.ds(1) + (eta/(h**beta0))*(delta)*self.exact.uD_expr[i]*v[i]*self.ds(1) - self.exact.g_expr[i]*v[i]*self.ds(2)                  
        else:
            raise ParameterError("problem_params['family'] must be either 'CG' or 'DG'. DG to be implemented")

        self.F = self.F_bnd
        for i in range(self.exact.size):
            self.F += self.exact.rhs_expr[i]*v[i]*ufl.dx

        self.mass_form = fem.form(self.mass)
        self.diff_form = fem.form(self.diff)
        self.f_bnd_form = fem.form(self.F_bnd)
        self.f_form = fem.form(self.F)        
        
        self.rhs_expr_interp = [None]*self.exact.size
        for i in range(self.exact.size):
            self.rhs_expr_interp[i] = fem.Expression(self.exact.rhs_expr[i],self.V.sub(i).element.interpolation_points())
        self.interp_f = fenicsx_mesh(self.init)

    def assemble_vec_mat(self):
        self.b = fem.petsc.create_vector(self.f_form)

        if self.family=='CG' and self.exact.bnd_cond!='N':
            self.M = fem.petsc.assemble_matrix(self.mass_form,bcs=self.bc)
            self.K = fem.petsc.assemble_matrix(self.diff_form,bcs=self.bc)        
        else:
            self.M = fem.petsc.assemble_matrix(self.mass_form)            
            self.K = fem.petsc.assemble_matrix(self.diff_form)                                

        self.M.assemble()                    
        self.K.assemble()

        if self.mass_lumping:            
            for i in range(self.exact.size):                
                self.uD.sub(i).interpolate(lambda x: 1.0 + 0.*x[0])            
            mass_lumped = ufl.action(self.mass, self.uD)
            self.Ml = fem.petsc.assemble_matrix(self.mass_form)
            self.Ml.zeroEntries()
            self.ml = fem.petsc.create_vector(fem.form(mass_lumped))
            with self.ml.localForm() as m_loc:
                m_loc.set(0)
            fem.petsc.assemble_vector(self.ml, fem.form(mass_lumped))
            self.ml.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            self.ml.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.set_bc(self.ml, bcs=self.bc) # self.uD is ==1 so will set 1 at boundary nodes
            self.Ml.setDiagonal(self.ml)
            if self.exact.bnd_cond!='N':
                self.get_DirBC(self.uD,self.t0) #restore uD to its value            

    def define_mass_solver(self):
        self.solver_M = PETSc.KSP().create(self.domain.comm)
        if self.mass_lumping:
            self.solver_M.setOperators(self.Ml)
        else:
            self.solver_M.setOperators(self.M)
        self.solver_M.setType(self.solver_ksp)
        self.solver_M.getPC().setType(self.solver_pc)       

    def mass_mult_add(self,x,y,z):
        # performs z = M*x+y
        if self.mass_lumping:
            self.Ml.multAdd(x,y,z)          
        else:
            self.M.multAdd(x,y,z)   

    def invert_mass_matrix(self, x, y):        
        # solves y = M*y
        if self.mass_lumping:
            y.pointwiseDivide(x,self.ml)
        else:
            self.solver_M.solve(x, y)      

    def write_solution(self,uh,t):        
        if self.enable_output:            
            uh.values.name = "u"         
            for i in range(self.exact.size):   
                self.xdmf.write_function(uh.sub(i), t)

    def initial_value(self):
        u0 = self.dtype_u(self.init,val=0.)
        u0.values.name = "u"     
        for i in range(self.exact.size):
            u0.values.sub(i).interpolate(fem.Expression(self.exact.u0_expr[i],self.V.sub(i).element.interpolation_points()))       
        return u0

    def u_exact(self,t):
        u_ex = self.dtype_u(init=self.init,val=0.)
        self.get_exact(u_ex,t)
        return u_ex
    
    def get_exact(self,u_ex,t):
        if not self.know_exact:
            raise Exception('Exact solution unknown.')
        self.exact.t.value = t
        u_ex.values.name = "u"     
        for i in range(self.exact.size):
            u_ex.values.sub(i).interpolate(fem.Expression(self.exact.sol_expr[i],self.V.sub(i).element.interpolation_points()))             

    def compute_errors(self,uh,t):
        # Compute L2 error and error at nodes
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_ex = self.u_exact(t)
        errors_L2 = []
        norms_L2_sol = []
        rel_errors_L2 = []
        for i in range(self.exact.size):
            errors_L2.append( np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((uh.sub(i) - u_ex.values.sub(i))**2 * ufl.dx)), op=MPI.SUM)) )
            norms_L2_sol.append( np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(u_ex.values.sub(i)**2 * ufl.dx)), op=MPI.SUM)) )
            rel_errors_L2.append(errors_L2[-1]/norms_L2_sol[-1])
        if self.domain.comm.rank == 0:
            print(f"L2-errors: {errors_L2}")
            print(f"Relative L2-errors: {rel_errors_L2}")

    def get_size(self):        
        return self.uD.sub(0).vector.getSize()
    
    def eval_on_points(self,u):

        if not hasattr(self,'points_on_proc'):
            bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)
            self.cells = []
            self.points_on_proc = []
            # Find cells whose bounding-box collide with the the points
            cell_candidates = geometry.compute_collisions(bb_tree, self.exact.eval_points)
            # Choose one of the cells that contains the point
            colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, self.exact.eval_points)
            for i, point in enumerate(self.exact.eval_points):
                if len(colliding_cells.links(i))>0:
                    self.points_on_proc.append(point)
                    self.cells.append(colliding_cells.links(i)[0])
            self.points_on_proc = np.array(self.points_on_proc, dtype=np.float64)
            self.loc_glob_map = []
            for i in range(self.points_on_proc.shape[0]):
                p = self.points_on_proc[i,:]
                for j in range(self.exact.eval_points.shape[0]):
                    q = self.exact.eval_points[j,:]
                    if np.linalg.norm(p-q)<np.max([np.linalg.norm(p),np.linalg.norm(q)])*1e-5:
                        self.loc_glob_map.append(j)
                        break

        # eval only on u[0]
        u_val_loc =  u.values.sub(0).eval(self.points_on_proc, self.cells)
        u_val_loc = np.reshape(u_val_loc,(self.points_on_proc.shape[0],1))
        data = self.loc_glob_map, u_val_loc
        data = self.domain.comm.gather(data, root=0)
        if self.domain.comm.rank==0:
            u_val = np.zeros((self.exact.eval_points.shape[0],1))
            for i in range(self.domain.comm.size):
                loc_glob_map, u_val_loc = data[i]
                for j,u in zip(loc_glob_map, u_val_loc):
                    u_val[j] = u
        else:
            assert data==None
            u_val = None

        return u_val

    def eval_f(self, u, t, fh=None):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        self.exact.update_time(t)

        if fh is None:
            fh = self.dtype_f(init=self.V,val=0.)    

        # update the Dirichlet boundary conditions. Needed to apply lifting and as well to evaluate the
        # f_form in the case where boundary conditions are applied via penalty (e.g. DG)
        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)
        
        # update uh, which may be needed to compute the f_form (in the case where f(u,t) depends on u)
        if hasattr(self.exact,'uh'):
            self.exact.uh.copy(u)
            self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)            
                                                                          
        # evaluate f_form  
        with self.b.localForm() as loc_b:
            loc_b.set(0)
                
        if self.f_interp:
            fem.petsc.assemble_vector(self.b, self.f_bnd_form)
        else:
            fem.petsc.assemble_vector(self.b, self.f_form)
                
        # apply stiffness matrix
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.apply_lifting(self.b, [self.diff_form], bcs=[self.bc])
        
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        self.K.multAdd(-u.values.vector,self.b,self.b) 

        self.invert_mass_matrix(self.b, fh.values.vector)

        if self.f_interp:         
            for i in range(self.exact.size):       
                self.interp_f.values.sub(i).interpolate(self.rhs_expr_interp[i])
            fh += self.interp_f

        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh.values.vector, bcs=self.bc, scale=0.)
                    
        return fh

class parabolic_system_imex(parabolic_system):

    dtype_u = fenicsx_mesh
    dtype_f = rhs_fenicsx_mesh

    def __init__(self, **problem_params):        
        super(parabolic_system_imex,self).__init__(**problem_params)      
        
        self.prev_factor = -1.        
        self.solver= PETSc.KSP().create(self.domain.comm)
        self.solver.setType(self.solver_ksp)
        self.solver.getPC().setType(self.solver_pc)

    def eval_f(self, u, t, fh=None):
        """
        Evaluates F(u,t) = M^-1*( f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        self.exact.update_time(t)

        if fh is None:
            fh = self.dtype_f(init=self.V,val=0.)    

        # update the Dirichlet boundary conditions. Needed to apply lifting and as well to evaluate the
        # f_form in the case where boundary conditions are applied via penalty (e.g. DG)
        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)
        
        # update uh, which may be needed to compute the f_form (in the case where f(u,t) depends on u)
        if hasattr(self.exact,'uh'):
            self.exact.uh.copy(u)
            self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)            
        
        # evaluate explicit (non stiff) part M^-1*f(u,t)
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        if self.f_interp:
            fem.petsc.assemble_vector(self.b, self.f_bnd_form)
        else:
            fem.petsc.assemble_vector(self.b, self.f_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
        self.invert_mass_matrix(self.b, fh.expl.values.vector)
        if self.f_interp:     
            for i in range(self.exact.size):
                self.interp_f.values.sub(i).interpolate(self.rhs_expr_interp[i])             
            fh.expl += self.interp_f
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh.expl.values.vector, bcs=self.bc, scale=0.)
                
        # evaluate implicit (stiff) part M^1*A*u
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.apply_lifting(self.b, [self.diff_form], bcs=[self.bc])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
        self.K.multAdd(-u.values.vector,self.b,self.b)                 
        self.invert_mass_matrix(self.b, fh.impl.values.vector)        
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh.impl.values.vector, bcs=self.bc, scale=0.)
                            
        return fh
    
    def solve_system(self, rhs, factor, u0, t, u_sol = None):        

        self.exact.update_time(t)

        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)

        if u_sol is None:
            u_sol = self.dtype_u(self.V)                

        with self.b.localForm() as loc_b:
            loc_b.set(0)     

        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.apply_lifting(self.b, [self.diff_form], bcs=[self.bc])
            with self.b.localForm() as loc_b:
                loc_b *= factor
        
        self.mass_mult_add(rhs.values.vector,self.b,self.b)  
        if self.family=='CG' and self.exact.bnd_cond!='N':
            rhs.values.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.apply_lifting(self.b, [self.mass_form], bcs=[self.bc], x0=[rhs.values.vector])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, bcs=self.bc, scale=1.+factor)                
        
        if factor==0.:
            self.invert_mass_matrix(self.b, u_sol.values.vector)
        else:
            if abs(factor-self.prev_factor)>1e-8*factor:
                self.prev_factor = factor
                if self.mass_lumping:
                    self.solver.setOperators(self.Ml+factor*self.K)
                else:
                    self.solver.setOperators(self.M+factor*self.K)            

            self.solver.solve(self.b, u_sol.values.vector)
            # print(f'Solver converged in {self.solver.getIterationNumber()} iterations.')

        return u_sol
    

# class parabolic_system_imexexp(parabolic_system_imex):
#     def __init__(self, problem_params, dtype_f=rhs_fenicsx_mesh):        
#         super(parabolic_system_imexexp,self).__init__(problem_params,dtype_f)                 

#     def define_variational_forms(self):        
#         super().define_variational_forms()

#         v = ufl.TestFunctions(self.V)

#         splitting = 'exp_nonstiff'
#         self.u_exp_expr = self.exact.u_exp_expr[splitting]
#         self.rhs_expl_expr = self.exact.rhs_nonstiff_expr[splitting]
    
#         self.u_exp = 0.
#         self.u_exp_indexes = [False]*self.exact.size
#         self.u_exp_expr_interp = [None]*self.exact.size
#         for i in range(self.exact.size):            
#             if self.u_exp_expr[i] is not None:
#                 self.u_exp_expr_interp[i] = fem.Expression(self.u_exp_expr[i],self.V.sub(i).element.interpolation_points())
#                 self.u_exp += self.u_exp_expr[i]*v[i]*ufl.dx                                
#                 self.u_exp_indexes[i] = True
#             else:
#                 self.u_exp += self.exact.uh.sub(i)*v[i]*ufl.dx       
#         self.u_exp_form = fem.form(self.u_exp)

#         self.F_expl = self.F_bnd
#         self.expl_indexes = [False]*self.exact.size
#         self.rhs_expl_expr_interp = [None]*self.exact.size
#         for i in range(self.exact.size):
#             if self.rhs_expl_expr[i] is not None:
#                 self.rhs_expl_expr_interp[i] = fem.Expression(self.rhs_expl_expr[i],self.V.sub(i).element.interpolation_points())   
#                 self.F_expl += self.rhs_expl_expr[i]*v[i]*ufl.dx
#                 self.expl_indexes[i] = True           
#         self.f_expl_form = fem.form(self.F_expl)

#     def eval_f(self, u, t, fh=None):
#         raise Exception("Shouldn't call parabolic_system_imexexp.eval_f.")
    
#     def explicit_exponential_step(self, u, factor, t, u_sol = None):

#         self.exact.update_time(t)
        
#         if u_sol is None:
#             u_sol = self.dtype_u(init=self.V,val=0.)    

#         if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
#             self.get_DirBC(self.uD,t)
        
#         # update uh, which may be needed to compute the f forms (in the case where f(u,t) depends on u)
#         if hasattr(self.exact,'uh'):
#             self.exact.uh.copy(u)
#             self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    
        
#         with self.b.localForm() as loc_b:
#             loc_b.set(0)

#         if self.f_interp:
#             fem.petsc.assemble_vector(self.b, self.f_bnd_form)
#         else:
#             fem.petsc.assemble_vector(self.b, self.f_expl_form)     

#         self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
#         self.invert_mass_matrix(self.b, u_sol.values.vector)
    
#         if self.f_interp:
#             self.interp_f.zero()
#             for i in range(self.exact.size):
#                 if self.expl_indexes[i]:
#                     self.interp_f.values.sub(i).interpolate(self.rhs_expl_expr_interp[i])             
#             u_sol += self.interp_f
        
#         u_sol *= factor                
#         u_sol += u

#         if self.family=='CG' and self.exact.bnd_cond!='N':
#             fem.petsc.set_bc(u_sol.values.vector, bcs=self.bc, scale=1.)

#         # compute u_sol = u_inf + exp(lambda*dt)*(u_sol-u_inf)
#         self.exact.update_dt(factor) # for computing the exponentials
#         if hasattr(self.exact,'uh'):
#             self.exact.uh.copy(u_sol)
#             self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    

#         if self.f_interp:
#             for i in range(self.exact.size):
#                 if self.u_exp_indexes[i]:
#                     u_sol.values.sub(i).interpolate(self.u_exp_expr_interp[i])           
#         else:
#             with self.b.localForm() as loc_b:
#                 loc_b.set(0)
#             fem.petsc.assemble_vector(self.b, self.u_exp_form)   
#             self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
#             self.invert_mass_matrix(self.b, u_sol.values.vector)                  
#         if self.family=='CG' and self.exact.bnd_cond!='N' and i in self.exact.sp_ind:
#             fem.petsc.set_bc(u_sol.values.vector, bcs=self.bc, scale=1.)

#         return u_sol


# class parabolic_system_multirate(parabolic_system):

#     dtype_u = fenicsx_mesh
#     dtype_f = rhs_fenicsx_mesh

#     def __init__(self, **problem_params):        
#         super(parabolic_system_multirate,self).__init__(**problem_params)                 

#     def define_variational_forms(self):        
#         super().define_variational_forms()

#         v = ufl.TestFunctions(self.V)

#         splitting = 'stiff_nonstiff'
#         self.u_exp_expr = self.exact.u_exp_expr[splitting]
#         self.rhs_exp_expr = self.exact.rhs_exp_expr[splitting]
#         self.rhs_stiff_expr = self.exact.rhs_stiff_expr[splitting]
#         self.rhs_nonstiff_expr = self.exact.rhs_nonstiff_expr[splitting]
#         self.rhs_stiff_args = self.exact.rhs_stiff_args[splitting]
    
#         self.F_stiff = 0.
#         self.stiff_indexes = [False]*self.exact.size
#         self.rhs_stiff_expr_interp = [None]*self.exact.size
#         for i in range(self.exact.size):            
#             if self.rhs_stiff_expr[i] is not None:
#                 self.rhs_stiff_expr_interp[i] = fem.Expression(self.rhs_stiff_expr[i],self.V.sub(i).element.interpolation_points())
#                 self.F_stiff += self.rhs_stiff_expr[i]*v[i]*ufl.dx                                
#                 self.stiff_indexes[i] = True
#         self.f_stiff_form = fem.form(self.F_stiff)

#         self.F_nonstiff = self.F_bnd
#         self.nonstiff_indexes = [False]*self.exact.size
#         self.rhs_nonstiff_expr_interp = [None]*self.exact.size
#         for i in range(self.exact.size):
#             if self.rhs_nonstiff_expr[i] is not None:
#                 self.rhs_nonstiff_expr_interp[i] = fem.Expression(self.rhs_nonstiff_expr[i],self.V.sub(i).element.interpolation_points())   
#                 self.F_nonstiff += self.rhs_nonstiff_expr[i]*v[i]*ufl.dx
#                 self.nonstiff_indexes[i] = True           
#         self.f_nonstiff_form = fem.form(self.F_nonstiff)

#     def eval_f(self, u, t, eval_impl=True, eval_expl=True, fh=None):
#         """
#         Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

#         Returns:
#             dtype_u: solution as mesh
#         """

#         self.exact.update_time(t)

#         if fh is None:
#             fh = self.dtype_f(init=self.V,val=0.)    

#         # update the Dirichlet boundary conditions. Needed to apply lifting and as well to evaluate the
#         # f_form in the case where boundary conditions are applied via penalty (e.g. DG)
#         if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
#             self.get_DirBC(self.uD,t)
        
#         # update uh, which may be needed to compute the f_form (in the case where f(u,t) depends on u)
#         if hasattr(self.exact,'uh'):
#             self.exact.uh.copy(u)
#             self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)       
        
#         # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
#         if eval_expl:                                                
#             fh.expl = self.eval_f_nonstiff(u,t,fh.expl) 
                
#         # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
#         if eval_impl:
#             fh.impl = self.eval_f_stiff(u,t,fh.impl)
                            
#         return fh
    
#     def eval_f_nonstiff(self,u,t,fh_nonstiff):

#         with self.b.localForm() as loc_b:
#             loc_b.set(0)
#         if self.f_interp:
#             fem.petsc.assemble_vector(self.b, self.f_bnd_form)
#         else:
#             fem.petsc.assemble_vector(self.b, self.f_nonstiff_form)     
#         self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
#         self.invert_mass_matrix(self.b, fh_nonstiff.values.vector)

#         if self.f_interp:# and any(self.nonstiff_indexes):
#             self.interp_f.zero()
#             for i in range(self.exact.size):
#                 if self.nonstiff_indexes[i]:
#                     self.interp_f.values.sub(i).interpolate(self.rhs_nonstiff_expr_interp[i])             
#                 # else:
#                 #     self.interp_f.values.sub(i).collapse().x.array[:] = 0.
#             fh_nonstiff += self.interp_f

#         if self.family=='CG' and self.exact.bnd_cond!='N':
#             fem.petsc.set_bc(fh_nonstiff.values.vector, [self.bc], scale=0.)

#         return fh_nonstiff

#     def eval_f_stiff(self,u,t,fh_stiff):

#         with self.b.localForm() as loc_b:
#             loc_b.set(0)
#         changed_b = False
#         if not self.f_interp and any(self.stiff_indexes):            
#             fem.petsc.assemble_vector(self.b, self.f_stiff_form)
#             changed_b = True
#         if self.family=='CG' and self.exact.bnd_cond!='N':
#             fem.petsc.apply_lifting(self.b, [self.diff_form], [[self.bc]])
#             changed_b = True
#         if changed_b:
#             self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
        
#         self.K.multAdd(-u.values.vector,self.b,self.b)             
#         self.invert_mass_matrix(self.b, fh_stiff.values.vector)
        
#         if self.f_interp and any(self.stiff_indexes):
#             self.interp_f.zero()
#             for i in range(self.exact.size):
#                 if self.stiff_indexes[i]:
#                     self.interp_f.values.sub(i).interpolate(self.rhs_stiff_expr_interp[i])             
#                 # else:
#                 #     self.interp_f.values.sub(i).collapse().x.array[:] = 0.
#             fh_stiff += self.interp_f        

#         if self.family=='CG' and self.exact.bnd_cond!='N':
#             fem.petsc.set_bc(fh_stiff.values.vector, [self.bc], scale=0.)

#         return fh_stiff


class parabolic_system_exp_expl_impl(parabolic_system_imex):

    dtype_u = fenicsx_mesh
    dtype_f = exp_rhs_fenicsx_mesh

    def __init__(self, **problem_params):        
        super(parabolic_system_exp_expl_impl,self).__init__(**problem_params)

    def define_variational_forms(self):        
        super().define_variational_forms()

        v = ufl.TestFunctions(self.V)

        splitting = self.splitting
        self.lmbda_expr = self.exact.lmbda_expr[splitting]
        self.yinf_expr = self.exact.yinf_expr[splitting]
        self.rhs_stiff_expr = self.exact.rhs_stiff_expr[splitting]
        self.rhs_nonstiff_expr = self.exact.rhs_nonstiff_expr[splitting]
        self.rhs_nonstiff_args = self.exact.rhs_nonstiff_args[splitting]
        self.rhs_stiff_args = self.exact.rhs_stiff_args[splitting]
        self.rhs_exp_args = self.exact.rhs_exp_args[splitting]
    
        self.phi_max = 5
        self.phi_expr_interp = [None]*self.exact.size
        self.lmbda_expr_interp = [None]*self.exact.size
        self.phi_one_f = 0.  
        self.F_exp = 0.              
        self.phi_one_f_expr_interp = [None]*self.exact.size
        self.rhs_exp_expr_interp = [None]*self.exact.size
        self.exp_indexes = [False]*self.exact.size
        for i in range(self.exact.size):            
            if self.lmbda_expr[i] is not None:
                self.phi_one_f_expr_interp[i] = fem.Expression(\
                                                ((ufl.exp(self.lmbda_expr[i]*self.exact.dt)-1.)/(self.exact.dt))*(self.exact.uh.sub(i)-self.yinf_expr[i]),\
                                                self.V.sub(i).element.interpolation_points())
                self.phi_one_f += ((ufl.exp(self.lmbda_expr[i]*self.exact.dt)-1.)/(self.exact.dt))*(self.exact.uh.sub(i)-self.yinf_expr[i])*v[i]*ufl.dx    

                self.phi_expr_interp[i] = []
                phi_k = ufl.exp(self.lmbda_expr[i]*self.exact.dt) # phi_0
                k_fac = 1. # 0!
                self.phi_expr_interp[i].append(fem.Expression(phi_k,self.V.sub(i).element.interpolation_points()))
                for k in range(1,self.phi_max+1):                    
                    phi_k = (phi_k-1./k_fac)/(self.lmbda_expr[i]*self.exact.dt)
                    self.phi_expr_interp[i].append(fem.Expression(phi_k,self.V.sub(i).element.interpolation_points()))
                    k_fac = k_fac*k
    
                self.lmbda_expr_interp[i] = fem.Expression(self.lmbda_expr[i], self.V.sub(i).element.interpolation_points())                                
                self.rhs_exp_expr_interp[i] = fem.Expression(self.lmbda_expr[i]*(self.exact.uh.sub(i)-self.yinf_expr[i]),self.V.sub(i).element.interpolation_points())
                self.F_exp += self.lmbda_expr[i]*(self.exact.uh.sub(i)-self.yinf_expr[i])*v[i]*ufl.dx                                              
                self.exp_indexes[i] = True  
        self.phi_one_f_form = fem.form(self.phi_one_f)
        self.f_exp_form = fem.form(self.F_exp)

        self.F_stiff = 0.
        self.stiff_indexes = [False]*self.exact.size
        self.rhs_stiff_expr_interp = [None]*self.exact.size
        for i in range(self.exact.size):            
            if self.rhs_stiff_expr[i] is not None:
                self.rhs_stiff_expr_interp[i] = fem.Expression(self.rhs_stiff_expr[i],self.V.sub(i).element.interpolation_points())
                self.F_stiff += self.rhs_stiff_expr[i]*v[i]*ufl.dx                                
                self.stiff_indexes[i] = True
        self.f_stiff_form = fem.form(self.F_stiff)

        self.F_nonstiff = self.F_bnd
        self.nonstiff_indexes = [False]*self.exact.size
        self.rhs_nonstiff_expr_interp = [None]*self.exact.size
        for i in range(self.exact.size):
            if self.rhs_nonstiff_expr[i] is not None:
                self.rhs_nonstiff_expr_interp[i] = fem.Expression(self.rhs_nonstiff_expr[i],self.V.sub(i).element.interpolation_points())   
                self.F_nonstiff += self.rhs_nonstiff_expr[i]*v[i]*ufl.dx
                self.nonstiff_indexes[i] = True           
        self.f_nonstiff_form = fem.form(self.F_nonstiff)

        self.one = self.dtype_u(init=self.V,val=0.)    
        for i in range(self.exact.size):
            self.one.values.sub(i).interpolate(fem.Expression(fem.Constant(self.domain,1.),self.V.sub(i).element.interpolation_points()))     

    def eval_f(self, u, t, eval_impl=True, eval_expl=True, eval_exp=True, fh=None):
        """
        Evaluates F(u,t) = M^-1*( A*u + f(u,t) )

        Returns:
            dtype_u: solution as mesh
        """

        self.exact.update_time(t)

        if fh is None:
            fh = self.dtype_f(init=self.V,val=0.)    

        # update the Dirichlet boundary conditions. Needed to apply lifting and as well to evaluate the
        # f_form in the case where boundary conditions are applied via penalty (e.g. DG)
        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)
        
        # update uh, which may be needed to compute the f_form (in the case where f(u,t) depends on u)
        if hasattr(self.exact,'uh'):
            self.exact.uh.copy(u)
            self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)       
        
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

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        if self.f_interp:
            fem.petsc.assemble_vector(self.b, self.f_bnd_form)
        else:
            fem.petsc.assemble_vector(self.b, self.f_nonstiff_form)     
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
        self.invert_mass_matrix(self.b, fh_nonstiff.values.vector)

        if self.f_interp and any(self.nonstiff_indexes):
            self.interp_f.zero()
            for i in range(self.exact.size):
                if self.nonstiff_indexes[i]:
                    self.interp_f.values.sub(i).interpolate(self.rhs_nonstiff_expr_interp[i])             
            fh_nonstiff += self.interp_f

        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh_nonstiff.values.vector, bcs=self.bc, scale=0.)

        return fh_nonstiff

    def eval_f_stiff(self,u,t,fh_stiff):

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        changed_b = False
        if not self.f_interp and any(self.stiff_indexes):            
            fem.petsc.assemble_vector(self.b, self.f_stiff_form)
            changed_b = True
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.apply_lifting(self.b, [self.diff_form], [self.bc])
            changed_b = True
        if changed_b:
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
        
        self.K.multAdd(-u.values.vector,self.b,self.b)             
        self.invert_mass_matrix(self.b, fh_stiff.values.vector)
        
        if self.f_interp and any(self.stiff_indexes):
            self.interp_f.zero()
            for i in range(self.exact.size):
                if self.stiff_indexes[i]:
                    self.interp_f.values.sub(i).interpolate(self.rhs_stiff_expr_interp[i])             
            fh_stiff += self.interp_f        

        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh_stiff.values.vector, bcs=self.bc, scale=0.)

        return fh_stiff
    
    def eval_f_exp(self,u,t,fh_exp):
        
        if not self.f_interp and any(self.exp_indexes): 
            with self.b.localForm() as loc_b:
                loc_b.set(0)           
            fem.petsc.assemble_vector(self.b, self.f_exp_form)
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
            self.invert_mass_matrix(self.b, fh_exp.values.vector)
        else:
            fh_exp.zero()

        if self.f_interp and any(self.exp_indexes):
            self.interp_f.zero()
            for i in range(self.exact.size):
                if self.exp_indexes[i]:
                    self.interp_f.values.sub(i).interpolate(self.rhs_exp_expr_interp[i])             
            fh_exp += self.interp_f

        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(fh_exp.values.vector, bcs=self.bc, scale=0.)

        return fh_exp
    
    def phi_eval(self, u, factor, t, k, u_sol = None):

        self.exact.update_time(t)
        self.exact.update_dt(factor) # for computing the exponentials

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)            

        self.exact.uh.copy(u)
        self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    

        if self.f_interp:
            for i in range(self.exact.size):
                if self.exp_indexes[i]:
                    u_sol.values.sub(i).interpolate(self.phi_expr_interp[i][k])           
        else:
            raise Exception('Not implemented')
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.phi_one_form)   
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
            self.invert_mass_matrix(self.b, u_sol.values.vector)                  

        u_sol.copy_sub(self.one,[i for i in range(self.exact.size) if not self.exp_indexes[i]])
        
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(u_sol.values.vector, bcs=self.bc, scale=1.)

        return u_sol
    
    # def phi_one_eval(self, u, factor, t, u_sol = None):

    #     self.exact.update_time(t)
    #     self.exact.update_dt(factor) # for computing the exponentials

    #     if u_sol is None:
    #         u_sol = self.dtype_u(init=self.V,val=0.)    

    #     if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
    #         self.get_DirBC(self.uD,t)

    #     if hasattr(self.exact,'uh'):
    #         self.exact.uh.copy(u)
    #         self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    

    #     if self.f_interp:
    #         for i in range(self.exact.size):
    #             if self.exp_indexes[i]:
    #                 u_sol.values.sub(i).interpolate(self.phi_one_expr_interp[i])           
    #     else:
    #         with self.b.localForm() as loc_b:
    #             loc_b.set(0)
    #         fem.petsc.assemble_vector(self.b, self.phi_one_form)   
    #         self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
    #         self.invert_mass_matrix(self.b, u_sol.values.vector)                  

    #     u_sol.copy_sub(self.one,[i for i in range(self.exact.size) if not self.exp_indexes[i]])
        
    #     if self.family=='CG' and self.exact.bnd_cond!='N':
    #         fem.petsc.set_bc(u_sol.values.vector, bcs=self.bc, scale=1.)

    #     return u_sol
    
    def lmbda_eval(self, u, t, u_sol = None):

        self.exact.update_time(t)
        # self.exact.update_dt(factor) # for computing the exponentials

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)        

        self.exact.uh.copy(u)
        self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    

        if self.f_interp:
            for i in range(self.exact.size):
                if self.exp_indexes[i]:
                    u_sol.values.sub(i).interpolate(self.lmbda_expr_interp[i])           
        else:
            raise Exception('not implemented')
            # with self.b.localForm() as loc_b:
            #     loc_b.set(0)
            # fem.petsc.assemble_vector(self.b, self.phi_one_form)   
            # self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
            # self.invert_mass_matrix(self.b, u_sol.values.vector)                  

        u_sol.zero_sub([i for i in range(self.exact.size) if not self.exp_indexes[i]])

        return u_sol
    
    def phi_one_f_eval(self, u, factor, t, u_sol = None):

        self.exact.update_time(t)
        self.exact.update_dt(factor) # for computing the exponentials

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)    

        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)

        if hasattr(self.exact,'uh'):
            self.exact.uh.copy(u)
            self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    

        if self.f_interp:
            for i in range(self.exact.size):
                if self.exp_indexes[i]:
                    u_sol.values.sub(i).interpolate(self.phi_one_f_expr_interp[i])           
        else:
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.phi_one_f_form)   
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
            self.invert_mass_matrix(self.b, u_sol.values.vector)                  

        u_sol.zero_sub([i for i in range(self.exact.size) if not self.exp_indexes[i]])
        
        if self.family=='CG' and self.exact.bnd_cond!='N':
            fem.petsc.set_bc(u_sol.values.vector, bcs=self.bc, scale=1.)

        return u_sol