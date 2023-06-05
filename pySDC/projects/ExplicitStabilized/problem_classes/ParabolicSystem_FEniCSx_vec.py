import numpy as np
from dolfinx import mesh, fem, io, nls, log, geometry
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh import fenicsx_mesh
from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh_vec import fenicsx_mesh_vec, rhs_fenicsx_mesh_vec, exp_rhs_fenicsx_mesh_vec
from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError

class parabolic_system(ptype):

    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = fenicsx_mesh_vec

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

        def dtype_u_vec_fixed(init,val=0.):
            return self.dtype_u_vec(init,val,self.exact.size)
        def dtype_f_vec_fixed(init,val=0.):
            return self.dtype_f_vec(init,val,self.exact.size)
        
        self.dtype_u = dtype_u_vec_fixed
        self.dtype_f = dtype_f_vec_fixed

        self.V = fem.FunctionSpace(self.domain, (problem_params['family'],problem_params['order']))
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
        self.uD = [fem.Function(self.V) for _ in range(self.exact.size)]
        if self.exact.bnd_cond!='N':
            self.get_DirBC(self.uD,self.t0)

        self.find_Dirichlet_dofs()

        self.bc = [None]*self.exact.size
        for i in self.exact.sp_ind:
            self.bc[i] = fem.dirichletbc(self.uD[i], self.dofs_Dbc)

    def get_DirBC(self,u_D,t):        
        self.exact.t.value = t
        for i in self.exact.sp_ind:
            u_D.sub(i).interpolate(fem.Expression(self.exact.uD_expr[i],self.V.element.interpolation_points()))   # find way to interpolate only on dirichlet nodes self.dofs_Dbc

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
        self.dofs_Dbc = fem.locate_dofs_topological(self.V, fdim, facets)

    def define_variational_forms(self):
        
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        
        self.mass  = u * v * ufl.dx

        self.diff = [None]*self.exact.size
        self.F_bnd = [0]*self.exact.size
        self.F = [None]*self.exact.size
        
        if self.family=='CG':
            for i in self.exact.sp_ind:
                self.diff[i] = ufl.dot(self.exact.diff[i]*ufl.grad(u), ufl.grad(v)) * ufl.dx 
                self.F_bnd[i] = - self.exact.g_expr[i]*v*self.ds(2)                 
        elif self.family=='DG':            
            n = ufl.FacetNormal(self.domain)
            h = ufl.FacetArea(self.domain)
            dim = self.dim
            p = self.order
            beta0 = 1./(dim-1)
            if dim==2:
                eta = fem.Constant(self.domain, ScalarType(20.))*p*(p+1)
            elif dim==3:
                eta = fem.Constant(self.domain, ScalarType(20.))*p*(p+2)*(h**0.5)            
            
            for i in self.exact.sp_ind:
                diff_tens = self.exact.diff[i]
                delta = ufl.dot(n,diff_tens*n)
                om_p = delta('-')/(delta('+')+delta('-'))
                om_m = delta('+')/(delta('+')+delta('-'))
                gamma = 2.*delta('+')*delta('-')/(delta('-')+delta('+'))                
                def avg_o(w):                                
                    return om_p*w('+')+om_m*w('-')
                
                self.diff[i] = ufl.inner(diff_tens*ufl.grad(u), ufl.grad(v))*ufl.dx \
                        - ufl.inner(avg_o(diff_tens*ufl.grad(v)), ufl.jump(u, n))*ufl.dS \
                        - ufl.inner(ufl.jump(v, n), avg_o(diff_tens*ufl.grad(u)))*ufl.dS \
                        + eta/(h**beta0)*gamma*ufl.inner(ufl.jump(v, n), ufl.jump(u, n))*ufl.dS \
                        - ufl.inner(diff_tens*ufl.grad(v), u*n)*self.ds(1) \
                        - ufl.inner(v*n, diff_tens*ufl.grad(u))*self.ds(1) \
                        + (eta/(h**beta0))*delta*v*u*self.ds(1)
                self.F_bnd[i] = - ufl.inner(diff_tens*ufl.grad(v), self.exact.uD_expr[i]*n)*self.ds(1) + (eta/(h**beta0))*(delta)*self.exact.uD_expr[i]*v*self.ds(1) - self.exact.g_expr[i]*v*self.ds(2)                  
        else:
            raise ParameterError("problem_params['family'] must be either 'CG' or 'DG'")

        for i in range(self.exact.size):
            self.F[i] = self.exact.rhs_expr[i]*v*ufl.dx + self.F_bnd[i]

        self.mass_form = fem.form(self.mass)
        self.diff_form = [None]*self.exact.size
        self.f_form = [None]*self.exact.size
        self.f_bnd_form = [None]*self.exact.size
        for i in self.exact.sp_ind:
            self.diff_form[i] = fem.form(self.diff[i])
            self.f_bnd_form[i] = fem.form(self.F_bnd[i])
        for i in range(self.exact.size):
            self.f_form[i] = fem.form(self.F[i])
        
        self.rhs_expr_interp = [fem.Expression(self.exact.rhs_expr[i],self.V.element.interpolation_points()) for i in range(self.exact.size)]
        self.interp_f = fenicsx_mesh(self.init,0.)

    def assemble_vec_mat(self):
        self.b = fem.petsc.create_vector(self.f_form[0])

        if self.family=='CG' and self.exact.bnd_cond!='N':
            sp0 = self.exact.sp_ind[0]
            self.M = fem.petsc.assemble_matrix(self.mass_form,bcs=[self.bc[sp0]])
            self.M.assemble()
            self.K = [None]*self.exact.size
            for i in self.exact.sp_ind:
                self.K[i] = fem.petsc.assemble_matrix(self.diff_form[i],bcs=[self.bc[i]])
                self.K[i].assemble()
        else:
            self.M = fem.petsc.assemble_matrix(self.mass_form)
            self.M.assemble()
            self.K = [None]*self.exact.size
            for i in self.exact.sp_ind:
                self.K[i] = fem.petsc.assemble_matrix(self.diff_form[i])                    
                self.K[i].assemble()

        if self.mass_lumping:
            sp0 = self.exact.sp_ind[0]
            self.uD[sp0].interpolate(lambda x: 1.0 + 0.*x[0])
            mass_lumped = ufl.action(self.mass, self.uD[sp0])
            self.Ml = fem.petsc.assemble_matrix(self.mass_form)
            self.Ml.zeroEntries()
            self.ml = fem.petsc.create_vector(fem.form(mass_lumped))
            with self.ml.localForm() as m_loc:
                m_loc.set(0)
            fem.petsc.assemble_vector(self.ml, fem.form(mass_lumped))
            self.ml.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            self.ml.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.set_bc(self.ml, [self.bc[sp0]]) # self.uD is ==1 so will set 1 at boundary nodes
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
            for i in range(self.exact.size):
                uh.sub(i).name = f"u_{i+1}"            
                self.xdmf.write_function(uh.sub(i), t)            

    def initial_value(self):
        u0 = self.dtype_u(self.init,val=0.)
        for i in range(self.exact.size):
            u0.sub(i).name = f"u_{i+1}"
            u0.sub(i).interpolate(fem.Expression(self.exact.u0_expr[i],self.V.element.interpolation_points()))       
        return u0

    def u_exact(self,t):
        u_ex = self.dtype_u(init=self.init,val=0.)
        self.get_exact(u_ex,t)
        return u_ex
    
    def get_exact(self,u_ex,t):
        if not self.know_exact:
            raise Exception('Exact solution unknown.')
        self.exact.t.value = t
        for i in range(self.exact.size):
            u_ex.sub(i).name = f"u_{i+1}"
            u_ex.sub(i).interpolate(fem.Expression(self.exact.sol_expr[i],self.V.element.interpolation_points()))             
    
    def compute_errors(self,uh,t):
        # Compute L2 error and error at nodes
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_ex = self.u_exact(t)
        errors_L2 = []
        norms_L2_sol = []
        rel_errors_L2 = []
        for i in range(self.exact.size):
            errors_L2.append( np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form((uh.sub(i) - u_ex.sub(i))**2 * ufl.dx)), op=MPI.SUM)) )
            norms_L2_sol.append( np.sqrt(self.domain.comm.allreduce(fem.assemble_scalar(fem.form(u_ex.sub(i)**2 * ufl.dx)), op=MPI.SUM)) )
            rel_errors_L2.append(errors_L2[-1]/norms_L2_sol[-1])
        if self.domain.comm.rank == 0:
            print(f"L2-errors: {errors_L2}")
            print(f"Relative L2-errors: {rel_errors_L2}")

    def get_size(self):
        sp0 = self.exact.sp_ind[0]
        return self.uD[sp0].vector.getSize()
    
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
        u_val_loc =  u.sub(0).eval(self.points_on_proc, self.cells)
        u_val_loc = np.reshape(u_val_loc,(self.points_on_proc.shape[0],1))
        # u_val_glob = np.zeros((self.exact.eval_points[0],2))
        # for i in range(self.points_on_proc.shape[0]):
        #     u_val_glob[i,0] = self.loc_glob_map[i]
        #     u_val_glob[i,1] = u_val_loc[i]
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
        
        sp0 = self.exact.sp_ind[0]

                                                                  
        for i in range(self.exact.size):
            # evaluate f_form  
            with self.b.localForm() as loc_b:
                loc_b.set(0)
                
            if self.f_interp and i in self.exact.sp_ind:
                fem.petsc.assemble_vector(self.b, self.f_bnd_form[i])
            elif not self.f_interp:
                fem.petsc.assemble_vector(self.b, self.f_form[i])
                
            # apply stiffness matrix
            if self.family=='CG' and self.exact.bnd_cond!='N' and i in self.exact.sp_ind:
                fem.petsc.apply_lifting(self.b, [self.diff_form[i]], [[self.bc[i]]])

            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            if i in self.exact.sp_ind:
                self.K[i].multAdd(-u[i].values.vector,self.b,self.b) 

            self.invert_mass_matrix(self.b, fh[i].values.vector)

            if self.f_interp:                
                self.interp_f.values.interpolate(self.rhs_expr_interp[i])
                fh.val_list[i] += self.interp_f

            if self.family=='CG' and self.exact.bnd_cond!='N':
                fem.petsc.set_bc(fh[i].values.vector, [self.bc[sp0]], scale=0.)
                    
        return fh

class parabolic_system_imex(parabolic_system):

    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = rhs_fenicsx_mesh_vec

    def __init__(self, **problem_params):        
        super(parabolic_system_imex,self).__init__(**problem_params)      

        self.solver = [None]*self.exact.size
        self.prev_factor = -1.
        for i in self.exact.sp_ind:
            self.solver[i] = PETSc.KSP().create(self.domain.comm)
            self.solver[i].setType(self.solver_ksp)
            self.solver[i].getPC().setType(self.solver_pc)

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
        
        sp0 = self.exact.sp_ind[0]

        # evaluate explicit (non stiff) part M^-1*f(u,t)
        for i in range(self.exact.size):
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            if self.f_interp and i in self.exact.sp_ind:
                fem.petsc.assemble_vector(self.b, self.f_bnd_form[i])
            elif not self.f_interp:
                fem.petsc.assemble_vector(self.b, self.f_form[i])                
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
            self.invert_mass_matrix(self.b, fh.expl[i].values.vector)
            if self.f_interp:     
                self.interp_f.values.interpolate(self.rhs_expr_interp[i])             
                fh.expl.val_list[i] += self.interp_f
            if self.family=='CG' and self.exact.bnd_cond!='N':
                fem.petsc.set_bc(fh.expl[i].values.vector, [self.bc[sp0]], scale=0.)
                
        # evaluate implicit (stiff) part M^1*A*u
        for i in range(self.exact.size):
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            if self.family=='CG' and self.exact.bnd_cond!='N' and i in self.exact.sp_ind:
                fem.petsc.apply_lifting(self.b, [self.diff_form[i]], [[self.bc[i]]])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
            if i in self.exact.sp_ind:
                self.K[i].multAdd(-u[i].values.vector,self.b,self.b)                 
            self.invert_mass_matrix(self.b, fh.impl[i].values.vector)        
            if self.family=='CG' and self.exact.bnd_cond!='N':
                fem.petsc.set_bc(fh.impl[i].values.vector, [self.bc[sp0]], scale=0.)
                            
        return fh
    
    def solve_system(self, rhs, factor, u0, t, u_sol = None):        

        self.exact.update_time(t)

        if self.family=='CG' and self.exact.bnd_cond!='N' and not self.exact.cte_Dirichlet:
            self.get_DirBC(self.uD,t)

        if u_sol is None:
            u_sol = self.dtype_u(self.V)        

        if abs(factor-self.prev_factor)>1e-8*factor:
            self.prev_factor = factor
            if self.mass_lumping:
                for i in [i for i in range(self.exact.size) if i in self.exact.sp_ind]:
                    self.solver[i].setOperators(self.Ml+factor*self.K[i])
            else:
                for i in [i for i in range(self.exact.size) if i in self.exact.sp_ind]:
                    self.solver[i].setOperators(self.M+factor*self.K[i])            
        
        for i in range(self.exact.size):
            if i in self.exact.sp_ind:
                with self.b.localForm() as loc_b:
                    loc_b.set(0)     

                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.apply_lifting(self.b, [self.diff_form[i]], [[self.bc[i]]])
                    with self.b.localForm() as loc_b:
                        loc_b *= factor
                
                self.mass_mult_add(rhs[i].values.vector,self.b,self.b)  
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    rhs[i].values.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    fem.petsc.apply_lifting(self.b, [self.mass_form], [[self.bc[i]]], x0=[rhs[i].values.vector])
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                    fem.petsc.set_bc(self.b, [self.bc[i]], scale=1.+factor)                
                
                if factor==0.:
                    self.invert_mass_matrix(self.b, u_sol[i].values.vector)
                else:
                    self.solver[i].solve(self.b, u_sol[i].values.vector)
            
            else:
                with self.b.localForm() as loc_b:
                    loc_b.set(0)     
                
                self.mass_mult_add(rhs[i].values.vector,self.b,self.b)  
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    rhs[i].values.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    fem.petsc.apply_lifting(self.b, [self.mass_form], [[self.bc[i]]], x0=[rhs[i].values.vector])

                self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(self.b, [self.bc[self.exact.sp_ind[0]]], scale=0.)                        
                                
                self.invert_mass_matrix(self.b, u_sol[i].values.vector)

        return u_sol


class parabolic_system_exp_expl_impl(parabolic_system_imex):

    dtype_u_vec = fenicsx_mesh_vec
    dtype_f_vec = exp_rhs_fenicsx_mesh_vec

    def __init__(self, **problem_params):        
        super(parabolic_system_exp_expl_impl,self).__init__(**problem_params)                 

    def define_variational_forms(self):        
        super().define_variational_forms()

        v = ufl.TestFunction(self.V)

        splitting = self.splitting
        self.lmbda_expr = self.exact.lmbda_expr[splitting]
        self.yinf_expr = self.exact.yinf_expr[splitting]
        self.rhs_stiff_expr = self.exact.rhs_stiff_expr[splitting]
        self.rhs_nonstiff_expr = self.exact.rhs_nonstiff_expr[splitting]
        self.rhs_nonstiff_args = self.exact.rhs_nonstiff_args[splitting]
        self.rhs_stiff_args = self.exact.rhs_stiff_args[splitting]
        self.rhs_exp_args = self.exact.rhs_exp_args[splitting]

        self.phi_max = 5
        self.lmbda_expr_interp = [None]*self.exact.size
        self.phi = [None]*self.exact.size
        self.phi_form = [None]*self.exact.size
        self.phi_expr_interp = [None]*self.exact.size
        self.phi_one_f = [None]*self.exact.size
        self.phi_one_f_form = [None]*self.exact.size
        self.phi_one_f_expr_interp = [None]*self.exact.size
        self.F_exp = [None]*self.exact.size
        self.f_exp_form = [None]*self.exact.size
        self.rhs_exp_expr_interp = [None]*self.exact.size
        self.exp_indexes = [False]*self.exact.size
        for i in range(self.exact.size):            
            if self.lmbda_expr[i] is not None:
                self.lmbda_expr_interp[i] = fem.Expression(self.lmbda_expr[i],self.V.element.interpolation_points())

                self.phi_expr_interp[i] = [None]*(self.phi_max+1)
                self.phi[i] = [None]*(self.phi_max+1)
                self.phi_form[i] = [None]*(self.phi_max+1)
                phi_k = ufl.exp(self.lmbda_expr[i]*self.exact.dt)                
                self.phi_expr_interp[i][0] = fem.Expression(phi_k,self.V.element.interpolation_points())
                self.phi[i][0] = phi_k*v*ufl.dx
                self.phi_form[i][0] = fem.form(self.phi[i][0])
                k_fac = 1. # 0!
                for k in range(1,self.phi_max+1):                    
                    phi_k = (phi_k-1./k_fac)/(self.lmbda_expr[i]*self.exact.dt)                    
                    self.phi_expr_interp[i][k] = fem.Expression(phi_k,self.V.element.interpolation_points())
                    self.phi[i][k] = phi_k*v*ufl.dx
                    self.phi_form[i][k] = fem.form(self.phi[i][k])                    
                    k_fac = k_fac*k

                self.phi_one_f_expr_interp[i] = fem.Expression(\
                                                ((ufl.exp(self.lmbda_expr[i]*self.exact.dt)-1.)/(self.exact.dt))*(self.exact.uh.sub(i)-self.yinf_expr[i]),\
                                                self.V.element.interpolation_points())
                self.phi_one_f[i] = ((ufl.exp(self.lmbda_expr[i]*self.exact.dt)-1.)/(self.exact.dt))*(self.exact.uh.sub(i)-self.yinf_expr[i])*v*ufl.dx    
                self.phi_one_f_form[i] = fem.form(self.phi_one_f[i])
                self.rhs_exp_expr_interp[i] = fem.Expression(self.lmbda_expr[i]*(self.exact.uh.sub(i)-self.yinf_expr[i]),self.V.element.interpolation_points())
                self.F_exp[i] = self.lmbda_expr[i]*(self.exact.uh.sub(i)-self.yinf_expr[i])*v*ufl.dx                                              
                self.f_exp_form[i] = fem.form(self.F_exp[i])
                self.exp_indexes[i] = True          
        
        self.F_stiff = [None]*self.exact.size
        self.f_stiff_form = [None]*self.exact.size
        self.rhs_stiff_expr_interp = [None]*self.exact.size
        self.stiff_indexes = [False]*self.exact.size
        for i in range(self.exact.size):            
            if self.rhs_stiff_expr[i] is not None:
                self.F_stiff[i] = self.rhs_stiff_expr[i]*v*ufl.dx
                self.f_stiff_form[i] = fem.form(self.F_stiff[i])
                self.rhs_stiff_expr_interp[i] = fem.Expression(self.rhs_stiff_expr[i],self.V.element.interpolation_points())
                self.stiff_indexes[i] = True
        
        self.F_nonstiff = [None]*self.exact.size        
        self.f_nonstiff_form = [None]*self.exact.size        
        self.rhs_nonstiff_expr_interp = [None]*self.exact.size        
        self.nonstiff_indexes = [False]*self.exact.size
        for i in range(self.exact.size):
            if self.rhs_nonstiff_expr[i] is not None:
                self.F_nonstiff[i] = self.rhs_nonstiff_expr[i]*v*ufl.dx
                if i in self.exact.sp_ind:
                    self.F_nonstiff[i] += self.F_bnd[i]
                self.f_nonstiff_form[i] = fem.form(self.F_nonstiff[i])
                self.rhs_nonstiff_expr_interp[i] = fem.Expression(self.rhs_nonstiff_expr[i],self.V.element.interpolation_points())   
                self.nonstiff_indexes[i] = True     
            elif i in self.exact.sp_ind:                
                self.F_nonstiff[i] = self.F_bnd[i]
                self.f_nonstiff_form[i] = fem.form(self.F_nonstiff[i])

        self.one = self.dtype_u(init=self.V,val=0.)    
        for i in range(self.exact.size):
            self.one.sub(i).interpolate(fem.Expression(fem.Constant(self.domain,1.),self.V.element.interpolation_points()))    

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
        
        # evaluate explicit (non stiff) part M^-1*f_nonstiff(u,t)
        if eval_expl:                                                
            fh.expl = self.eval_f_nonstiff(u,t,fh.expl) 
                
        # evaluate implicit (stiff) part M^1*A*u+M^-1*f_stiff(u,t)
        if eval_impl:
            fh.impl = self.eval_f_stiff(u,t,fh.impl)

        # evaluate exponential part
        if eval_exp:                                                
            fh.exp = self.eval_f_exp(u,t,fh.exp) 
                            
        return fh
    
    def eval_f_nonstiff(self,u,t,fh_nonstiff):
        sp0 = self.exact.sp_ind[0]
        for i in range(self.exact.size):
            if not (self.nonstiff_indexes[i] or i in self.exact.sp_ind):
                fh_nonstiff[i].zero()
            else:
                with self.b.localForm() as loc_b:
                    loc_b.set(0)
                changed_b = False
                if self.f_interp and i in self.exact.sp_ind:
                    fem.petsc.assemble_vector(self.b, self.f_bnd_form[i])
                    changed_b = True
                elif not self.f_interp and (self.nonstiff_indexes[i] or i in self.exact.sp_ind):
                    fem.petsc.assemble_vector(self.b, self.f_nonstiff_form[i])     
                    changed_b = True           
                if changed_b:
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                
                    self.invert_mass_matrix(self.b, fh_nonstiff[i].values.vector)
                else:
                    fh_nonstiff[i].zero()
                if self.f_interp and self.nonstiff_indexes[i]:
                    self.interp_f.values.interpolate(self.rhs_nonstiff_expr_interp[i])             
                    fh_nonstiff.val_list[i] += self.interp_f
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(fh_nonstiff[i].values.vector, [self.bc[sp0]], scale=0.)

        return fh_nonstiff

    def eval_f_stiff(self,u,t,fh_stiff):
        sp0 = self.exact.sp_ind[0]
        for i in range(self.exact.size):
            if not (self.stiff_indexes[i] or i in self.exact.sp_ind):
                fh_stiff[i].zero()
            else:
                with self.b.localForm() as loc_b:
                    loc_b.set(0)
                changed_b = False
                if not self.f_interp and self.stiff_indexes[i]:                    
                    fem.petsc.assemble_vector(self.b, self.f_stiff_form[i])
                    changed_b = True
                if self.family=='CG' and self.exact.bnd_cond!='N' and i in self.exact.sp_ind:
                    fem.petsc.apply_lifting(self.b, [self.diff_form[i]], [[self.bc[i]]])
                    changed_b = True
                if changed_b:
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) 
                if i in self.exact.sp_ind:
                    self.K[i].multAdd(-u[i].values.vector,self.b,self.b)     
                    changed_b = True
                if changed_b:            
                    self.invert_mass_matrix(self.b, fh_stiff[i].values.vector)
                else:
                    fh_stiff[i].zero()
                if self.f_interp and self.stiff_indexes[i]:              
                    self.interp_f.values.interpolate(self.rhs_stiff_expr_interp[i])
                    fh_stiff.val_list[i] += self.interp_f
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(fh_stiff[i].values.vector, [self.bc[sp0]], scale=0.)

        return fh_stiff
    
    def eval_f_exp(self,u,t,fh_exp):
        sp0 = self.exact.sp_ind[0]
        for i in range(self.exact.size):
            if not self.exp_indexes[i]:
                fh_exp[i].zero()
            else:                
                if self.f_interp:                    
                    fh_exp.val_list[i].values.interpolate(self.rhs_exp_expr_interp[i])                    
                else:                              
                    with self.b.localForm() as loc_b:
                        loc_b.set(0)                
                    fem.petsc.assemble_vector(self.b, self.f_exp_form[i])
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                          
                    self.invert_mass_matrix(self.b, fh_exp[i].values.vector)
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(fh_exp[i].values.vector, [self.bc[sp0]], scale=0.)

        return fh_exp
    
    def phi_eval(self, u, factor, t, k, u_sol = None):
        
        self.exact.update_time(t)
        self.exact.update_dt(factor) # for computing the exponentials

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)    
        
        self.exact.uh.copy(u)
        self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    
        
        sp0 = self.exact.sp_ind[0]
        
        for i in range(self.exact.size):
            if self.exp_indexes[i]:
                if self.f_interp:                    
                    u_sol.val_list[i].values.interpolate(self.phi_expr_interp[i][k])                    
                else:                              
                    with self.b.localForm() as loc_b:
                        loc_b.set(0)                
                    fem.petsc.assemble_vector(self.b, self.phi_form[i][k])
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                          
                    self.invert_mass_matrix(self.b, u_sol[i].values.vector)
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(u_sol[i].values.vector, [self.bc[sp0]], scale=0.)
            
        u_sol.copy_sub(self.one,[i for i in range(self.exact.size) if not self.exp_indexes[i]])

        return u_sol
    
    def lmbda_eval(self, u, t, u_sol = None):
        
        self.exact.update_time(t)        

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)    
        
        self.exact.uh.copy(u)
        self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    
        
        sp0 = self.exact.sp_ind[0]
        
        for i in range(self.exact.size):
            if self.exp_indexes[i]:
                if self.f_interp:                    
                    u_sol.val_list[i].values.interpolate(self.lmbda_expr_interp[i])                    
                else:      
                    raise Exception('not implemented')                        
                    with self.b.localForm() as loc_b:
                        loc_b.set(0)                
                    fem.petsc.assemble_vector(self.b, self.phi_one_form[i])
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                          
                    self.invert_mass_matrix(self.b, u_sol[i].values.vector)
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(u_sol[i].values.vector, [self.bc[sp0]], scale=0.)
            
        u_sol.zero_sub([i for i in range(self.exact.size) if not self.exp_indexes[i]])

        return u_sol
    
    def phi_one_f_eval(self, u, factor, t, u_sol = None):
        
        self.exact.update_time(t)
        self.exact.update_dt(factor) # for computing the exponentials

        if u_sol is None:
            u_sol = self.dtype_u(init=self.V,val=0.)    
        
        if hasattr(self.exact,'uh'):
            self.exact.uh.copy(u)
            self.exact.uh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)    
        
        sp0 = self.exact.sp_ind[0]
        
        for i in range(self.exact.size):
            if not self.exp_indexes[i]:
                u_sol[i].zero()
            else:                
                if self.f_interp:                    
                    u_sol.val_list[i].values.interpolate(self.phi_one_f_expr_interp[i])                    
                else:                              
                    with self.b.localForm() as loc_b:
                        loc_b.set(0)                
                    fem.petsc.assemble_vector(self.b, self.phi_one_f_form[i])
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)                          
                    self.invert_mass_matrix(self.b, u_sol[i].values.vector)
                if self.family=='CG' and self.exact.bnd_cond!='N':
                    fem.petsc.set_bc(u_sol[i].values.vector, [self.bc[sp0]], scale=0.)

        return u_sol
    

