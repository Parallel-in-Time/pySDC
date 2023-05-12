import numpy as np
import logging

class rho_estimator:

    def __init__(self,problem,freq=1):

        self.logger = logging.getLogger('rho_estimator')

        self.P = problem
        self.freq = freq
        self.count = 0
        self.eigvec = self.P.dtype_f(self.P.init, val="random")        

        if hasattr(self.eigvec,"expl"):
            self.imex = True
            class pair:
                def __init__(self,impl,expl):
                    self.impl = impl
                    self.expl = expl
            self.eigval = pair(0.,0.)
            self.n_f_eval = pair(0,0)
        else:
            self.imex = False
            self.eigval = 0.
            self.n_f_eval = 0

    def rho(self,y,t,fy=None):

        if self.count%self.freq==0:
            if self.imex:
                if fy is not None:               
                    self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(lambda x: self.P.eval_f(x,t,eval_impl=False,eval_expl=True).expl,y,fy.expl,self.eigval.expl,self.eigvec.expl,self.n_f_eval.expl)
                    self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(lambda x: self.P.eval_f(x,t,eval_impl=True,eval_expl=False).impl,y,fy.impl,self.eigval.impl,self.eigvec.impl,self.n_f_eval.impl)
                else:                
                    self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(lambda x: self.P.eval_f(x,t,eval_impl=False,eval_expl=True).expl,y,None,self.eigval.expl,self.eigvec.expl,self.n_f_eval.expl)
                    self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(lambda x: self.P.eval_f(x,t,eval_impl=True,eval_expl=False).impl,y,None,self.eigval.impl,self.eigvec.impl,self.n_f_eval.impl)
            else:
                if fy is not None:
                    self.eigval, self.eigvec, self.n_f_eval = self.rho_f(lambda x: self.P.eval_f(x,t),y,fy,self.eigval,self.eigvec,self.n_f_eval)
                else:
                    self.eigval, self.eigvec, self.n_f_eval = self.rho_f(lambda x: self.P.eval_f(x,t),y,None,self.eigval,self.eigvec,self.n_f_eval)

        self.count += 1

        return self.eigval
    
    def rho_f(self,f,y,fy,eigval,eigvec,n_f_eval):
        """
        Estimates spectral radius of df/dy, for f with one component.
        eigval,eigvec are guesses for the dominant eigenvalue,eigenvector. n_f_eval counts the number of f evaluations
        fy can be None or an already available evaluation of f(y)
        
        It is a nonlinear power method based on finite differentiation: df/dy(y)*v = f(y+v)-f(y) + O(|v|^2)
        The Rayleigh quotient (dot prod) is replaced (bounded) with an l2-norm.
        The algorithm used is a small change (initial vector and stopping criteria) of that of
        Sommeijer-Shampine-Verwer, implemented in RKC.
        When a guess is provided, in general it converges in 1-2 iterations.
        """

        maxiter=100
        safe=1.05
        tol = 1e-3    
        small = 1e-8
        n_f_eval_0 = n_f_eval

        z = eigvec
        if fy is None:
            fy = f(y)
            n_f_eval += 1

        y_norm = abs(y)
        z_norm = abs(z)
    
        # Building the vector z so that the difference z-yn is small
        if y_norm!=0.0 and z_norm!=0.0:
            # here z -> y+z*|y|*small/|z|
            dzy = y_norm*small
            quot = dzy/z_norm
            # z *= quot
            # z += y
            z.aypx(quot,y)
        elif y_norm!=0.0:        
            # here z-> y*(1+small)
            dzy = y_norm*small
            z.copy(y)
            z *= 1.+small
        elif z_norm!=0.0:
            # here z-> z*small/|z|
            dzy = small
            quot = dzy/z_norm
            z *= quot
        else:
            # here z=0 becomes z=random and z = z*small/|z|
            z = self.P.dtype_f(self.P.init, val="random")
            dzy = small
            z_norm = abs(z)
            quot = dzy/z_norm
            z *= quot
            
        """
        Here dzy=|z-y| and z=y+(small perturbation)
        In the following loop dzy=|z-yn| remains true, even with the new z
        """

        # Start the power method for non linear operator f
        for iter in range(1,maxiter+1):        
            
            eigvec = f(z)
            eigvec -= fy
            n_f_eval += 1
            
            dfzfy = abs(eigvec)
                    
            eigval_old = eigval
            eigval = dfzfy/dzy # approximation of the Rayleigh quotient (not with dot product but just norms)
            eigval = safe*eigval      
                            
            if abs(eigval-eigval_old)<= eigval*tol:        
                # The last perturbation is stored. It will very likely be a
                # good starting point for the next rho call.
                eigvec = z
                eigvec -= y
                break 

            if dfzfy!=0.0:
                quot = dzy/dfzfy
                z = eigvec
                z.aypx(quot,y)
                # z *= quot
                # z += y; # z is built such that dzy=|z-yn| is still true
            else:
                raise Exception("Spectral radius estimation error.")
        
        if iter==maxiter and abs(eigval-eigval_old)>eigval*tol:
            self.logger.warning("Spectral radius estimator did not converge.")

        self.logger.info(f'Converged to rho = {eigval} in {iter} iterations and {n_f_eval-n_f_eval_0} function evaluations.')

        return eigval, eigvec, n_f_eval
    
