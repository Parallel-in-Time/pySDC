import logging


class rho_estimator:
    def __init__(self, problem, freq=1):
        self.logger = logging.getLogger("rho_estimator")

        self.P = problem
        self.freq = freq
        self.count = 0
        self.eigvec = self.P.dtype_f(self.P.init, val="random")

        if hasattr(self.eigvec, "expl"):
            self.imex = True

            class pair:
                def __init__(self, impl, expl):
                    self.impl = impl
                    self.expl = expl

            self.eigval = pair(0.0, 0.0)
            self.n_f_eval = pair(0, 0)
            # self.diagonal = pair(self.P.diagonal_impl,self.P.diagonal_expl)
        else:
            self.imex = False
            self.eigval = 0.0
            self.n_f_eval = 0
            # self.diagonal = self.P.diagonal

        self.fx = self.P.dtype_f(self.P.init, val='random')

    def rho(self, y, t, fy=None):
        if self.imex:
            if fy is not None:
                self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(
                    lambda x: self.P.eval_f(x, t, eval_impl=False, eval_expl=True, eval_exp=False).expl, self.fx.expl, y, fy.expl, self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl
                )
                self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
                    lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False).impl, self.fx.impl, y, fy.impl, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
                )
            else:
                self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(
                    lambda x: self.P.eval_f(x, t, eval_impl=False, eval_expl=True, eval_exp=False).expl, self.fx.expl, y, None, self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl
                )
                self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
                    lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False).impl, self.fx.impl, y, None, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
                )
        else:
            if fy is not None:
                self.eigval, self.eigvec, self.n_f_eval = self.rho_f(lambda x: self.P.eval_f(x, t), self.fx, y, fy, self.eigval, self.eigvec, self.n_f_eval)
            else:
                self.eigval, self.eigvec, self.n_f_eval = self.rho_f(lambda x: self.P.eval_f(x, t), self.fx, y, None, self.eigval, self.eigvec, self.n_f_eval)

        return self.eigval

    def rho_expl(self, y, t, fy=None):
        if fy is not None:
            self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(
                lambda x: self.P.eval_f(x, t, eval_impl=False, eval_expl=True, eval_exp=False).expl, self.fx.expl, y, fy.expl, self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl
            )
        else:
            self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl = self.rho_f(
                lambda x: self.P.eval_f(x, t, eval_impl=False, eval_expl=True, eval_exp=False).expl, self.fx.expl, y, None, self.eigval.expl, self.eigvec.expl, self.n_f_eval.expl
            )

        return self.eigval.expl

    # def rho_impl(self, y, t, fy=None):
    #     if fy is not None:
    #         self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
    #             lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False).impl, y, fy.impl, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
    #         )
    #     else:
    #         self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
    #             lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False).impl, y, None, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
    #         )

    #     return self.eigval.impl

    # def rho_f(self, f, y, fy, eigval, eigvec, n_f_eval):
    #     # if diagonal:
    #     #     return self.rho_f_diagonal(f,y,fy,eigval,eigvec,n_f_eval)
    #     # else:
    #     return self.rho_f_nondiagonal(f, y, fy, eigval, eigvec, n_f_eval)

    # def rho_f_nondiagonal(self, f, y, fy, eigval, eigvec, n_f_eval):
    #     """
    #     Estimates spectral radius of df/dy, for f with one component.
    #     eigval,eigvec are guesses for the dominant eigenvalue,eigenvector. n_f_eval counts the number of f evaluations
    #     fy can be None or an already available evaluation of f(y)

    #     It is a nonlinear power method based on finite differentiation: df/dy(y)*v = f(y+v)-f(y) + O(|v|^2)
    #     The Rayleigh quotient (dot prod) is replaced (bounded) with an l2-norm.
    #     The algorithm used is a small change (initial vector and stopping criteria) of that of
    #     Sommeijer-Shampine-Verwer, implemented in RKC.
    #     When a guess is provided, in general it converges in 1-2 iterations.
    #     """

    #     maxiter = 100
    #     safe = 1.05
    #     tol = 1e-3
    #     small = 1e-8
    #     n_f_eval_0 = n_f_eval

    #     z = eigvec
    #     if fy is None:
    #         fy = f(y)
    #         n_f_eval += 1

    #     y_norm = abs(y)
    #     z_norm = abs(z)

    #     # Building the vector z so that the difference z-yn is small
    #     if y_norm != 0.0 and z_norm != 0.0:
    #         # here z -> y+z*|y|*small/|z|
    #         dzy = y_norm * small
    #         quot = dzy / z_norm
    #         # z *= quot
    #         # z += y
    #         z.aypx(quot, y)
    #     elif y_norm != 0.0:
    #         # here z-> y*(1+small)
    #         dzy = y_norm * small
    #         z.copy(y)
    #         z *= 1.0 + small
    #     elif z_norm != 0.0:
    #         # here z-> z*small/|z|
    #         dzy = small
    #         quot = dzy / z_norm
    #         z *= quot
    #     else:
    #         # here z=0 becomes z=random and z = z*small/|z|
    #         z = self.P.dtype_u(self.P.init, val="random")
    #         dzy = small
    #         z_norm = abs(z)
    #         quot = dzy / z_norm
    #         z *= quot

    #     """
    #     Here dzy=|z-y| and z=y+(small perturbation)
    #     In the following loop dzy=|z-yn| remains true, even with the new z
    #     """

    #     # Start the power method for non linear operator f
    #     for iter in range(1, maxiter + 1):
    #         eigvec = f(z)
    #         eigvec -= fy
    #         n_f_eval += 1

    #         dfzfy = abs(eigvec)

    #         eigval_old = eigval
    #         eigval = dfzfy / dzy  # approximation of the Rayleigh quotient (not with dot product but just norms)
    #         eigval = safe * eigval

    #         self.logger.debug(f"rho_estimator: iter = {iter}, eigval = {eigval}")

    #         if abs(eigval - eigval_old) <= eigval * tol:
    #             # The last perturbation is stored. It will very likely be a
    #             # good starting point for the next rho call.
    #             eigvec = z
    #             eigvec -= y
    #             break

    #         if dfzfy != 0.0:
    #             quot = dzy / dfzfy
    #             z = eigvec
    #             z.aypx(quot, y)
    #             # z *= quot
    #             # z += y; # z is built such that dzy=|z-yn| is still true
    #         else:
    #             raise Exception("Spectral radius estimation error.")

    #     if iter == maxiter and abs(eigval - eigval_old) > eigval * tol:
    #         self.logger.warning("Spectral radius estimator did not converge.")

    #     self.logger.info(f"Converged to rho = {eigval:1.2e} in {iter} iterations and {n_f_eval-n_f_eval_0} function evaluations.")

    #     return eigval, eigvec, n_f_eval

    def rho_impl(self, y, t, fy=None):
        if fy is not None:
            self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
                lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fx).impl, self.fx.impl, y, fy.impl, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
            )
        else:
            self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl = self.rho_f(
                lambda x: self.P.eval_f(x, t, eval_impl=True, eval_expl=False, eval_exp=False, fh=self.fx).impl, self.fx.impl, y, None, self.eigval.impl, self.eigvec.impl, self.n_f_eval.impl
            )

        return self.eigval.impl

    def rho_f(self, f, fx, y, fy, eigval, eigvec, n_f_eval):
        # if diagonal:
        #     return self.rho_f_diagonal(f,y,fy,eigval,eigvec,n_f_eval)
        # else:
        return self.rho_f_nondiagonal(f, fx, y, fy, eigval, eigvec, n_f_eval)

    def rho_f_nondiagonal(self, f, fx, y, fy, eigval, eigvec, n_f_eval):
        """
        Estimates spectral radius of df/dy.
        eigval, eigvec are guesses for the dominant eigenvalue,eigenvector. n_f_eval counts the number of f evaluations
        fy can be None or an already available evaluation of f(y)

        It is a nonlinear power method based on finite differentiation: df/dy(y)*v = f(y+v)-f(y) + O(|v|^2)
        The Rayleigh quotient (dot prod) is replaced (bounded) with an l2-norm.
        The algorithm used is a small change (initial vector and stopping criteria) of that of
        Sommeijer-Shampine-Verwer, implemented in RKC.
        When a guess is provided, in general it converges in 1-2 iterations.
        """

        maxiter = 100
        safe = 1.05
        tol = 1e-3
        small = 1e-8
        n_f_eval_0 = n_f_eval

        z = eigvec
        if fy is None:
            fy = self.P.dtype_u(self.P.init, val=0.0)
            f(y)
            fy.copy(fx)
            n_f_eval += 1

        y_norm = abs(y)
        z_norm = abs(z)

        # Building the vector z so that the difference z-yn is small
        if y_norm != 0.0 and z_norm != 0.0:
            # here z -> y+z*|y|*small/|z|
            dzy = y_norm * small
            quot = dzy / z_norm
            # z *= quot
            # z += y
            z.aypx(quot, y)
        elif y_norm != 0.0:
            # here z-> y*(1+small)
            dzy = y_norm * small
            z.copy(y)
            z *= 1.0 + small
        elif z_norm != 0.0:
            # here z-> z*small/|z|
            dzy = small
            quot = dzy / z_norm
            z *= quot
        else:
            # here z=0 becomes z=random and z = z*small/|z|
            z = self.P.dtype_u(self.P.init, val="random")
            dzy = small
            z_norm = abs(z)
            quot = dzy / z_norm
            z *= quot

        """
        Here dzy=|z-y| and z=y+(small perturbation)
        In the following loop dzy=|z-yn| remains true, even with the new z
        """

        # Start the power method for non linear operator f
        for iter in range(1, maxiter + 1):
            eigvec = f(z)
            eigvec -= fy
            n_f_eval += 1

            dfzfy = abs(eigvec)

            eigval_old = eigval
            eigval = dfzfy / dzy  # approximation of the Rayleigh quotient (not with dot product but just norms)
            eigval = safe * eigval

            self.logger.debug(f"rho_estimator: iter = {iter}, eigval = {eigval}")

            if abs(eigval - eigval_old) <= eigval * tol:
                # The last perturbation is stored. It will very likely be a
                # good starting point for the next rho call.
                eigvec = z
                eigvec -= y
                break

            if dfzfy != 0.0:
                quot = dzy / dfzfy
                z.copy(eigvec)
                z.aypx(quot, y)
                # z *= quot
                # z += y; # z is built such that dzy=|z-yn| is still true
            else:
                raise Exception("Spectral radius estimation error.")

        if iter == maxiter and abs(eigval - eigval_old) > eigval * tol:
            self.logger.warning("Spectral radius estimator did not converge.")

        self.logger.info(f"Converged to rho = {eigval:1.2e} in {iter} iterations and {n_f_eval-n_f_eval_0} function evaluations.")

        return eigval, eigvec, n_f_eval

    # def rho_f_diagonal(self, f, y, fy, eigval, eigvec, n_f_eval):
    #     """
    #     Estimates spectral radius of df/dy, for f with one component.
    #     eigval,eigvec are guesses for the dominant eigenvalue,eigenvector. n_f_eval counts the number of f evaluations
    #     fy can be None or an already available evaluation of f(y)

    #     It is a nonlinear power method based on finite differentiation: df/dy(y)*v = f(y+v)-f(y) + O(|v|^2)
    #     The Rayleigh quotient (dot prod) is replaced (bounded) with an l2-norm.
    #     The algorithm used is a small change (initial vector and stopping criteria) of that of
    #     Sommeijer-Shampine-Verwer, implemented in RKC.
    #     When a guess is provided, in general it converges in 1-2 iterations.
    #     """

    # import numpy as np
    # import ufl
    # from dolfinx import fem
    #     maxiter = 100
    #     safe = 1.05
    #     tol = 1e-3
    #     small = 1e-8
    #     n_f_eval_0 = n_f_eval

    #     z = eigvec
    #     if fy is None:
    #         fy = f(y)
    #         n_f_eval += 1

    #     y_norm_expr = 0.0
    #     z_norm_expr = 0.0
    #     for i in range(y.size):
    #         y_norm_expr += y.sub(i) ** 2
    #         z_norm_expr += z.sub(i) ** 2
    #     y_norm_expr = ufl.sqrt(y_norm_expr)
    #     z_norm_expr = ufl.sqrt(z_norm_expr)

    #     y_norm = type(y.sub(0))(y.sub(0).function_space)
    #     y_norm.interpolate(fem.Expression(y_norm_expr, y.sub(0).function_space.element.interpolation_points()))
    #     z_norm = type(y.sub(0))(y.sub(0).function_space)
    #     z_norm.interpolate(fem.Expression(z_norm_expr, y.sub(0).function_space.element.interpolation_points()))

    #     # dzy = y_norm*small
    #     dzy = type(y.sub(0))(y.sub(0).function_space)
    #     dzy.interpolate(fem.Expression(y_norm * small, y.sub(0).function_space.element.interpolation_points()))
    #     for i in range(y.size):
    #         z.val_list[i].values.interpolate(fem.Expression(y.sub(i) + dzy * z.sub(i) / z_norm, y.sub(0).function_space.element.interpolation_points()))

    #     """
    #     Here dzy=|z-y| and z=y+(small perturbation)
    #     In the following loop dzy=|z-yn| remains true, even with the new z
    #     """

    #     # Start the power method for non linear operator f
    #     eigvals = type(y.sub(0))(y.sub(0).function_space)
    #     for iter in range(1, maxiter + 1):
    #         eigvec = f(z)
    #         eigvec -= fy
    #         n_f_eval += 1

    #         dfzfy = 0.0
    #         for i in range(y.size):
    #             dfzfy += eigvec.sub(i) ** 2
    #         dfzfy = ufl.sqrt(dfzfy)

    #         eigvals.interpolate(fem.Expression(dfzfy / dzy, y.sub(0).function_space.element.interpolation_points()))

    #         eigval_old = eigval
    #         eigval = safe * np.max(np.abs(eigvals.x.array))

    #         self.logger.debug(f"rho_estimator: iter = {iter}, eigval = {eigval}")

    #         if abs(eigval - eigval_old) <= eigval * tol:
    #             # The last perturbation is stored. It will very likely be a
    #             # good starting point for the next rho call.
    #             eigvec = z
    #             eigvec -= y
    #             break

    #         if dfzfy != 0.0:
    #             quot = dzy / dfzfy
    #             for i in range(z.size):
    #                 z.val_list[i].values.interpolate(fem.Expression(y.sub(i) + quot * eigvec.sub(i), y.sub(0).function_space.element.interpolation_points()))
    #         else:
    #             raise Exception("Spectral radius estimation error.")

    #     if iter == maxiter and abs(eigval - eigval_old) > eigval * tol:
    #         self.logger.warning("Spectral radius estimator did not converge.")

    #     self.logger.info(f"Converged to rho = {eigval:1.2e} in {iter} iterations and {n_f_eval-n_f_eval_0} function evaluations.")

    #     return eigval, eigvec, n_f_eval
