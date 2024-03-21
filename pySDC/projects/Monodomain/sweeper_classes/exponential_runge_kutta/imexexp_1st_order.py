import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError, ParameterError
from pySDC.core.Collocation import CollBase
import numdifftools.fornberg as fornberg
import scipy


class imexexp_1st_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEXEXP sweeper using implicit/explicit/exponential Euler as base integrator
    In the cardiac electrphysiology community this is known as Rush-Larsen scheme.

    The underlying intergrator is exponential Runge-Kutta, leading to exponential SDC (ESDC).
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if "QI" not in params:
            params["QI"] = "IE"

        # call parent's initialization routine
        super(imexexp_1st_order, self).__init__(params)

        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.delta = np.diagonal(self.QI)[1:]

        self.lmbda = None

        self.lambda_and_phi_outdated = True

        # Compute weights w such that PiQ^(k)(0) = sum_{j=0}^{M-1} w[k,j]*Q[j], k=0,...,M-1
        # Used to express the derivatives of a polynomial in x=0 in terms of the values of the polynomial at the collocation nodes
        M = self.coll.num_nodes
        c = self.coll.nodes
        self.w = fornberg.fd_weights_all(c, 0.0, M - 1)

        # Define the quadature rule for the evaluation of the phi_i(z) functions. Indeed, we evaluate them as integrals in order to avoid round off errors.
        phi_num_nodes = 5  # seems to be enough in most cases
        self.phi_coll = CollBase(num_nodes=phi_num_nodes, tleft=0, tright=1, node_type='LEGENDRE', quad_type='GAUSS')

    def phi_eval_lists(self, P, factors, indeces, phi, lmbda, update_non_exp_indeces=True):
        """
        Evaluate the phi_k functions at the points factor_i*lmbda

        Arguments:
            P: problem class
            factors: list of factors to multiply lmbda with. len(factors)=len(self.num_nodes)
            indeces: list of indeces k for the phi_k functions
            phi: list of lists of dtype_u: some space to store the results
            lmbda: dtype_u: the value of lmbda
            update_non_exp_indeces: bool: if True, the phi functions are also evaluated at the non-exponential indeces. Hence, where lambda=0 and thus we return phi_k(0) (using analytical value)

        Returns:
            list of lists of dtype_u: [[phi_k(factor_i*lambda) for i in range(len(factors))] for k in indeces]

        """

        N_fac = len(factors)
        N_ind = len(indeces)

        if phi is None:
            # make some space
            phi = [[P.dtype_u(init=P.init, val=0.0) for i in range(N_fac)] for j in range(N_ind)]
        else:
            # zero out the provided space
            for n in range(N_fac):
                for m in range(N_ind):
                    phi[m][n].zero_sub(P.rhs_exp_indeces)

        factorials = scipy.special.factorial(np.array(indeces) - 1)
        # the quadrature rule is used to evaluate the phi functions as integrals. This is not the same as the one used in the ESDC method!!!!
        c = self.phi_coll.nodes
        b = self.phi_coll.weights

        # iterate only over the indeces of the problem having an exponential term
        # hence we update only the subvectors phi_k(factor*lambda)_i in the vector of vectors phi_k(factor*lambda)
        for i in P.rhs_exp_indeces:
            # iterate over all the factors
            for n in range(N_fac):
                factor = factors[n]
                # compute e^((1-c_j)*factor*lmbda) for nodes c_j on the quadrature rule
                exp_terms = [
                    np.exp(((1.0 - c[j]) * factor) * lmbda.np_array(i)) for j in range(self.phi_coll.num_nodes)
                ]
                # iterate over all the indeces k (phi_k)
                for m in range(N_ind):
                    k = indeces[m]
                    km1_fac = factorials[m]  # (k-1)!
                    if k == 0:
                        # rmemeber: phi_0(z) = e^z
                        phi[m][n].np_array(i)[:] = np.exp(factor * lmbda.np_array(i))
                    else:
                        # using the quadrature rule approximate the integral \int_0^1 e^{(1-s)*factor*lambda}*s^{k-1}/(k-1)! ds
                        for j in range(self.phi_coll.num_nodes):
                            phi[m][n].np_array(i)[:] += ((b[j] * c[j] ** (k - 1)) / km1_fac) * exp_terms[j]

        # update the indeces where lambda=0 (i.e. the non-exponential indeces), there phi_k(0) = 1/k!
        if update_non_exp_indeces:
            for n in range(N_fac):
                for m in range(N_ind):
                    k = indeces[m]
                    phi[m][n].copy_sub(P.one, P.rhs_non_exp_indeces)
                    if k > 1:
                        km1_fac = factorials[m]
                        k_fac = km1_fac * k
                        phi[m][n].imul_sub(1.0 / k_fac, P.rhs_non_exp_indeces)

        return phi

    def compute_lambda_phi_Qmat_exp(self):

        if not hasattr(self, "old_V"):
            # make some space for the old value of u[0]
            self.u_old = self.level.prob.dtype_u(init=self.level.prob.init, val=0.0)

        # everything that is computed in this if statement depends on u[0] only
        # To save computations we recompute that only if u[0] has changed.
        # Also, we check only for the first component u[0][0] of u[0] to save more computations.
        # Remember that u[0][0] is a sub_vector representing the potential on the whole mesh and is enough to check if u[0] has changed.
        if not np.allclose(self.u_old[0].numpy_array[:], self.level.u[0][0].numpy_array[:], rtol=1e-10, atol=1e-10):

            self.u_old[0].numpy_array[:] = self.level.u[0][0].numpy_array[:]

            L = self.level
            P = L.prob
            M = self.coll.num_nodes
            c = self.coll.nodes

            # compute lambda(u) of the exponential term f_exp(u)=lmbda(u)*(u-y_inf(u))
            self.lmbda = P.lmbda_eval(L.u[0], L.time)

            if not hasattr(self, "phi"):
                # make some space
                self.phi = [[P.dtype_u(init=P.init, val=0.0) for i in range(M)] for k in range(M + 1)]
                self.phi_one = [[P.dtype_u(init=P.init, val=0.0) for i in range(M)]]
            # evaluate the phi_k(dt*c_i*lambda) functions at the collocation nodes c_i for k=0,...,M
            self.phi = self.phi_eval_lists(P, L.dt * c, list(range(M + 1)), self.phi, self.lmbda, False)
            # evaluates phi_1(dt*delta_i*lambda) for delta_i = c_i - c_{i-1}
            self.phi_one = self.phi_eval_lists(P, L.dt * self.delta, [1], self.phi_one, self.lmbda, True)

            # compute weight for the integration of \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr, where PiQ(r) is a polynomial interpolating
            # Q(c_i)=Q[i].
            # We do so as \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr = \sum_{j=0}^{M-1} Qmat_exp[i,j]*Q[j]
            if not hasattr(self, "Qmat_exp"):
                # make some space
                self.Qmat_exp = [[P.dtype_u(init=P.init, val=0.0) for j in range(M)] for i in range(M)]
            for i in range(M):
                for j in range(M):
                    # zero out previous values
                    self.Qmat_exp[i][j].zero_sub(P.rhs_exp_indeces)
                    for k in range(M):
                        # self.Qmat_exp[i][j] += self.w[k, j] * c[i] ** (k + 1) * self.phi[k + 1][i]
                        self.Qmat_exp[i][j].axpy_sub(
                            self.w[k, j] * c[i] ** (k + 1), self.phi[k + 1][i], P.rhs_exp_indeces
                        )

            self.lambda_and_phi_outdated = False

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp) using exponential Runge-Kutta

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        self.compute_lambda_phi_Qmat_exp()

        if not hasattr(self, "Q"):
            self.Q = [P.dtype_u(init=P.init, val=0.0) for _ in range(M)]
            self.tmp = P.dtype_u(init=P.init, val=0.0)

        for k in range(M):
            # self.Q[k] = L.f[k + 1].exp + self.lmbda * (L.u[0] - L.u[k + 1])  # at the indeces of the exponential rhs, otherwise 0
            self.Q[k].zero_sub(P.rhs_exp_indeces)
            self.Q[k].iadd_sub(L.u[0], P.rhs_exp_indeces)
            self.Q[k].axpy_sub(-1.0, L.u[k + 1], P.rhs_exp_indeces)
            self.Q[k].imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.Q[k].iadd_sub(L.f[k + 1].exp, P.rhs_exp_indeces)

        # integrate RHS over all collocation nodes
        me = [P.dtype_u(init=P.init, val=0.0) for _ in range(M)]
        for m in range(1, self.coll.num_nodes + 1):
            for j in range(1, self.coll.num_nodes + 1):
                # me[m - 1] += self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl) + self.Qmat_exp[m - 1][j - 1] * (self.Q[j - 1])
                me[m - 1].axpy_sub(self.coll.Qmat[m, j], L.f[j].impl, P.rhs_stiff_indeces)
                me[m - 1].axpy_sub(self.coll.Qmat[m, j], L.f[j].expl, P.rhs_nonstiff_indeces)
                self.tmp.copy_sub(self.Q[j - 1], P.rhs_exp_indeces)
                self.tmp.imul_sub(self.Qmat_exp[m - 1][j - 1], P.rhs_exp_indeces)
                me[m - 1].iadd_sub(self.tmp, P.rhs_exp_indeces)
            me[m - 1] *= L.dt

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        integral = self.integrate()
        for m in range(M):
            if L.tau[m] is not None:
                integral[m] += L.tau[m]
        for i in range(1, M):
            integral[M - i] -= integral[M - i - 1]

        def myprint(name, v):
            for j in range(len(v)):
                vtmp = [v[j].val_list[i].values[31] for i in range(4)]
                vtmp = np.array(vtmp)
                print(f"{name}[{j}] = {vtmp}")

        print("Before Sweep")
        myprint("u", L.u)
        myprint("integral", integral)
        myprint("f.expl", [L.f[i].expl for i in range(M + 1)])
        myprint("f.impl", [L.f[i].impl for i in range(M + 1)])
        myprint("f.exp", [L.f[i].exp for i in range(M + 1)])

        # prepare the integral term
        for m in range(M):
            # integral[m] -= L.dt * self.delta[m] * (L.f[m].expl + L.f[m + 1].impl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m])))
            integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m + 1].impl, P.rhs_stiff_indeces)
            integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m].expl, P.rhs_nonstiff_indeces)
            self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            integral[m].axpy_sub(-L.dt * self.delta[m], self.tmp, P.rhs_exp_indeces)

        print("After modifying integral")
        myprint("integral", integral)

        # do the sweep
        for m in range(M):
            # tmp = L.u[m] + integral[m] + L.dt * self.delta[m] * (L.f[m].expl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m])))
            self.tmp.zero()
            self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].expl, P.rhs_nonstiff_indeces)
            self.tmp.aypx(L.dt * self.delta[m], integral[m])
            self.tmp += L.u[m]

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                self.tmp, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m], L.u[m + 1]
            )

            # update function values
            P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m], fh=L.f[m + 1])

        print("After Sweep")
        myprint("u", L.u)
        myprint("integral", integral)
        myprint("f.expl", [L.f[i].expl for i in range(M + 1)])
        myprint("f.impl", [L.f[i].impl for i in range(M + 1)])
        myprint("f.exp", [L.f[i].exp for i in range(M + 1)])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise CollocationError("This option is not implemented yet.")

        return None

    def compute_residual(self, stage=''):
        """
        Computation of the residual using the collocation matrix Q

        Args:
            stage (str): The current stage of the step the level belongs to
        """

        # get current level and problem description
        L = self.level

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        rel_res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            res[m] += L.u[0]
            res[m] -= L.u[m + 1]
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))
            # the different components of the monodomain equation have very different magnitude therefore we use a tailored relative norm here to avoid the cancellation of the smaller components
            rel_res_norm.append(res[m].rel_norm(L.u[0]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(rel_res_norm)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = rel_res_norm[-1]
        else:
            raise ParameterError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                f'full_abs, last_abs, full_rel or last_rel instead'
            )

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
