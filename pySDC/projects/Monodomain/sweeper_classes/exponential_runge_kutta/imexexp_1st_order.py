import numpy as np

from pySDC.core.sweeper import Sweeper
from pySDC.core.errors import CollocationError, ParameterError
from pySDC.core.collocation import CollBase
import numdifftools.fornberg as fornberg
import scipy


class imexexp_1st_order(Sweeper):
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
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)
        self.delta = np.diagonal(self.QI)[1:]

        # Compute weights w such that PiQ^(k)(0) = sum_{j=0}^{M-1} w[k,j]*Q[j], k=0,...,M-1
        # Used to express the derivatives of a polynomial in x=0 in terms of the values of the polynomial at the collocation nodes
        M = self.coll.num_nodes
        c = self.coll.nodes
        self.w = fornberg.fd_weights_all(c, 0.0, M - 1).transpose()

        # Define the quadature rule for the evaluation of the phi_i(z) functions. Indeed, we evaluate them as integrals in order to avoid round off errors.
        phi_num_nodes = 5  # seems to be enough in most cases
        self.phi_coll = CollBase(num_nodes=phi_num_nodes, tleft=0, tright=1, node_type='LEGENDRE', quad_type='GAUSS')

    def phi_eval(self, factors, indeces, phi, lmbda):
        """
        Evaluate the phi_k functions at the points factors[i]*lmbda, for all k in indeces

        Arguments:
            factors: list of factors to multiply lmbda with.
            indeces: list of indeces k for the phi_k functions. Since we use the integral formulation, k=0 is not allowed (not needed neither).
            phi: an instance of mesh with shape (len(factors),len(indeces),*lmbda.shape) (i.e., some space to store the results)
                 it will filled as: phi[i,k][:] = phi_{indeces[k]}(factor[i]*lmbda[:])
            lmbda: dtype_u: the value of lmbda
        """

        assert 0 not in indeces, "phi_0 is not implemented, since the integral definition is not valid for k=0."

        # the quadrature rule used to evaluate the phi functions as integrals. This is not the same as the one used in the ESDC method!!!!
        c = self.phi_coll.nodes
        b = self.phi_coll.weights

        k = np.array(indeces)
        km1_fac = scipy.special.factorial(k - 1)  # (k-1)!

        # Here we use the quadrature rule to approximate the integral
        # phi_{k}(factor[i]*lmbda[:,:])= \int_0^1 e^{(1-s)*factor[i]*lambda[:,:]}*s^{k-1}/(k-1)! ds

        # First, compute e^((1-c[j])*factor[i]*lmbda[:]) for nodes c[j] on the quadrature rule and all factors[i]
        exp_terms = np.exp(((1.0 - c[None, :, None, None]) * factors[:, None, None, None]) * lmbda[None, None, :, :])
        # Then, compute the terms c[j]^{k-1}/(k-1)! for all nodes c[j] and all k and multiply with the weights b[j]
        wgt_tmp = (b[:, None] * c[:, None] ** (k[None, :] - 1)) / km1_fac[None, :]
        # Finally, compute the integral by summing over the quadrature nodes
        phi[:] = np.sum(wgt_tmp[None, :, :, None, None] * exp_terms[:, :, None, :, :], axis=1)

    def compute_lambda_phi_Qmat_exp(self):

        if not hasattr(self, "u_old"):
            # make some space for the old value of u[0]
            self.u_old = self.level.prob.dtype_u(init=self.level.prob.init, val=0.0)

        # everything that is computed in this if statement depends on u[0] only
        # To save computations we recompute that only if u[0] has changed.
        # Also, we check only for the first component u[0][0] of u[0] to save more computations.
        # Remember that u[0][0] is a vector representing the electric potential on the whole mesh and is enough to check if the whole u[0] has changed.
        if not np.allclose(self.u_old[0], self.level.u[0][0], rtol=1e-10, atol=1e-10):

            self.u_old[:] = self.level.u[0]

            L = self.level
            P = L.prob
            M = self.coll.num_nodes
            c = self.coll.nodes

            # compute lambda(u) of the exponential term f_exp(u)=lmbda(u)*(u-y_inf(u))
            # and select only the indeces with exponential terms (others are zeros)
            self.lmbda = P.lmbda_eval(L.u[0], L.time)[P.rhs_exp_indeces]

            if not hasattr(self, "phi"):
                # make some space
                self.phi = P.dtype_u(init=P.init_exp_extruded((M, M)), val=0.0)
                self.phi_one = P.dtype_u(init=P.init_exp_extruded((M, 1)), val=0.0)

            # evaluate the phi_k(dt*c_i*lambda) functions at the collocation nodes c_i for k=1,...,M
            self.phi_eval(L.dt * c, list(range(1, M + 1)), self.phi, self.lmbda)
            # evaluates phi_1(dt*delta_i*lambda) for delta_i = c_i - c_{i-1}
            self.phi_eval(L.dt * self.delta, [1], self.phi_one, self.lmbda)

            # compute weight for the integration of \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr,
            # where PiQ(r) is a polynomial interpolating some nodal values Q(c_i)=Q[i].
            # The integral of PiQ will be approximated as:
            # \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr ~= \sum_{j=0}^{M-1} Qmat_exp[i,j]*Q[j]

            k = np.arange(0, M)
            wgt_tmp = self.w[None, :, :] * c[:, None, None] ** (k[None, None, :] + 1)
            self.Qmat_exp = np.sum(wgt_tmp[:, :, :, None, None] * self.phi[:, None, :, :, :], axis=2)

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
            self.Q = P.dtype_u(init=P.init_exp_extruded((M,)), val=0.0)

        for k in range(M):
            self.Q[k][:] = L.f[k + 1].exp[P.rhs_exp_indeces] + self.lmbda * (
                L.u[0][P.rhs_exp_indeces] - L.u[k + 1][P.rhs_exp_indeces]
            )

        # integrate RHS over all collocation nodes
        me = [P.dtype_u(init=P.init, val=0.0) for _ in range(M)]
        for m in range(1, M + 1):
            for j in range(1, M + 1):
                me[m - 1][P.rhs_stiff_indeces] += self.coll.Qmat[m, j] * L.f[j].impl[P.rhs_stiff_indeces]
                me[m - 1][P.rhs_nonstiff_indeces] += self.coll.Qmat[m, j] * L.f[j].expl[P.rhs_nonstiff_indeces]
            me[m - 1][P.rhs_exp_indeces] += np.sum(self.Qmat_exp[m - 1] * self.Q, axis=0)

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

        # prepare the integral term
        for m in range(M):
            integral[m][P.rhs_stiff_indeces] += -L.dt * self.delta[m] * L.f[m + 1].impl[P.rhs_stiff_indeces]
            integral[m][P.rhs_nonstiff_indeces] += -L.dt * self.delta[m] * L.f[m].expl[P.rhs_nonstiff_indeces]
            integral[m][P.rhs_exp_indeces] += (
                -L.dt
                * self.delta[m]
                * self.phi_one[m][0]
                * (L.f[m].exp[P.rhs_exp_indeces] + self.lmbda * (L.u[0][P.rhs_exp_indeces] - L.u[m][P.rhs_exp_indeces]))
            )

        # do the sweep
        for m in range(M):

            tmp = L.u[m] + integral[m]
            tmp[P.rhs_exp_indeces] += (
                L.dt
                * self.delta[m]
                * self.phi_one[m][0]
                * (L.f[m].exp[P.rhs_exp_indeces] + self.lmbda * (L.u[0][P.rhs_exp_indeces] - L.u[m][P.rhs_exp_indeces]))
            )
            tmp[P.rhs_nonstiff_indeces] += L.dt * self.delta[m] * L.f[m].expl[P.rhs_nonstiff_indeces]

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                tmp, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m], L.u[m + 1]
            )

            # update function values
            P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m], fh=L.f[m + 1])

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

    def rel_norm(self, a, b):
        norms = []
        for i in range(len(a)):
            norms.append(np.linalg.norm(a[i]) / np.linalg.norm(b[i]))
        return np.average(norms)

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
            rel_res_norm.append(self.rel_norm(res[m], L.u[0]))

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
