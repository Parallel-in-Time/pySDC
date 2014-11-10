from pySDC.sweeper_classes.generic_LU import generic_LU
import numpy as np

class mod_LU_sweeper(generic_LU):

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here might be a simple copy from u[M] (if right point is a collocation node) or
        a full evaluation of the Picard formulation (if right point is not a collocation node)
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        if L.uend is not None:
            uend_old = P.dtype_u(L.uend)
        else:
            uend_old = None

        # check if Mth node is equal to right point (flag is set in collocation class)
        if self.coll.right_is_node:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            L.uend += self.integrate(self.coll.weights)
            #FIXME: do we need some sort of tau correction here as well?

        return uend_old


    def predict(self,S):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """


        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0],L.time)

        dt = L.dt
        accepted = False
        while not accepted:

            # copy u[0] to all collocation nodes, evaluate RHS
            for m in range(1,self.coll.num_nodes+1):
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m],L.time+L.dt*self.coll.nodes[m-1])

            # indicate that this level is now ready for sweeps
            L.status.unlocked = True

            L.sweep.update_nodes()
            L.sweep.compute_residual()

            pred_iter = np.ceil(np.log10(L.status.residual/1E-10))
            # print('Predicted niter: ',pred_iter)

            if pred_iter > 8:
                S.dt = S.dt/2
                print('Setting dt down to ',S.dt,S.time)
            elif pred_iter < 8:
                S.dt = 2*S.dt
                print('Setting dt up to ',S.dt, S.time)
            else:
                accepted = True


        dt_new = S.dt

        return dt_new