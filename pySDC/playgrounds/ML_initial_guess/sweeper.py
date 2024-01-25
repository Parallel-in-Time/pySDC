from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class GenericImplicitML_IG(generic_implicit):
    def predict(self):
        """
        Initialise node with  machine learning initial guess
        """
        if self.params.initial_guess != 'NN':
            return super().predict()

        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        for m in range(1, self.coll.num_nodes + 1):
            L.u[m] = P.ML_predict(L.u[0], L.time, L.dt * self.coll.nodes[m - 1])
            L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True
