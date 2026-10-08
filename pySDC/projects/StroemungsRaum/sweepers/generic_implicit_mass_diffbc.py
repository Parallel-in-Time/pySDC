from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass


class generic_implicit_mass_diffbc(generic_implicit_mass):
    """
    Variant of ``generic_implicit_mass`` for problems that impose their boundary conditions in
    differentiated form.

    Such a problem needs the collocation data of the current step to build the boundary values
    of its stages by quadrature, and a problem class cannot see that data on its own: it only
    ever learns the time of the node it is asked to solve at. This sweeper hands it over once
    per step, before sweeping.

    The problem class must provide ``prepare_step(t0, dt, coll)``.
    """

    def predict(self):
        """
        Supply the collocation data of this step to the problem, then predict as usual.

        ``predict`` rather than ``update_nodes`` because the controller calls it exactly once
        per step, while ``update_nodes`` runs once per sweep and would rebuild the same
        boundary conditions on every iteration.

        Returns:
            None
        """
        L = self.level
        L.prob.prepare_step(L.time, L.dt, self.coll)

        return super().predict()
