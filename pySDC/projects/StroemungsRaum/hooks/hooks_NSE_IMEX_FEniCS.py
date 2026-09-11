from pySDC.core.hooks import Hooks
import dolfin as df
import os


class LogLiftDrag(Hooks):
    """
    Hook class to compute lift and drag forces on the cylinder at the end of each step,
    and add them to the statistics
    """

    def post_step(self, step, level_number):
        """
        Record lift and drag forces on the cylinder at the end of each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()
        P = L.prob

        rho = 1

        # normal pointing out of obstacle
        n = -df.FacetNormal(P.V.mesh())

        # tangential velocity component at the interface of the obstacle
        u_t = df.inner(df.as_vector((n[1], -n[0])), L.uend.values)

        # compute the drag and lift coefficients
        drag = df.Form(2 / 0.1 * (P.nu / rho * df.inner(df.grad(u_t), n) * n[1] - P.pn * n[0]) * P.dsc)
        lift = df.Form(-2 / 0.1 * (P.nu / rho * df.inner(df.grad(u_t), n) * n[0] + P.pn * n[1]) * P.dsc)

        # assemble the scalar values
        CD = df.assemble(drag)
        CL = df.assemble(lift)

        # add to statistics
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='lift_post_step',
            value=CL,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='drag_post_step',
            value=CD,
        )


class LogWriteSolutions(Hooks):
    """
    Hook class to log the velocity and prssure solutions and write them in XDMF files
    """

    def pre_run(self, step, level_number):
        """
        Default routine called before time-loop starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_run(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        # create XDMF file for visualization output
        path = f"{os.path.dirname(__file__)}/../data/navier_stokes/"
        P.xdmffile_p = df.XDMFFile(path + 'Cylinder_pressure.xdmf')
        P.xdmffile_u = df.XDMFFile(path + 'Cylinder_velocity.xdmf')

    def post_step(self, step, level_number):
        """
        Write the solutions after each time step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """

        super().post_step(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()
        P = L.prob
        t = L.time

        P.xdmffile_p.write_checkpoint(P.pn, "pn", t, df.XDMFFile.Encoding.HDF5, True)
        P.xdmffile_u.write_checkpoint(L.uend.values, "un", t, df.XDMFFile.Encoding.HDF5, True)

    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """

        super().post_run(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        P.xdmffile_p.close()
        P.xdmffile_u.close()
