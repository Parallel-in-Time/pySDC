import os
import xml.etree.ElementTree as ET

import firedrake as fd
import numpy as np
from firedrake.petsc import PETSc

from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake

#: Boundary ids of the facets of :func:`cylinder_mesh`
INFLOW, OUTFLOW, WALLS, CYLINDER = 1, 2, 3, 4


def cylinder_mesh(comm=fd.COMM_WORLD):
    """
    The mesh of the channel with a cylinder, read from ``meshs/cylinder.xml``, the DOLFIN XML file the FEniCS version
    uses, so that both discretize the same mesh. Firedrake cannot read that format, so the vertices and triangles are
    put into a PETSc DMPlex, and the boundary facets are labelled with the geometric predicates of the FEniCS version:
    the inflow at x = 0, the outflow at x = 2.2, the walls at y = 0 and y = 0.41, and the cylinder for the rest.

    Args:
        comm (mpi4py.MPI.Intracomm): Communicator to distribute the mesh over

    Returns:
        firedrake.MeshGeometry: The mesh, with the boundary ids `INFLOW`, `OUTFLOW`, `WALLS` and `CYLINDER`
    """
    if comm.rank == 0:
        mesh = ET.parse(f'{os.path.dirname(__file__)}/../meshs/cylinder.xml').getroot().find('mesh')
        coords = np.array([[float(v.get('x')), float(v.get('y'))] for v in mesh.find('vertices')])
        cells = np.array([[int(c.get(key)) for key in ('v0', 'v1', 'v2')] for c in mesh.find('cells')], dtype=np.int32)
    else:
        coords, cells = np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)

    plex = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=comm)
    plex.markBoundaryFaces('boundary_faces', 1)
    plex.createLabel('Face Sets')
    if plex.getStratumSize('boundary_faces', 1) > 0:
        for face in plex.getStratumIS('boundary_faces', 1).getIndices():
            x, y = plex.computeCellGeometryFVM(face)[1]
            if np.isclose(x, 0):
                label = INFLOW
            elif np.isclose(x, 2.2):
                label = OUTFLOW
            elif np.isclose(y, 0) or np.isclose(y, 0.41):
                label = WALLS
            else:
                label = CYLINDER
            plex.setLabelValue('Face Sets', face, label)
    plex.removeLabel('boundary_faces')
    return fd.Mesh(plex, comm=comm)


class NavierStokes2DMonolithicFiredrake(FormFiredrake):
    r"""
    Incompressible Navier-Stokes flow around a cylinder (the DFG benchmark 2D-3), monolithic, with Firedrake and
    given by its weak forms. This is the port of
    :class:`~pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_FEniCS.fenics_NSE_2D_Monolithic`,
    on the same mesh.

    The velocity :math:`u` and the pressure :math:`p` solve

    .. math::
        \frac{\partial u}{\partial t} = - u \cdot \nabla u + \nu \Delta u - \nabla p, \quad 0 = \nabla \cdot u,

    with no-slip on the walls and the cylinder, the inflow profile
    :math:`u = (6 \sin(\pi t / 8)\, y (0.41 - y) / 0.41^2, 0)` at :math:`x = 0` and :math:`p = 0` at the outflow. With
    Taylor-Hood elements (degree `order` for :math:`u`, one lower for :math:`p`) and test functions :math:`v, q`, the
    weak forms are

    .. math::
        \int u_t \cdot v\,dx = -\int (u \cdot \nabla u) \cdot v\,dx - \nu \int \nabla u : \nabla v\,dx
        + \int p\, \nabla \cdot v\,dx + \int (\nabla \cdot u)\, q\,dx.

    The mass form has no pressure part, so the mass matrix is singular and the incompressibility constraint is an
    algebraic equation. The mass-matrix sweepers handle that, since they never invert the mass matrix. Everything else
    comes from :class:`~pySDC.implementations.problem_classes.FormFiredrake.FormFiredrake`.

    Parameters
    ----------
    nu : float, optional
        Kinematic viscosity :math:`\nu`.
    order : int, optional
        Degree of the velocity elements.
    mesh : firedrake.MeshGeometry, optional
        The mesh, by default :func:`cylinder_mesh`.
    solver_parameters : dict, optional
        PETSc options for the nonlinear solves, by default Newton with tolerances like those of the FEniCS version.
    LHS_cache_size : int, optional
        Number of solvers kept, see ``FormFiredrake``.
    """

    def __init__(self, nu=0.001, order=2, mesh=None, solver_parameters=None, LHS_cache_size=12):
        """Initialization, see the class docstring for the arguments."""
        mesh = cylinder_mesh() if mesh is None else mesh
        V = fd.VectorFunctionSpace(mesh, 'CG', order)
        Q = fd.FunctionSpace(mesh, 'CG', order - 1)
        W = V * Q

        def mass(w, z):
            u, _ = fd.split(w)
            v, _ = fd.split(z)
            return fd.inner(u, v) * fd.dx

        def rhs(w, z):
            u, p = fd.split(w)
            v, q = fd.split(z)
            return (
                -fd.inner(fd.dot(u, fd.nabla_grad(u)), v)
                - nu * fd.inner(fd.nabla_grad(u), fd.nabla_grad(v))
                + p * fd.div(v)
                + fd.div(u) * q
            ) * fd.dx

        t = fd.Constant(0.0)
        y = fd.SpatialCoordinate(mesh)[1]
        u_in = fd.as_vector([4.0 * 1.5 * fd.sin(fd.pi * t / 8) * y * (0.41 - y) / 0.41**2, 0.0])
        bcs = (
            fd.DirichletBC(W.sub(0), fd.Constant((0.0, 0.0)), (WALLS, CYLINDER)),
            fd.DirichletBC(W.sub(1), 0.0, OUTFLOW),
            fd.DirichletBC(W.sub(0), u_in, INFLOW),
        )
        if solver_parameters is None:
            solver_parameters = {'snes_atol': 1e-10, 'snes_rtol': 1e-9}

        super().__init__(W, mass, rhs, bcs=bcs, t=t, solver_parameters=solver_parameters, LHS_cache_size=LHS_cache_size)
        self._makeAttributeAndRegister('nu', 'order', localVars=locals(), readOnly=True)
        self.mesh = mesh

    def u_exact(self, t):
        """
        The initial value: fluid at rest, with the boundary values at time `t`. There is no exact solution.

        Args:
            t (float): Time of the boundary values

        Returns:
            dtype_u: The initial value
        """
        self.t.assign(t)
        me = self.u_init
        for bc in self.bcs:
            bc.apply(me.functionspace)
        return me

    def drag_lift(self, w):
        """
        Drag and lift coefficients of the cylinder, as in the DFG benchmark and the FEniCS version.

        Args:
            w (dtype_u): Velocity and pressure

        Returns:
            tuple of float: Drag and lift coefficients
        """
        u, p = fd.split(w.functionspace)
        n = -fd.FacetNormal(self.mesh)
        u_t = fd.inner(fd.as_vector((n[1], -n[0])), u)
        ds = fd.ds(CYLINDER)
        drag = 2 / 0.1 * (self.nu * fd.inner(fd.grad(u_t), n) * n[1] - p * n[0]) * ds
        lift = -2 / 0.1 * (self.nu * fd.inner(fd.grad(u_t), n) * n[0] + p * n[1]) * ds
        return fd.assemble(drag), fd.assemble(lift)
