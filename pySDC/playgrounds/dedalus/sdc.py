#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDC time-integrator for Dedalus
"""
# Python imports
import numpy as np
from scipy.linalg import blas
from collections import deque

# Dedalus import
from dedalus.core.system import CoeffSystem
from dedalus.tools.array import apply_sparse, csr_matvecs

from sdc_core import IMEXSDCCore


class SpectralDeferredCorrectionIMEX(IMEXSDCCore):

    steps = 1

    # -------------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------------
    def __init__(self, solver):

        # Store class attributes as instance attributes
        self.infos = self.getInfos()

        # Store solver as attribute
        self.solver = solver
        self.subproblems = [sp for sp in solver.subproblems if sp.size]
        self.stages = self.M    # need this for solver.log_stats()

        # Create coefficient systems for steps history
        c = lambda: CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.MX0, self.RHS = c(), c()
        self.LX = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[c() for _ in range(self.M)] for _ in range(2)])

        if not self.leftIsNode and self.initSweep == "QDelta":
            self.F0, self.LX0 = c(), c()

        # Attributes
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.dt = None
        self.firstEval = True

    @property
    def M(self):
        return len(self.nodes)

    @property
    def rightIsNode(self):
        return np.allclose(self.nodes[-1], 1.0)

    @property
    def leftIsNode(self):
        return np.allclose(self.nodes[0], 0.0)

    @property
    def doProlongation(self):
        return not self.rightIsNode or self.forceProl

    def _computeMX0(self, MX0):
        """
        Compute MX0 term used in RHS of both initStep and sweep methods

        Update the MX0 attribute of the timestepper object.
        """
        state = self.solver.state
        # Assert coefficient space
        self._requireStateCoeffSpace(state)
        # Compute and store MX0
        MX0.data.fill(0)
        for sp in self.subproblems:
            spX = sp.gather_inputs(state)
            apply_sparse(sp.M_min, spX, axis=0, out=MX0.get_subdata(sp))

    def _updateLHS(self, dt, init=False):
        """Update LHS and LHS solvers for each subproblem

        Parameters
        ----------
        dt : float
            Time-step for the updated LHS.
        init : bool, optional
            Wether or not initialize the LHS_solvers attribute for each
            subproblem. The default is False.
        """
        # Attribute references
        qI = self.QDeltaI
        solver = self.solver

        # Update LHS and LHS solvers for each subproblems
        for sp in solver.subproblems:
            if init:
                # Eventually instantiate list of solver (ony first time step)
                sp.LHS_solvers = [[None for _ in range(self.M)] for _ in range(self.nSweeps)]
            for k in range(self.nSweeps):
                for m in range(self.M):
                    if solver.store_expanded_matrices:
                        raise NotImplementedError("code correction required")
                        np.copyto(sp.LHS.data, sp.M_exp.data)
                        self.axpy(a=dt*qI[k, m, m], x=sp.L_exp.data, y=sp.LHS.data)
                    else:
                        sp.LHS = (sp.M_min + dt*qI[k, m, m]*sp.L_min)
                    sp.LHS_solvers[k][m] = solver.matsolver(sp.LHS, solver)
            if self.initSweep == "QDELTA":
                raise NotImplementedError()

    def _evalLX(self, LX):
        """
        Evaluate LX using the current state, and store it

        Parameters
        ----------
        LX : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.

        Returns
        -------
        None.

        """
        # Attribute references
        solver = self.solver
        # Assert coefficient spacec
        self._requireStateCoeffSpace(solver.state)
        # Evaluate matrix vector product and store
        for sp in solver.subproblems:
            spX = sp.gather_inputs(solver.state)
            apply_sparse(sp.L_min, spX, axis=0, out=LX.get_subdata(sp))

    def _evalF(self, F, time, dt, wall_time):
        """
        Evaluate the F operator from the current solver state

        Note
        ----
        After evaluation, state fields are left in grid space

        Parameters
        ----------
        time : float
            Time of evaluation.
        F : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.
        dt : float
            Current time step.
        wall_time : float
            Current wall time.
        """

        solver = self.solver
        # Evaluate non linear term on current state
        t0 = solver.sim_time
        solver.sim_time = time
        if self.firstEval:
            solver.evaluator.evaluate_scheduled(
                wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
            self.firstEval = False
        else:
            solver.evaluator.evaluate_group('F')
        # Store F evaluation
        for sp in solver.subproblems:
            sp.gather_outputs(solver.F, out=F.get_subdata(sp))
        # Put back initial solver simulation time
        solver.sim_time = t0

    def _solveAndStoreState(self, k, m):
        """
        Solve LHS * X = RHS using the LHS associated to a given node,
        and store X into the solver state.
        It uses the current RHS attribute of the object.

        Parameters
        ----------
        k : int
            Sweep index (0 for the first sweep).
        m : int
            Index of the nodes.
        """
        # Attribute references
        solver = self.solver
        RHS = self.RHS

        self._presetStateCoeffSpace(solver.state)

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)
            spX = sp.LHS_solvers[k][m].solve(spRHS)  # CREATES TEMPORARY
            sp.scatter_inputs(spX, solver.state)

    def _requireStateCoeffSpace(self, state):
        """Transform current state fields in coefficient space.
        If already in coefficient space, doesn't do anything."""
        for field in state:
            field.require_coeff_space()

    def _presetStateCoeffSpace(self, state):
        """Allow to write fields in coefficient space into current state
        fields, without transforming current state in coefficient space."""
        for field in state:
            field.preset_layout('c')

    def _initSweep(self):
        """
        Initialize node terms for one given time-step

        Parameters
        ----------
        iType : str, optional
            Type of initialization, can be :
            - iType="QDELTA" : use QDelta[I,E] for coarse time integration.
            - iType="COPY" : just copy the values from the initial solution.
            - iType="FNO" : use an FNO-ML model to predict values (incoming ...)
        """
        # Attribute references
        tau, qI, qE = self.nodes, self.QDeltaI, self.QDeltaE
        solver = self.solver
        t0, dt, wall_time = solver.sim_time, self.dt, self.wall_time
        RHS, MX0, Fk, LXk = self.RHS, self.MX0, self.F[0], self.LX[0]
        if not self.leftIsNode and self.initSweep == "QDELTA":
            F0, LX0 = self.F0, self.LX0
        axpy = self.axpy

        if self.initSweep == 'QDELTA':

            # Eventual initial field evaluation
            if not self.leftIsNode:
                if np.any(self.dtauE):
                    self._evalF(F0, t0, dt, wall_time)
                if np.any(self.dtauI):
                    self._evalLX(LX0)

            # Loop on all quadrature nodes
            for m in range(self.M):

                # Build RHS
                if RHS.data.size:

                    # Initialize with MX0 term
                    np.copyto(RHS.data, MX0.data)

                    # Add eventual initial field evaluation
                    if not self.leftIsNode:
                        if self.dtauE[m]:
                            axpy(a=dt*self.dtauE[m], x=F0.data, y=RHS.data)
                        if self.dtauI[m]:
                            axpy(a=-dt*self.dtauI[m], x=LX0.data, y=RHS.data)

                    # Add F and LX terms (already computed)
                    for i in range(m):
                        axpy(a=dt*qE[m, i], x=Fk[i].data, y=RHS.data)
                        axpy(a=-dt*qI[m, i], x=LXk[i].data, y=RHS.data)

                # Solve system and store node solution in solver state
                self._solveAndStoreState(m)

                # Evaluate and store LX with current state
                self._evalLX(LXk[m])

                # Evaluate and store F(X, t) with current state
                self._evalF(Fk[m], t0+dt*tau[m], dt, wall_time)

        elif self.initSweep == 'COPY':
            self._evalLX(LXk[0])
            self._evalF(Fk[0], t0, dt, wall_time)
            for m in range(1, self.M):
                np.copyto(LXk[m].data, LXk[0].data)
                np.copyto(Fk[m].data, Fk[0].data)

        else:
            raise NotImplementedError(f'initSweep={self.initSweep}')

    def _sweep(self, k):
        """Perform a sweep for the current time-step"""
        # Attribute references
        tau, qI, qE, q = self.nodes, self.QDeltaI, self.QDeltaE, self.Q
        solver = self.solver
        t0, dt, wall_time = solver.sim_time, self.dt, self.wall_time
        RHS, MX0 = self.RHS, self.MX0
        Fk, LXk, Fk1, LXk1 = self.F[0], self.LX[0], self.F[1], self.LX[1]
        axpy = self.axpy

        # Loop on all quadrature nodes
        for m in range(self.M):

            # Build RHS
            if RHS.data.size:

                # Initialize with MX0 term
                np.copyto(RHS.data, MX0.data)

                # Add quadrature terms
                for i in range(self.M):
                    axpy(a=dt*q[m, i], x=Fk[i].data, y=RHS.data)
                    axpy(a=-dt*q[m, i], x=LXk[i].data, y=RHS.data)

                if not self.diagonal:
                    # Add F and LX terms from iteration k+1 and previous nodes
                    for i in range(m):
                        axpy(a=dt*qE[k, m, i], x=Fk1[i].data, y=RHS.data)
                        axpy(a=-dt*qI[k, m, i], x=LXk1[i].data, y=RHS.data)
                    # Add F and LX terms from iteration k and previous nodes
                    for i in range(m):
                        axpy(a=-dt*qE[k, m, i], x=Fk[i].data, y=RHS.data)
                        axpy(a=dt*qI[k, m, i], x=LXk[i].data, y=RHS.data)

                # Add LX terms from iteration k+1 and current nodes
                axpy(a=dt*qI[k, m, m], x=LXk[m].data, y=RHS.data)

            # Solve system and store node solution in solver state
            self._solveAndStoreState(k, m)

            # Avoid non necessary RHS evaluations work
            if not self.forceProl and k == self.nSweeps-1:
                if self.diagonal:
                    continue
                elif m == self.M-1:
                    continue

            # Evaluate and store LX with current state
            self._evalLX(LXk1[m])
            # Evaluate and store F(X, t) with current state
            self._evalF(Fk1[m], t0+dt*tau[m], dt, wall_time)

        # Inverse position for iterate k and k+1 in storage
        # ie making the new evaluation the old for next iteration
        self.F.rotate()
        self.LX.rotate()

    def _prolongation(self):
        """Compute prolongation (needed if last node != 1)"""
        # Attribute references
        solver = self.solver
        w, dt = self.weights, self.dt
        RHS, MX0, Fk, LXk = self.RHS, self.MX0, self.F[0], self.LX[0]
        axpy = self.axpy

        # Build RHS
        if RHS.data.size:
            # Initialize with MX0 term
            np.copyto(RHS.data, MX0.data)
            # Add quadrature terms
            for i in range(self.M):
                axpy(a=dt*w[i], x=Fk[i].data, y=RHS.data)
                axpy(a=-dt*w[i], x=LXk[i].data, y=RHS.data)

        self._presetStateCoeffSpace(solver.state)

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            # Solve using LHS of the node
            spX = sp.M_solver.solve(spRHS)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, solver.state)

    def step(self, dt, wall_time):
        """
        Compute the next time-step solution using the time-stepper method,
        and modify to state field of the solver

        Note
        ----
        State fields should be left in grid space after at the end of the step.

        Parameters
        ----------
        dt : float
            Lenght of the current time-step.
        wall_time : float
            Current wall time for the simulation.
        """
        self.wall_time = wall_time

        # Initialize and/or update LHS terms, depending on dt
        if dt != self.dt:
            self._updateLHS(dt, init=self.dt is None)
            self.dt = dt

        # Compute MX0 for the whole time step
        self._computeMX0(self.MX0)

        # Initialize node values
        self._initSweep()

        # Performs sweeps
        for k in range(self.nSweeps):
            self._sweep(k)

        # Compute prolongation if needed
        if self.doProlongation:
            self._prolongation()

        # Update simulation time and reset evaluation tag
        self.solver.sim_time += dt
        self.firstEval = True
