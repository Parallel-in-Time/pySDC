#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem class for Dedalus
"""
from pySDC.core.problem import Problem

import numpy as np
from scipy.linalg import blas
from collections import deque

import dedalus.public as d3

from dedalus.core.system import CoeffSystem
from dedalus.core.evaluator import Evaluator
from dedalus.tools.array import apply_sparse
from dedalus.core.field import Field


# -----------------------------------------------------------------------------
# Tentative for a cleaner interface for pySDC
# -----------------------------------------------------------------------------
class FValues(object):

    def copy(self) -> "FValues":
        # TODO : return a copy of itself
        raise NotImplementedError

class Tendency(object):

    def __iadd__(self, f:FValues) -> "Tendency":
        # TODO : inplace addition with other full tendencies
        raise NotImplementedError

    def axpy(self, a:float, x:FValues|"Tendency") -> "Tendency":
        # Note : if a == 0, this should be a no-op ... or better, assert a != 0
        if isinstance(x, FValues):
            # TODO : y += a*x when x contains all tendencies
            pass
        if isinstance(x, Tendency):
            # TODO : y += a*x when x is only one tendency
            pass
        raise ValueError("wrong type for x")

class UValues(object):

    def __init__(self, init=None, val=0.0):
        # TODO : constructor requirements ?
        # -> Step.init_step : copy of initial tendency with u[0] = P.dtype_u(u0)
        if isinstance(init, UValues):
            pass
        else:
            pass


    def setValues(self, other:"UValues"):
        # TODO : copy other values into current instance
        raise NotImplementedError

    def toTendency(self) -> Tendency:
        """Convert into a Tendency instance"""
        raise NotImplementedError



class BaseProblem(Problem):

    dtype_u = UValues
    dtype_f = FValues

    def eval_f(self, u:UValues, t:float, f:FValues) -> FValues:
        # TODO : inplace evaluate f(u, t)
        raise NotImplementedError

    def solve_system(self, rhs:Tendency, dt:float, u:UValues, t:float) -> UValues:
        # TODO : inplace modify u with the system solve
        #   u + dt*f_I(u, t) = rhs
        #   using u as initial solution for an eventual iterative solver
        raise NotImplementedError

    def apply_mass_matrix(self, u:UValues) -> Tendency:
        """Apply eventual mass matrix on u to produce a tendency (default: no mass matrix)"""
        return u.toTendency()

# -----------------------------------------------------------------------------
# Dedalus specific implementation
# -----------------------------------------------------------------------------
class DedalusTendency(Tendency):

    def __init__(self):
        self.vals:CoeffSystem = None

    def __iadd__(self, f: "DedalusTendency"):
        self.vals.data += f.vals.data
        return self

    def axpy(self, a:float, x:"DedalusTendency"|"DedalusFValues") -> "DedalusTendency":
        if isinstance(x, DedalusTendency):
            self.vals.data += a*x.vals.data
        if isinstance(x, DedalusFValues):
            self.vals.data += a*(x.impl.vals.data + x.expl.vals.data)


class DedalusFValues(FValues):

    def __init__(self):
        self.impl:DedalusTendency = None
        self.expl:DedalusTendency = None


class DedalProblem(BaseProblem):

    dtype_u = UValues
    dtype_f = FValues

    def eval_f(self, u:UValues, t:float) -> FValues:
        # TODO : evaluate f(u, t)
        pass

    def solve_system(self, rhs:Tendency, dt:float, u:UValues, t:float) -> UValues:
        # TODO : inplace modify u with the system solve
        #   u + dt*f_I(u, t) = rhs
        #   using u as initial solution for an eventual iterative solver
        pass

    def apply_mass_matrix(self, u:UValues, rhs:Tendency) -> Tendency:
        # TODO : inplace evaluation in rhs of mass-matrix multiplication
        pass

# -----------------------------------------------------------------------------
# First interface
# -----------------------------------------------------------------------------
def State(cls, fields):
    return [f.copy() for f in fields]


class DedalusProblem(Problem):

    dtype_u = State
    dtype_f = State

    # Dummy class to trick Dedalus
    class DedalusTimeStepper:
        steps = 1
        stages = 1
        def __init__(self, solver):
            self.solver = solver

    def __init__(self, problem:d3.IVP, nNodes, collUpdate=False):

        self.DedalusTimeStepper.stages = nNodes
        solver = problem.build_solver(self.DedalusTimeStepper)
        self.solver = solver

        # From new version
        self.subproblems = [sp for sp in solver.subproblems if sp.size]
        self.evaluator:Evaluator = solver.evaluator
        self.F_fields = solver.F

        self.M = nNodes

        c = lambda: CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.MX0, self.RHS = c(), c()
        self.LX = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[c() for _ in range(self.M)] for _ in range(2)])

        # Attributes
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.dt = None
        self.firstEval = True
        self.init = True

        # Instantiate M solver, needed only for collocation update
        if collUpdate:
            for sp in solver.subproblems:
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                else:
                    sp.LHS = sp.M_min @ sp.pre_right
                sp.M_solver = solver.matsolver(sp.LHS, solver)
        self.collUpdate = collUpdate

    @property
    def state(self):
        return self.solver.state

    def stateCopy(self):
        return [f.copy() for f in self.solver.state]


    def computeMX0(self):
        """
        Compute MX0 term used in RHS of both initStep and sweep methods

        Update the MX0 attribute of the timestepper object.
        """
        MX0 = self.MX0
        state:list[Field] = self.solver.state

        # Require coefficient space
        for f in state:
            f.require_coeff_space()

        # Compute and store MX0
        for sp in self.subproblems:
            spX = sp.gather_inputs(state)
            apply_sparse(sp.M_min, spX, axis=0, out=MX0.get_subdata(sp))


    def updateLHS(self, dt, qI, init=False):
        """Update LHS and LHS solvers for each subproblem

        Parameters
        ----------
        dt : float
            Time-step for the updated LHS.
        qI : 2darray
            QDeltaI coefficients.
        init : bool, optional
            Wether or not initialize the LHS_solvers attribute for each
            subproblem. The default is False.
        """
        # Update only if different dt
        if self.dt == dt:
            return

        # Attribute references
        solver = self.solver

        # Update LHS and LHS solvers for each subproblems
        for sp in solver.subproblems:
            if self.init:
                # Instanciate list of solvers (ony first time step)
                sp.LHS_solvers = [None] * self.M
                self.init = False
            for i in range(self.M):
                if solver.store_expanded_matrices:
                    # sp.LHS.data[:] = sp.M_exp.data + k_Hii*sp.L_exp.data
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                    self.axpy(a=dt*qI[i, i], x=sp.L_exp.data, y=sp.LHS.data)
                else:
                    sp.LHS = (sp.M_min + dt*qI[i, i]*sp.L_min)  # CREATES TEMPORARY
                sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)


    def evalLX(self, LX):
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
        state:list[Field] = self.solver.state
        # Require coefficient space
        for f in state:
            f.require_coeff_space()

        # Evaluate matrix vector product and store
        for sp in self.solver.subproblems:
            spX = sp.gather_inputs(self.solver.state)
            apply_sparse(sp.L_min, spX, axis=0, out=LX.get_subdata(sp))


    def evalF(self, time, F):
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
        """
        solver = self.solver
        # Evaluate non linear term on current state
        t0 = solver.sim_time
        solver.sim_time = time
        if self.firstEval:
            solver.evaluator.evaluate_scheduled(
                sim_time=time, timestep=solver.dt,
                iteration=0, wall_time=0)
            self.firstEval = False
        else:
            solver.evaluator.evaluate_group('F')
        # Store F evaluation
        for sp in solver.subproblems:
            sp.gather_outputs(solver.F, out=F.get_subdata(sp))
        # Put back initial solver simulation time
        solver.sim_time = t0

    def solveAndStoreState(self, m):
        """
        Solve LHS * X = RHS using the LHS associated to a given node,
        and store X into the solver state.
        It uses the current RHS attribute of the object.

        Parameters
        ----------
        m : int
            Index of the nodes.
        """
        # Attribute references
        solver = self.solver
        RHS = self.RHS

        for field in solver.state:
            field.preset_layout('c')

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)
            spX = sp.LHS_solvers[m].solve(spRHS)  # CREATES TEMPORARY
            sp.scatter_inputs(spX, solver.state)
