"""The heat equation with an inexact node-local solve, for the delivered-accuracy measurements."""

from pySDC.projects.DeltaSDC.inexact import inexact
from pySDC.projects.DeltaSDC.problems import heat_delta

heat_inexact = inexact(heat_delta)
