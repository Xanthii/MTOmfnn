# -*- coding: utf-8 -*-

"""Add topopt to the context of any module that imports this module."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mtomfnn  # noqa

import numpy
from mtomfnn.boundary_conditions import MBBBeamBoundaryConditions, CantileverBoundaryConditions
from mtomfnn.micro_structures import SIMP
from mtomfnn.problems import ComplianceProblem
from mtomfnn.guis import GUI
from mtomfnn.filters import SensitivityBasedFilter, DensityBasedFilter
from mtomfnn.solvers import OCSolver





nelx, nely = 60, 20  # Number of elements in the x and y
volfrac = 0.3  # Volume fraction for constraints
rmin = 2.4  # Filter radius


model_path = '../micro_data/2d_crossed/nn_model/'

bc = MBBBeamBoundaryConditions(nelx, nely)
micro = SIMP()

problem = ComplianceProblem(bc=bc, micro=micro)

gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)
solver = OCSolver(problem, volfrac, topopt_filter, gui, maxeval=100)
solver.optimize()


input("Press enter...")