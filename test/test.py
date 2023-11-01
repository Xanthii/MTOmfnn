"""Add topopt to the context of any module that imports this module."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mtomfnn  # noqa

import numpy
import numpy as np
from mtomfnn.boundary_conditions import MBBBeamBoundaryConditions, CantileverBoundaryConditions
from mtomfnn.micro_structures import Crossed_2D
from mtomfnn.problems import ComplianceProblem
from mtomfnn.mfnn import *
from mtomfnn.guis import GUI
from mtomfnn.filters import SensitivityBasedFilter
from mtomfnn.solvers import OCSolver




nelx, nely = 60, 20  # Number of elements in the x and y
volfrac = 0.5  # Volume fraction for constraints
rmin = 2.4  # Filter radius


model_path = '../micro_data/2d_crossed/nn_model/'

bc = CantileverBoundaryConditions(nelx, nely)
micro = Crossed_2D(model_L=model_path + 'model_L.pth',
                   model_Linear=model_path + 'model_Linear.pth',
                   model_noLinear=model_path + 'model_noLinear.pth')

problem = ComplianceProblem(bc=bc, micro=micro)
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = SensitivityBasedFilter(nelx, nely, rmin)
solver = OCSolver(problem, volfrac, topopt_filter, gui, maxeval=100)
solver.optimize()

print(problem.CE.shape)
print(problem.CE.shape)
input("Press enter...")