# -*- coding: utf-8 -*-

"""
Solvers to solve topology optimization problems.

Todo:
    * Make TopOptSolver an abstract class
    * Rename the current TopOptSolver to MMASolver(TopOptSolver)
    * Create a TopOptSolver using originality criterion
"""
from __future__ import division

import numpy
import abc

from mtomfnn.problems import Problem
from mtomfnn.filters import *
from mtomfnn.guis import GUI


class TopOptSolver(abc.ABC):
    """Solver for topology optimization problems using NLopt's MMA solver."""

    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, maxeval=200, ftol_rel=1e-2):
        """
        Create a solver to solve the problem.

        Parameters
        ----------
        problem: :obj:`topopt.problems.Problem`
            The topology optimization problem to solve.
        volfrac: float
            The maximum fraction of the volume to use.
        filter: :obj:`topopt.filters.Filter`
            A filter for the solutions to reduce artefacts.
        gui: :obj:`topopt.guis.GUI`
            The graphical user interface to visualize intermediate results.
        maxeval: int
            The maximum number of evaluations to perform.
        ftol: float
            A floating point tolerance for relative change.

        """
        self.problem = problem
        self.filter = filter
        self.gui = gui
        n = problem.nelx * problem.nely
        self.xPhys = numpy.ones(n) * volfrac

        # set stopping criteria
        self.maxeval = maxeval
        self.ftol_rel = ftol_rel

        self.volfrac = volfrac  # max volume fraction to use

        # setup filter
        self.passive = problem.bc.passive_elements
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        self.active = problem.bc.active_elements
        if self.active.size > 0:
            self.xPhys[self.active] = 1

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the solver."""
        return "{} with {}".format(str(self.problem), str(self))

    def __repr__(self):
        """Create a representation of the solver."""
        return ("{}(problem={!r}, volfrac={:g}, filter={!r}, ".format(
            self.__class__.__name__, self.problem, self.volfrac, self.filter)
            + "gui={!r}, maxeval={:d}, ftol={:g})".format(
                self.gui, self.opt.get_maxeval(), self.opt.get_ftol_rel()))

    @abc.abstractmethod
    def optimize(self, x: numpy.ndarray) -> None: 
        pass


class OCSolver(TopOptSolver):

    def OC(self, x: numpy.ndarray, dobj: numpy.ndarray, dv:numpy.ndarray):
        minVF = 0
        maxVF = 1.
        l1 = 0
        l2 = 1e9
        move = 0.2
        xnew = numpy.zeros_like(x)
        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)     
            xnew[:] = numpy.maximum(minVF, numpy.maximum(x - move, numpy.minimum(maxVF, 
                    numpy.minimum(x + move, x * numpy.sqrt(-dobj /dv/ lmid)))))

          ########## problematic
            self.filter.filter_variables(xnew, self.xPhys)
          ########## problematic
            if numpy.sum(self.xPhys) - self.volfrac * self.problem.nelx * self.problem.nely > 0:
                l1 = lmid
            else:
                l2 = lmid
       
        return xnew
    

    
    def optimize(self):
        it = 0
        rel_change = 1.
        xold = self.xPhys.copy()
        while rel_change > self.ftol_rel and it < self.maxeval:

            obj = self.problem.compute_objective(self.xPhys)    
            
            if isinstance(self.filter, SensitivityBasedFilter):
                xold = self.xPhys.copy()
                
                self.filter.filter_objective_sensitivities(xold, self.problem.dobj)
                self.OC(xold, self.problem.dobj, self.problem.dv)     
                rel_change = numpy.max(self.xPhys - xold)
                vol = numpy.mean(self.xPhys)
                self.gui.update(self.xPhys)

                it += 1
                print(f' It.:{it}, Obj.:{obj}, Vol.:{vol} ch.:{rel_change}')
                

                
            if isinstance(self.filter, DensityBasedFilter):
                self.filter.filter_objective_sensitivities(self.xPhys, self.problem.dobj)
                self.filter.filter_volume_sensitivities(self.xPhys, self.problem.dv)
  
                xnew = self.OC(xold, self.problem.dobj, self.problem.dv)
           
                rel_change = numpy.max(xnew - xold)
                vol = numpy.mean(self.xPhys)
                   
    
                self.gui.update(self.xPhys)
                xold[:] = xnew
                it += 1
                print(f' It.:{it}, Obj.:{obj}, Vol.:{vol} ch.:{rel_change}')

        
        

    
