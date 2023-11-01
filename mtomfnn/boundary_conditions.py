# -*- coding: utf-8 -*-

"""Boundary conditions for topology optimization (forces and fixed nodes)."""

# Import standard library
from abc import ABC, abstractmethod

# Import modules
import numpy

# Import TopOpt modules
from .utils import xy_to_id


class BoundaryConditions(ABC):
    """
    Abstract class for boundary conditions to a topology optimization problem.

    Functionalty for geting fixed nodes, forces, and passive elements.

    Attributes
    ----------
    nelx: int
        The number of elements in the x direction.
    nely: int
        The number of elements in the y direction.

    """

    def __init__(self, nelx: int, nely: int):
        """
        Create the boundary conditions with the size of the grid.

        Parameters
        ----------
        nelx:
            The number of elements in the x direction.
        nely:
            The number of elements in the y direction.

        """
        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)

    def __str__(self) -> str:
        """Construct a string representation of the boundary conditions."""
        return self.__class__.__name__

    def __format__(self, format_spec) -> str:
        """Construct a formated representation of the boundary conditions."""
        return str(self)

    def __repr__(self) -> str:
        """Construct a representation of the boundary conditions."""
        return "{}(nelx={:d}, nely={:d})".format(
            self.__class__.__name__, self.nelx, self.nely)

    @property
    @abstractmethod
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes of the problem."""
        pass

    @property
    @abstractmethod
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector for the problem."""
        pass

    @property
    def passive_elements(self):
        """:obj:`numpy.ndarray`: Passive elements to be set to zero density."""
        return numpy.array([])

    @property
    def active_elements(self):
        """:obj:`numpy.ndarray`: Active elements to be set to full density."""
        return numpy.array([])


class MBBBeamBoundaryConditions(BoundaryConditions):
    """Boundary conditions for the Messerschmitt–Bölkow–Blohm (MBB) beam."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes in the bottom corners."""
        dofs = numpy.arange(self.ndof)
        fixed = numpy.union1d(dofs[0:2 * (self.nely + 1):2], numpy.array(
            [2 * (self.nelx + 1) * (self.nely + 1) - 1]))
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector in the top center."""
        f = numpy.zeros((self.ndof, 1))
        f[1, 0] = -1
        return f


class CantileverBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a cantilever."""

    @property
    def fixed_nodes(self):
        """:obj:`numpy.ndarray`: Fixed nodes on the left."""
        ys = numpy.arange(self.nely + 1)
        lefty_to_id = numpy.vectorize(
            lambda y: xy_to_id(0, y, self.nelx, self.nely))
        ids = lefty_to_id(ys)
        fixed = numpy.union1d(2 * ids, 2 * ids + 1)  # Fix both x and y dof
        return fixed

    @property
    def forces(self):
        """:obj:`numpy.ndarray`: Force vector in the middle right."""
        #  This Action point position is the midle point of free edge
        # f = numpy.zeros((self.ndof, 1))
        # dof_index = 2 * xy_to_id(
        #     self.nelx, self.nely // 2, self.nelx, self.nely) + 1
        # f[dof_index, 0] = -1
        # return f
        
        f = numpy.zeros((self.ndof, 1))
        
        f[-1,:] = -1
        return f