# -*- coding: utf-8 -*-

"""Graphics user interfaces for topology optimization."""
from __future__ import division

from matplotlib import colors
import matplotlib.cm as colormaps
import matplotlib.pyplot as plt
import numpy

from .utils import id_to_xy


class GUI(object):
    """
    Graphics user interface of the topology optimization.

    Draws the outputs a topology optimization problem.
    """

    def __init__(self, problem, title=""):
        """
        Create a plot and draw the initial design.

        Args:
            problem (topopt.Problem): problem to visualize
            title (str): title of the plot
        """
        self.problem = problem
        self.title = title
        plt.ion()  # Ensure that redrawing is possible
        self.init_subplots()
        plt.xlabel(title)
        #self.fig.tight_layout()
        self.plot_force_arrows()
        self.fig.show()

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the solver."""
        return str(self)

    def __repr__(self):
        """Create a representation of the solver."""
        return '{}(problem={!r}, title="{}")'.format(
                self.__class__.__name__, self.problem, self.title)

    def init_subplots(self):
        """Create the subplots."""
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            -numpy.zeros((self.problem.nely, self.problem.nelx)), cmap='gray',
            interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))

    def plot_force_arrows(self):
        """Add arrows to the plot for each force."""
        arrowprops = {"arrowstyle": "->", "connectionstyle": "arc3", "lw": 2,
                      "color": 0}
        nelx, nely, f = (self.problem.nelx, self.problem.nely, self.problem.f)
        cmap = plt.get_cmap("hsv", f.shape[1] + 1)
        for load_i in range(f.shape[1]):
            nz = numpy.nonzero(f[:, load_i])
            arrowprops["color"] = cmap(load_i)
            for i in range(nz[0].shape[0]):
                x, y = id_to_xy(nz[0][i] // 2, nelx, nely)
                x = max(min(x, nelx - 1), 0)
                y = max(min(y, nely - 1), 0)
                z = int(nz[0][i] % 2)
                mag = -50 * f[nz[0][i], load_i]
                self.ax.annotate(
                    "", xy=(x, y), xycoords="data",
                    xytext=(0 if z else mag, mag if z else 0),
                    textcoords="offset points",arrowprops=arrowprops)
             

    def update(self, xPhys, title=None):
        """Plot the results."""
        self.im.set_array(
            -xPhys.reshape((self.problem.nelx, self.problem.nely)).T)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if title is not None:
            plt.title(title)
        plt.pause(0.01)


