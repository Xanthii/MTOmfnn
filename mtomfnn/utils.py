# -*- coding: utf-8 -*-

import os
import typing
import re

import numpy
import scipy
import scipy.sparse
import scipy.io
from matplotlib import colors


def xy_to_id(x: int, y: int, nelx: int, nely: int, order: str = "F") -> int:
    """
    Map from 2D indices of a node to the flattened 1D index.

    The number of elements is (nelx x nely), and the number of nodes is
    (nelx + 1) x (nely + 1).

    Parameters
    ----------
    x:
        The x-coordinate of the node's positions.
    y:
        The y-coordinate of the node's positions.
    nelx:
        The number of elements in the x-direction.
    nely:
        The number of elements in the y-direction.
    order:
        The order of indecies. "F" for Fortran/column-major order and "C" for
        C/row-major order.

    Returns
    -------
        The index of the node in the flattened version.

    """
    if order == "C":
        return (y * (nelx + 1)) + x
    else:
        return (x * (nely + 1)) + y


def id_to_xy(index: int, nelx: int, nely: int, order: str = "F"
             ) -> typing.Tuple[int, int]:
    """
    Map from a 1D index to 2D indices of a node.

    The number of elements is (nelx x nely), and the number of nodes is
    (nelx + 1) x (nely + 1).

    Parameters
    ----------
    index:
        The 1D index to map to a 2D location.
    nelx:
        The number of elements in the x-direction.
    nely:
        The number of elements in the y-direction.
    order:
        The order of indecies. "F" for Fortran/column-major order and "C" for
        C/row-major order.

    Returns
    -------
        The index of the node in the flattened version.

    """
    if order == "C":
        y = index // (nelx + 1)
        x = index % (nelx + 1)
    else:
        x = index // (nely + 1)
        y = index % (nely + 1)
    return x, y


def deleterowcol(A: scipy.sparse.csc_matrix, delrow: numpy.ndarray,
                 delcol: numpy.ndarray) -> scipy.sparse.csc_matrix:
    """
    Delete the specified rows and columns from csc sparse matrix A.

    Assumes that matrix is in symmetric csc form!

    Parameters
    ----------
    A:
        Matrix
    delrow:
        Row indices to remove.
    delcol:
        Column indices to remove.

    Returns
    -------
        Matrix with rows and columns removed.

    """
    m = A.shape[0]
    keep = numpy.delete(numpy.arange(0, m), delrow)
    A = A[keep, :]
    keep = numpy.delete(numpy.arange(0, m), delcol)
    A = A[:, keep]
    return A


def squared_euclidean(x: numpy.ndarray) -> float:
    """
    Compute the squared euclidean length of x.

    Parameters
    ----------
    x:
        Vector to compute the squared norm of.

    Returns
    -------
        Squared norm of x = :math:`x^Tx`

    """
    return x.T.dot(x)