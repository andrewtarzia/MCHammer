"""
This module defines general-purpose objects, functions and classes.

"""

from scipy.spatial.distance import euclidean
import numpy as np


def get_atom_distance(position_matrix, atom1_id, atom2_id):
    """
    Return the distance between two atoms.

    """

    return float(euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id]
    ))


def rotation_matrix_arbitrary_axis(angle, axis):
    """
    Returns a rotation matrix of `angle` radians about `axis`.

    Parameters
    ----------
    angle : :class:`float`
        The size of the rotation in radians.

    axis : :class:`numpy.ndarray`
        A 3 element aray which represents a vector. The vector is the
        axis about which the rotation is carried out. Must be of
        unit magnitude.

    Returns
    -------
    :class:`numpy.ndarray`
        A ``3x3`` array representing a rotation matrix.

    """

    a = np.cos(angle/2)
    b, c, d = axis * np.sin(angle/2)

    e11 = np.square(a) + np.square(b) - np.square(c) - np.square(d)
    e12 = 2*(b*c - a*d)
    e13 = 2*(b*d + a*c)

    e21 = 2*(b*c + a*d)
    e22 = np.square(a) + np.square(c) - np.square(b) - np.square(d)
    e23 = 2*(c*d - a*b)

    e31 = 2*(b*d - a*c)
    e32 = 2*(c*d + a*b)
    e33 = np.square(a) + np.square(d) - np.square(b) - np.square(c)

    return np.array([
        [e11, e12, e13],
        [e21, e22, e23],
        [e31, e32, e33]
    ])


def normalize_vector(vector):
    """
    Normalizes the given vector.

    A new vector is returned, the original vector is not modified.

    Parameters
    ----------
    vector : :class:`np.ndarray`
        The vector to be normalized.

    Returns
    -------
    :class:`np.ndarray`
        The normalized vector.

    """

    return np.divide(vector, np.linalg.norm(vector))
