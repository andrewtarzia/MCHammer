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


def vector_angle(vector1, vector2):
    """
    Returns the angle between two vectors in radians.

    Parameters
    ----------
    vector1 : :class:`numpy.ndarray`
        The first vector.

    vector2 : :class:`numpy.ndarray`
        The second vector.

    Returns
    -------
    :class:`float`
        The angle between `vector1` and `vector2` in radians.

    """

    if np.all(np.equal(vector1, vector2)):
        return 0.

    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    # This if statement prevents returns of NaN due to floating point
    # inaccuracy.
    term = numerator/denominator
    if term >= 1.:
        return 0.0
    if term <= -1.:
        return np.pi
    return np.arccos(term)


def get_angle(position_matrix, atom1_id, atom2_id, atom3_id):
    """
    Return the angle between three atoms.

    """
    atom1_pos = position_matrix[atom1_id]
    atom2_pos = position_matrix[atom2_id]
    atom3_pos = position_matrix[atom3_id]
    v1 = atom1_pos - atom2_pos
    v2 = atom3_pos - atom2_pos
    return vector_angle(v1, v2)
