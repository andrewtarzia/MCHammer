"""
This module defines general-purpose objects, functions and classes.

"""

from scipy.spatial.distance import euclidean


def get_atom_distance(position_matrix, atom1_id, atom2_id):
    """
    Return the distance between two atoms.

    """

    return float(euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id]
    ))
