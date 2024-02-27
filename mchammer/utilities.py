"""This module defines general-purpose objects, functions and classes."""

import numpy as np
from scipy.spatial.distance import euclidean


def get_atom_distance(
    position_matrix: np.ndarray,
    atom1_id: int,
    atom2_id: int,
) -> float:
    """Return the distance between two atoms."""
    return float(
        euclidean(u=position_matrix[atom1_id], v=position_matrix[atom2_id])
    )
