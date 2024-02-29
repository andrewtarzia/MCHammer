"""This module defines general-purpose objects, functions and classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.spatial.distance import euclidean

if TYPE_CHECKING:
    import numpy as np


def get_atom_distance(
    position_matrix: np.ndarray,
    atom1_id: int,
    atom2_id: int,
) -> float:
    """Return the distance between two atoms."""
    return float(
        euclidean(u=position_matrix[atom1_id], v=position_matrix[atom2_id])
    )


def get_bond_vector(
    position_matrix: np.ndarray,
    bond_pair: tuple[int, int],
) -> np.ndarray:
    """Get vector from atom1 to atom2 in bond."""
    atom1_pos = position_matrix[bond_pair[0]]
    atom2_pos = position_matrix[bond_pair[1]]
    return atom2_pos - atom1_pos
