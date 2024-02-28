"""Module of Monte Carlo operations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .molecule import Molecule


def get_bond_vector(
    position_matrix: np.ndarray,
    bond_pair: tuple[int, int],
) -> np.ndarray:
    """Get vector from atom1 to atom2 in bond."""
    atom1_pos = position_matrix[bond_pair[0]]
    atom2_pos = position_matrix[bond_pair[1]]
    return atom2_pos - atom1_pos


def translate_atoms_along_vector(
    mol: Molecule,
    atom_ids: tuple[int, ...],
    vector: np.ndarray,
) -> Molecule:
    """Translate atoms with given ids along a vector."""
    new_position_matrix = deepcopy(mol.get_position_matrix())
    for atom in mol.get_atoms():
        if atom.get_id() not in atom_ids:
            continue
        pos = mol.get_position_matrix()[atom.get_id()]
        new_position_matrix[atom.get_id()] = pos - vector

    return mol.with_position_matrix(new_position_matrix)


def translate_molecule_along_vector(
    mol: Molecule,
    vector: np.ndarray,
) -> Molecule:
    """Translate a whole molecule along a vector."""
    return mol.with_displacement(vector)


def test_move(
    beta: float,
    curr_pot: float,
    new_pot: float,
    generator: np.random.Generator,
) -> bool:
    """Test an MC move based on boltzmann energy."""
    if new_pot < curr_pot:
        return True

    exp_term = np.exp(-beta * (new_pot - curr_pot))
    rand_number = generator.random()

    return exp_term > rand_number


def rotation_matrix_arbitrary_axis(
    angle: float,
    axis: np.ndarray,
) -> np.ndarray:
    """Returns a rotation matrix of `angle` radians about `axis`.

    Parameters
    ----------
    angle : :class:`float`
        The size of the rotation in radians.
    axis : :class:`numpy.ndarray`
        A 3 element aray which represents a vector. The vector is the
        axis about which the rotation is carried out. Must be of
        unit magnitude.

    Returns:
    -------
    :class:`numpy.ndarray`
        A ``3x3`` array representing a rotation matrix.
    """
    a = np.cos(angle / 2)
    b, c, d = axis * np.sin(angle / 2)

    e11 = np.square(a) + np.square(b) - np.square(c) - np.square(d)
    e12 = 2 * (b * c - a * d)
    e13 = 2 * (b * d + a * c)

    e21 = 2 * (b * c + a * d)
    e22 = np.square(a) + np.square(c) - np.square(b) - np.square(d)
    e23 = 2 * (c * d - a * b)

    e31 = 2 * (b * d - a * c)
    e32 = 2 * (c * d + a * b)
    e33 = np.square(a) + np.square(d) - np.square(b) - np.square(c)

    # Initialize as a scipy Rotation object, which normalizes the
    # matrix and allows for returns as quaternion or alternative
    # type in the future.
    return Rotation.from_matrix(
        np.array([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])
    ).as_matrix()


def rotate_molecule_by_angle(
    mol: Molecule,
    angle: float,
    axis: np.ndarray,
    origin: np.ndarray,
) -> Molecule:
    """Rotate a molecule by an angle about an axis and origin."""
    new_position_matrix = mol.get_position_matrix()
    # Set the origin of the rotation to "origin".
    new_position_matrix = new_position_matrix - origin
    # Perform rotation.
    rot_mat = rotation_matrix_arbitrary_axis(angle, axis)
    # Apply the rotation matrix on the position matrix, to get the
    # new position matrix.
    new_position_matrix = (rot_mat @ new_position_matrix.T).T
    # Return the centroid of the molecule to the original position.
    new_position_matrix = new_position_matrix + origin

    return mol.with_position_matrix(new_position_matrix)
