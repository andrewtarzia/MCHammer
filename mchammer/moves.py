"""
Moves
=====

#. :class:`.Translation`
#. :class:`.Rotation`

"""

from copy import deepcopy

from .utilities import rotation_matrix_arbitrary_axis, rotation_matrix


class Translation:
    """
    Translation.

    """

    def __init__(self, vector, movable_atom_ids):
        """
        Initialize a :class:`Translation` instance.

        Parameters
        ----------
        vector : :class:`np.array`
            Translation vector.

        movable_atom_ids

        """

        self._vector = vector
        self._movable_atom_ids = movable_atom_ids

    def perform_move(self, mol):
        """
        Return `mol` with move performed.

        """

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - self._vector

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def reverse_move(self, mol):
        """
        Return `mol` with move performed.

        """

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos + self._vector

        mol = mol.with_position_matrix(new_position_matrix)
        return mol


class RotationAboutAxis:
    """
    RotationAboutAxis.

    """

    def __init__(self, angle, movable_atom_ids, axis, origin):
        """
        Initialize a :class:`Rotation` instance.

        Parameters
        ----------
        angle : :class:`np.array`
            Translation vector.

        movable_atom_ids

        axis

        origin

        """
        self._angle = angle
        self._movable_atom_ids = movable_atom_ids
        self._axis = axis
        self._origin = origin

    def perform_move(self, mol):
        """
        Return `mol` with move performed.

        """

        mod_position_matrix = deepcopy(mol.get_position_matrix())
        # Set the origin of the rotation to "origin".
        mod_position_matrix = mod_position_matrix - self._origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(
            self._angle, self._axis
        )
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        mod_position_matrix = (rot_mat @ mod_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        mod_position_matrix = mod_position_matrix + self._origin

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            new_position_matrix[atom.get_id()] = (
                mod_position_matrix[atom.get_id()]
            )

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def reverse_move(self, mol):
        """
        Return `mol` with move performed.

        """

        mod_position_matrix = deepcopy(mol.get_position_matrix())
        # Set the origin of the rotation to "origin".
        mod_position_matrix = mod_position_matrix - self._origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(
            -self._angle, self._axis
        )
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        mod_position_matrix = (rot_mat @ mod_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        mod_position_matrix = mod_position_matrix + self._origin

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            new_position_matrix[atom.get_id()] = (
                mod_position_matrix[atom.get_id()]
            )

        mol = mol.with_position_matrix(new_position_matrix)
        return mol


class RotationBetweenVectors:
    """
    RotationBetweenVectors.

    """

    def __init__(self, start, target, movable_atom_ids, origin):
        """
        Initialize a :class:`Rotation` instance.

        Parameters
        ----------
        start : :class:`np.array`
            Translation vector.

        target

        movable_atom_ids


        origin

        """
        self._start = start
        self._target = target
        self._movable_atom_ids = movable_atom_ids
        self._origin = origin

    def perform_move(self, mol):
        """
        Return `mol` with move performed.

        """

        mod_position_matrix = deepcopy(mol.get_position_matrix())
        # Set the origin of the rotation to "origin".
        mod_position_matrix = mod_position_matrix - self._origin
        # Perform rotation.
        rot_mat = rotation_matrix(self._start, self._target)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        mod_position_matrix = (rot_mat @ mod_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        mod_position_matrix = mod_position_matrix + self._origin

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            new_position_matrix[atom.get_id()] = (
                mod_position_matrix[atom.get_id()]
            )

        mol = mol.with_position_matrix(new_position_matrix)
        return mol

    def reverse_move(self, mol):
        """
        Return `mol` with move performed.

        """
        mod_position_matrix = deepcopy(mol.get_position_matrix())
        # Set the origin of the rotation to "origin".
        mod_position_matrix = mod_position_matrix - self._origin
        # Perform rotation.
        rot_mat = rotation_matrix(self._target, self._start)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        mod_position_matrix = (rot_mat @ mod_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        mod_position_matrix = mod_position_matrix + self._origin

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in self._movable_atom_ids:
                continue
            new_position_matrix[atom.get_id()] = (
                mod_position_matrix[atom.get_id()]
            )

        mol = mol.with_position_matrix(new_position_matrix)
        return mol
