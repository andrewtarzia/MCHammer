"""
Moves
=====

#. :class:`.Translation`

"""

from copy import deepcopy


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
