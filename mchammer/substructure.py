"""
Substructure
============

#. :class:`.Substructure`

"""

import numpy as np

from .moves import Translation


class Substructure:
    """
    Substructure.

    """

    def __init__(self, atom_ids, disconnectors, target):
        """
        Initialize a :class:`Substructure` instance.

        Parameters
        ----------
        atom_ids : :class:`tuple` of :class:`int`
            Atom ids in substructure.

        target : :class:`float`

        """

        self._atom_ids = atom_ids
        self._disconnectors = disconnectors
        self._target = target
        self._target_unit = None

    def get_atom_ids(self):
        """
        Get atom ID.

        """
        return self._atom_ids

    def get_target(self):
        """
        Get target.

        """
        return self._target

    def get_target_unit(self):
        """
        Get target unit.

        """
        return self._target_unit

    def get_disconnectors(self):
        """
        Get disconnectors.

        """
        return self._disconnectors

    def compute_potential(self):
        """
        Compute substructure potential energy.

        """
        raise NotImplementedError()

    def calculate_move(self):
        """
        Calculate possible move defined by substructure.

        """
        raise NotImplementedError()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(ids={self.get_atom_ids()}, '
            f'disconnectors={self.get_disconnectors()}, '
            f'target={self.get_target()} {self.get_target_unit()})'
        )


class BondSubstructure(Substructure):
    """
    Bond substructure.

    """

    def __init__(self, atom_ids, disconnectors, target):
        """
        Initialize a :class:`BondSubstructure` instance.

        Parameters
        ----------
        atom_ids : :class:`tuple` of :class:`int`
            Atom ids in substructure.

        disconnectors

        target

        """

        self._atom_ids = atom_ids
        self._disconnectors = disconnectors
        self._target = target
        self._target_unit = 'Angstrom'

    def compute_potential(self, position_matrix, target, epsilon):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        if target is None:
            target = self._target

        distance = np.linalg.norm(self._get_bond_vector(
            position_matrix=position_matrix,
        ))

        potential = (distance - target) ** 2
        potential = epsilon * potential

        return potential

    def _get_bond_vector(self, position_matrix):
        """
        Get vector from atom1 to atom2 in bond.
        """

        atom1_pos = position_matrix[self._atom_ids[0]]
        atom2_pos = position_matrix[self._atom_ids[1]]
        return atom2_pos - atom1_pos

    def get_move(self, position_matrix, multiplier, movable_atom_ids):
        """
        Return Move class associated with substructure.

        """

        bond_vector = self._get_bond_vector(
            position_matrix=position_matrix,
        )
        print(bond_vector)
        # Define translation along long bond vector where
        # direction is from force, magnitude is randomly
        # scaled.
        bond_translation = -bond_vector * multiplier
        print(bond_translation)

        return Translation(bond_translation, movable_atom_ids)
