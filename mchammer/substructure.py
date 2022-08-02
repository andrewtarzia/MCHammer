"""
Substructure
============

#. :class:`.Substructure`

"""

import numpy as np

from .moves import Translation, RotationAboutAxis
from .utilities import vector_angle


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

    def compute_potential(self, position_matrix, epsilon, target=None):
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
        # Define translation along long bond vector where
        # direction is from force, magnitude is randomly
        # scaled.
        bond_translation = -bond_vector * multiplier

        return Translation(bond_translation, movable_atom_ids)


class AngleSubstructure(Substructure):
    """
    Angle substructure.

    """

    def __init__(self, atom_ids, disconnectors, target):
        """
        Initialize a :class:`AngleSubstructure` instance.

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
        self._target_unit = 'degrees'

    def compute_potential(self, position_matrix, epsilon):
        """
        Define an arbitrary parabolic angular potential.

        This potential has no relation to an empircal forcefield.

        """

        angle = self._get_angle(
            position_matrix=position_matrix,
        )

        potential = (angle - self._target) ** 2
        potential = epsilon * potential
        return potential

    def _get_angle(self, position_matrix):
        """
        Get angle between atom1-atom2 and atom2-atom3.

        """

        atom1_pos = position_matrix[self._atom_ids[0]]
        atom2_pos = position_matrix[self._atom_ids[1]]
        atom3_pos = position_matrix[self._atom_ids[2]]
        v1 = atom1_pos - atom2_pos
        v2 = atom3_pos - atom2_pos
        return np.degrees(vector_angle(v1, v2))

    def get_move(
        self,
        position_matrix,
        multiplier,
        movable_atom_ids,
        origin,
        axis,
    ):
        """
        Return Move class associated with substructure.

        """

        angle = self._get_angle(
            position_matrix=position_matrix,
        )
        angle_diff = self._target - angle
        rotation_angle = angle_diff * multiplier

        return RotationAboutAxis(
            angle=np.radians(rotation_angle),
            movable_atom_ids=movable_atom_ids,
            axis=axis,
            origin=origin,
        )
