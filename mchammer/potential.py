"""
Potential
=========

#. :class:`.Potential`
#. :class:`.MchPotential`

Classes for calculating the potential energy of molecules.

"""

import numpy as np

from itertools import combinations
from scipy.spatial.distance import cdist


class Potential:
    """
    Base class for potential calculators.

    """

    def __init__(self):
        """
        Initialize a :class:`Potential` instance.

        """

    def _get_bond_vector(self, position_matrix, bond_pair):
        """
        Get vector from atom1 to atom2 in bond.

        """

        atom1_pos = position_matrix[bond_pair[0]]
        atom2_pos = position_matrix[bond_pair[1]]
        return atom2_pos - atom1_pos

    def compute_potential(self, molecule):
        """
        Calculate potential energy.

        Parameters
        ----------
        molecule : :class:`mch.Molecule`
            Molecule to evaluate.

        """
        raise NotImplementedError()


class MchPotential(Potential):
    """
    Default mchammer potential function.

    """

    def __init__(
        self,
        target_bond_length=1.2,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_mu=3,
    ):
        """
        Initialize a :class:`MchPotential` instance.

        Parameters
        ----------
        target_bond_length : :class:`float`
            Target equilibrium bond length for long bonds to minimize
            to.

        bond_epsilon : :class:`float`, optional
            Value of epsilon used in the bond potential in MC moves.
            Determines strength of the bond potential.
            Defaults to 50.

        nonbond_epsilon : :class:`float`, optional
            Value of epsilon used in the nonbond potential in MC moves.
            Determines strength of the nonbond potential.
            Defaults to 20.

        nonbond_mu : :class:`float`, optional
            Value of mu used in the nonbond potential in MC moves.
            Determines the steepness of the nonbond potential.
            Defaults to 3.


        """

        self._target_bond_length = target_bond_length
        self._bond_epsilon = bond_epsilon
        self._nonbond_epsilon = nonbond_epsilon
        self._nonbond_mu = nonbond_mu

    def get_target_bond_length(self):
        return self._target_bond_length

    def get_bond_epsilon(self):
        return self._bond_epsilon

    def get_nonbond_epsilon(self):
        return self._nonbond_epsilon

    def get_nonbond_mu(self):
        return self._nonbond_mu

    def _nonbond_potential(self, distance, sigmas):
        """
        Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (sigmas/distance) ** self._nonbond_mu
            )
        )

    def _mixing_function(self, val1, val2):
        return (val1 + val2) / 2

    def _combine_sigma(self, radii1, radii2):
        """
        Combine radii using Lorentz-Berthelot rules.

        """

        len1 = len(radii1)
        len2 = len(radii2)

        mixed = np.zeros((len1, len2))
        for i in range(len1):
            for j in range(len2):
                mixed[i, j] = self._mixing_function(
                    radii1[i], radii2[j],
                )

        return mixed

    def _compute_nonbonded_potential(self, position_matrices, radii):
        nonbonded_potential = 0
        for pos_mat_pair, radii_pair in zip(
            combinations(position_matrices, 2),
            combinations(radii, 2),
        ):
            pair_dists = cdist(pos_mat_pair[0], pos_mat_pair[1])
            sigmas = self._combine_sigma(radii_pair[0], radii_pair[1])
            nonbonded_potential += np.sum(
                self._nonbond_potential(
                    distance=pair_dists.flatten(),
                    sigmas=sigmas.flatten(),
                )
            )

        return nonbonded_potential

    def compute_potential(self, molecule):
        subunit_molecules = molecule.get_subunit_molecules()
        component_position_matrices = (
            subunit_molecules[i].get_position_matrix()
            for i in subunit_molecules
        )
        component_radii = (
            tuple(j.get_radius() for j in (
                subunit_molecules[i].get_atoms())
            )
            for i in molecule.get_subunits()
        )
        nonbonded_potential = self._compute_nonbonded_potential(
            position_matrices=component_position_matrices,
            radii=component_radii,
        )
        position_matrix = molecule.get_position_matrix()
        system_potential = nonbonded_potential
        for substructure in molecule.get_substructures():
            system_potential += substructure.compute_potential(
                position_matrix=position_matrix,
                target=self._target_bond_length,
                epsilon=self._bond_epsilon,
            )

        return system_potential, nonbonded_potential
