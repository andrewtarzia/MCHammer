"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimising intermolecular distances.

"""

import numpy as np
import time

from scipy.spatial.distance import pdist
import random

from .results import Result, StepResult
from .utilities import get_atom_distance


class Optimizer:
    """
    Optimize target bonds using MC algorithm.

    A Metropolis MC algorithm is applied to perform rigid
    translations of the subunits separatred by the target bonds.

    """

    def __init__(
        self,
        step_size,
        target_bond_length,
        num_steps,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=1000,
    ):
        """
        Initialize a :class:`Collapser` instance.

        Parameters
        ----------
        step_size : :class:`float`
            The relative size of the step to take during step.

        target_bond_length : :class:`float`
            Target equilibrium bond length for long bonds to minimize
            to.

        num_steps : :class:`int`
            Number of MC moves to perform.

        bond_epsilon : :class:`float`, optional
            Value of epsilon used in the bond potential in MC moves.
            Determines strength of the bond potential.
            Defaults to 50.

        nonbond_epsilon : :class:`float`, optional
            Value of epsilon used in the nonbond potential in MC moves.
            Determines strength of the nonbond potential.
            Defaults to 20.

        nonbond_sigma : :class:`float`, optional
            Value of sigma used in the nonbond potential in MC moves.
            Defaults to 1.2.

        nonbond_mu : :class:`float`, optional
            Value of mu used in the nonbond potential in MC moves.
            Determines the steepness of the nonbond potential.
            Defaults to 3.

        beta : :class:`float`, optional
            Value of beta used in the in MC moves. Beta takes the
            place of the inverse boltzmann temperature.
            Defaults to 2.

        random_seed : :class:`int` or :class:`NoneType`, optional
            Random seed to use for MC algorithm. Should only be set to
            ``None`` if system-based random seed is desired. Defaults
            to 1000.

        """

        self._step_size = step_size
        self._target_bond_length = target_bond_length
        self._num_steps = num_steps
        self._bond_epsilon = bond_epsilon
        self._nonbond_epsilon = nonbond_epsilon
        self._nonbond_sigma = nonbond_sigma
        self._nonbond_mu = nonbond_mu
        self._beta = beta
        if random_seed is None:
            random.seed()
        else:
            random.seed(random_seed)

    def _get_bond_vector(self, position_matrix, bond_pair):
        """
        Get vector from atom1 to atom2 in bond.

        """

        atom1_pos = position_matrix[bond_pair[0]]
        atom2_pos = position_matrix[bond_pair[1]]
        return atom2_pos - atom1_pos

    def _bond_potential(self, distance):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        potential = (distance - self._target_bond_length) ** 2
        potential = self._bond_epsilon * potential

        return potential

    def _nonbond_potential(self, distance):
        """
        Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (self._nonbond_sigma/distance) ** self._nonbond_mu
            )
        )

    def _compute_nonbonded_potential(self, position_matrix):
        # Get all pairwise distances between atoms in each subunut.
        pair_dists = pdist(position_matrix)
        nonbonded_potential = np.sum(
            self._nonbond_potential(pair_dists)
        )

        return nonbonded_potential

    def _compute_potential(self, mol, bond_pair_ids):
        position_matrix = mol.get_position_matrix()
        nonbonded_potential = self._compute_nonbonded_potential(
            position_matrix=position_matrix,
        )
        system_potential = nonbonded_potential
        for bond in bond_pair_ids:
            system_potential += self._bond_potential(
                distance=get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
            )

        return system_potential, nonbonded_potential

    def _translate_atoms_along_vector(self, mol, atom_ids, vector):

        new_position_matrix = mol.get_position_matrix()
        for atom in mol.get_atoms():
            if atom.get_id() not in atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - vector

        mol.update_position_matrix(new_position_matrix)
        return mol

    def _test_move(self, curr_pot, new_pot):

        if new_pot < curr_pot:
            return True
        else:
            exp_term = np.exp(-self._beta*(new_pot-curr_pot))
            rand_number = random.random()

            if exp_term > rand_number:
                return True
            else:
                return False

    def _output_top_lines(self):

        string = (
            '====================================================\n'
            '                MCHammer optimisation               \n'
            '                ---------------------               \n'
            '                    Andrew Tarzia                   \n'
            '                ---------------------               \n'
            '                                                    \n'
            'Settings:                                           \n'
            f' step size = {self._step_size} \n'
            f' target bond length = {self._target_bond_length} \n'
            f' num. steps = {self._num_steps} \n'
            f' bond epsilon = {self._bond_epsilon} \n'
            f' nonbond epsilon = {self._nonbond_epsilon} \n'
            f' nonbond sigma = {self._nonbond_sigma} \n'
            f' nonbond mu = {self._nonbond_mu} \n'
            f' beta = {self._beta} \n'
            '====================================================\n\n'
        )

        return string

    def _run_first_step(
        self,
        mol,
        bond_pair_ids,
        subunits,
    ):

        step_result = StepResult(step=0)

        step_result.update_log(self._output_top_lines())
        step_result.update_log(
            f'There are {len(bond_pair_ids)} bonds to optimize.\n'
        )
        step_result.update_log(
            f'There are {len(subunits)} sub units with N atoms:\n'
            f'{[len(subunits[i]) for i in subunits]}\n'
        )
        step_result.update_log(
            '====================================================\n'
            '                 Running optimisation!              \n'
            '====================================================\n\n'
        )

        system_potential, nonbonded_potential = (
            self._compute_potential(
                mol=mol,
                bond_pair_ids=bond_pair_ids
            )
        )

        # Update properties at each step.
        step_result.set_position_matrix(mol.get_position_matrix())
        step_result.set_passed(None)
        step_result.set_system_potential(system_potential)
        step_result.set_nonbonded_potential(nonbonded_potential)
        step_result.set_max_bond_distance(max([
            get_atom_distance(
                position_matrix=mol.get_position_matrix(),
                atom1_id=bond[0],
                atom2_id=bond[1],
            )
            for bond in bond_pair_ids
        ]))
        step_result.update_log(
            'step system_potential nonbond_potential max_dist '
            'opt_bbs updated?\n'
        )
        step_result.update_log(
            f"{0} "
            f"{step_result.get_system_potential()} "
            f"{step_result.get_nonbonded_potential()} "
            f"{step_result.get_max_bond_distance()} "
            '-- --\n'
        )

        return step_result

    def _run_step(
        self,
        mol,
        bond_pair_ids,
        subunits,
        step,
        system_potential,
        nonbonded_potential,
    ):

        step_result = StepResult(step=step)
        position_matrix = mol.get_position_matrix()

        # Randomly select a bond to optimize from bonds.
        bond_ids = random.choice(bond_pair_ids)
        bond_vector = self._get_bond_vector(
            position_matrix=position_matrix,
            bond_pair=bond_ids
        )

        # Get subunits connected by selected bonds.
        subunit_1 = [
            i for i in subunits if bond_ids[0] in subunits[i]
        ][0]
        subunit_2 = [
            i for i in subunits if bond_ids[1] in subunits[i]
        ][0]

        # Choose subunit to move out of the two connected by the
        # bond randomly.
        moving_su = random.choice([subunit_1, subunit_2])
        moving_su_atom_ids = tuple(i for i in subunits[moving_su])

        # Random number from -1 to 1 for multiplying translation.
        rand = (random.random() - 0.5) * 2
        # Define translation along long bond vector where
        # direction is from force, magnitude is randomly
        # scaled.
        bond_translation = -bond_vector * self._step_size * rand

        # Define subunit COM vector to molecule COM.
        cent = mol.get_centroid()
        su_cent_vector = (
            mol.get_centroid(atom_ids=moving_su_atom_ids)-cent
        )
        com_translation = su_cent_vector * self._step_size * rand

        # Randomly choose between translation along long bond
        # vector or along BB-COM vector.
        translation_vector = random.choice([
            bond_translation,
            com_translation,
        ])

        # Translate building block.
        # Update atom position of building block.
        mol = self._translate_atoms_along_vector(
            mol=mol,
            atom_ids=moving_su_atom_ids,
            vector=translation_vector,
        )

        new_system_potential, new_nonbonded_potential = (
            self._compute_potential(
                mol=mol,
                bond_pair_ids=bond_pair_ids
            )
        )

        if self._test_move(system_potential, new_system_potential):
            updated = 'T'
            passed = True
            system_potential = new_system_potential
            nonbonded_potential = new_nonbonded_potential
        else:
            updated = 'F'
            passed = False
            # Reverse move.
            mol = self._translate_atoms_along_vector(
                mol=mol,
                atom_ids=moving_su_atom_ids,
                vector=-translation_vector,
            )

        # Update properties at each step.
        step_result.set_position_matrix(mol.get_position_matrix())
        step_result.set_passed(passed)
        step_result.set_system_potential(system_potential)
        step_result.set_nonbonded_potential(nonbonded_potential)
        step_result.set_max_bond_distance(max([
            get_atom_distance(
                position_matrix=mol.get_position_matrix(),
                atom1_id=bond[0],
                atom2_id=bond[1],
            )
            for bond in bond_pair_ids
        ]))

        step_result.update_log(
            f"{step} "
            f"{step_result.get_system_potential()} "
            f"{step_result.get_nonbonded_potential()} "
            f"{step_result.get_max_bond_distance()} "
            f'{bond_ids} {updated}\n'
        )

        return step_result

    def get_trajectory(self, mol, bond_pair_ids, subunits):
        """
        Get trajectory of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        subunits : :class:`.dict`
            The subunits of `mol` split by bonds defined by
            `bond_pair_ids`. Key is subunit identifier, Value is
            :class:`iterable` of atom ids in subunit.

        Returns
        -------
        result : :class:`.Result`
            The result of the optimization.

        """

        result = Result(start_time=time.time())
        step_result = self._run_first_step(
            mol,
            bond_pair_ids,
            subunits,
        )
        mol.update_position_matrix(
            step_result.get_position_matrix()
        )
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()
        result.add_step_result(step_result=step_result)

        for step in range(1, self._num_steps):
            step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )

            mol.update_position_matrix(
                step_result.get_position_matrix()
            )
            system_potential = step_result.get_system_potential()
            nonbonded_potential = step_result.get_nonbonded_potential()
            result.add_step_result(step_result=step_result)

        num_passed = result.get_number_passed()
        result.update_log(
            string=(
                '\n============================================\n'
                'Optimisation done:\n'
                f'{num_passed} steps passed: '
                f'{(num_passed/self._num_steps)*100}'
                '%\n'
                f'Total optimisation time: '
                f'{round(result.get_timing(time.time()), 4)}s\n'
                '============================================\n'
            ),
        )

        return result

    def get_result(self, mol, bond_pair_ids, subunits):
        """
        Get final result of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        subunits : :class:`.dict`
            The subunits of `mol` split by bonds defined by
            `bond_pair_ids`. Key is subunit identifier, Value is
            :class:`iterable` of atom ids in subunit.

        Returns
        -------
        result : :class:`.Result`
            The result of the optimization.

        """

        result = Result(start_time=time.time())
        step_result = self._run_first_step(
            mol,
            bond_pair_ids,
            subunits,
        )
        mol.update_position_matrix(
            step_result.get_position_matrix()
        )
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()

        for step in range(1, self._num_steps):
            step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )

            mol.update_position_matrix(
                step_result.get_position_matrix()
            )
            system_potential = step_result.get_system_potential()
            nonbonded_potential = step_result.get_nonbonded_potential()

        # Only add final step.
        result.add_step_result(step_result=step_result)

        return result
