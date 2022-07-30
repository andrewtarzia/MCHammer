"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimising intermolecular distances.

"""

import numpy as np
import time

from scipy.spatial.distance import pdist
from copy import deepcopy
import random

from .results import MCStepResult, Result
from .potential import MchPotential
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
        num_steps,
        potential_function=MchPotential(),
        beta=2,
        random_seed=1000,
    ):
        """
        Initialize a :class:`Optimizer` instance.

        Parameters
        ----------
        step_size : :class:`float`
            The relative size of the step to take during step.

        num_steps : :class:`int`
            Number of MC moves to perform.

        potential_function : :class:`spd.Potential`
            Function to calculate potential energy of a
            :class:`spd.Supramolecule`

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
        self._num_steps = num_steps
        self._potential_function = potential_function
        self._beta = beta
        if random_seed is None:
            random.seed()
            np.random.seed()
        else:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _translate_atoms_along_vector(self, mol, atom_ids, vector):

        new_position_matrix = deepcopy(mol.get_position_matrix())
        for atom in mol.get_atoms():
            if atom.get_id() not in atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - vector

        mol = mol.with_position_matrix(new_position_matrix)
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
            f' target bond length = {self._potential_function.get_target_bond_length()} \n'
            f' num. steps = {self._num_steps} \n'
            f' bond epsilon = {self._potential_function.get_bond_epsilon()} \n'
            f' nonbond epsilon = {self._potential_function.get_nonbond_epsilon()} \n'
            f' nonbond mu = {self._potential_function.get_nonbond_mu()} \n'
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

        system_potential, nonbonded_potential = (
            self._potential_function.compute_potential(
                molecule=mol,
                bond_pair_ids=bond_pair_ids,
            )
        )

        # Update properties at each step.
        max_bond_distance = max([
            get_atom_distance(
                position_matrix=mol.get_position_matrix(),
                atom1_id=bond[0],
                atom2_id=bond[1],
            )
            for bond in bond_pair_ids
        ])
        step_result = MCStepResult(
            step=0,
            position_matrix=mol.get_position_matrix(),
            passed=None,
            system_potential=system_potential,
            nonbonded_potential=nonbonded_potential,
            max_bond_distance=max_bond_distance,
            log=(
                f"{0} "
                f"{system_potential} "
                f"{nonbonded_potential} "
                f"{max_bond_distance} "
                '-- --\n'
            ),
        )

        return mol, step_result

    def _run_step(
        self,
        mol,
        bond_pair_ids,
        subunits,
        step,
        system_potential,
        nonbonded_potential,
    ):

        position_matrix = mol.get_position_matrix()

        raise SystemExit('This requires the substructures (rename) defined')
        # Randomly select a bond to optimize from bonds.
        substructure_id = random.choice(substructures)
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
        max_bond_distance = max([
            get_atom_distance(
                position_matrix=mol.get_position_matrix(),
                atom1_id=bond[0],
                atom2_id=bond[1],
            )
            for bond in bond_pair_ids
        ])
        step_result = MCStepResult(
            step=step,
            position_matrix=mol.get_position_matrix(),
            passed=passed,
            system_potential=system_potential,
            nonbonded_potential=nonbonded_potential,
            max_bond_distance=max_bond_distance,
            log=(
                f"{step} "
                f"{system_potential} "
                f"{nonbonded_potential} "
                f"{max_bond_distance} "
                f'{bond_ids} {updated}\n'
            ),
        )

        return mol, step_result

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
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.Result`
            The result of the optimization including all steps.

        """

        result = Result(start_time=time.time())

        result.update_log(self._output_top_lines())
        result.update_log(
            f'There are {len(bond_pair_ids)} bonds to optimize.\n'
        )
        result.update_log(
            f'There are {len(subunits)} sub units with N atoms:\n'
            f'{[len(subunits[i]) for i in subunits]}\n'
        )
        result.update_log(
            '====================================================\n'
            '                 Running optimisation!              \n'
            '====================================================\n\n'
        )
        result.update_log(
            'step system_potential nonbond_potential max_dist '
            'opt_bbs updated?\n'
        )
        mol, step_result = self._run_first_step(
            mol,
            bond_pair_ids,
            subunits,
        )
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()
        result.add_step_result(step_result=step_result)

        for step in range(1, self._num_steps):
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
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

        return mol, result

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
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.MCStepResult`
            The result of the final optimization step.

        """

        raise SystemExit('Actually no, the molecule class should take the subcomponentFactory')
        raise SystemExit('the subcomponent (created by factory) should carry potential information - starting with just angles.')
        raise SystemExit('new interface should require subcomponent (rename) factories and rotatable bond finding flag')
        raise SystemExit('remove bond pair ids and subunits approach')
        raise SystemExit('reinstate rotation options')

        if self._use_rotatable_bonds:
            raise SystemExit('rename')
            subcomponents = mol.get_rotatable_bonds()

        subcomponents = mol.get_subcompeonts()

        raise SystemExit('add new potential classes liek spindry')
        potential = self._potential_function(molecule)
        system_potential = potential.get_system_potential()
        nonbonded_potential = potential.get_nonbonded_potential()

        mol, step_result = self._run_first_step(
            mol,
            bond_pair_ids,
            subunits,
        )
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()

        for step in range(1, self._num_steps):
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )
            system_potential = step_result.get_system_potential()
            nonbonded_potential = step_result.get_nonbonded_potential()

        return mol, step_result
