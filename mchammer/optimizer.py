"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimising intermolecular distances.

"""

import numpy as np
import time
from collections import Counter
import random

from .results import MCStepResult, Result
from .potential import MchPotential
from .moves import Translation


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

    def _run_first_step(self, mol):

        system_potential, nonbonded_potential = (
            self._potential_function.compute_potential(mol)
        )

        # Update properties at each step.
        step_result = MCStepResult(
            step=0,
            position_matrix=mol.get_position_matrix(),
            passed=None,
            system_potential=system_potential,
            nonbonded_potential=nonbonded_potential,
            chosen_move=None,
            log=(
                f"{0} "
                f"{system_potential} "
                f"{nonbonded_potential} "
                '-- --\n'
            ),
        )

        return mol, step_result

    def _run_step(
        self,
        mol,
        step,
        system_potential,
        nonbonded_potential,
    ):

        position_matrix = mol.get_position_matrix()
        subunits = mol.get_subunits()
        substructures = tuple(mol.get_substructures())

        # Define potential moves.
        potential_moves = ('com_translation', 'substructure_move')
        chosen_move = random.choice(potential_moves)

        if chosen_move == 'substructure_move':
            # Randomly select a substructure to optimize from bonds.
            random_substructure = random.choice(substructures)
            substructure_ids = tuple(random_substructure.get_atom_ids())

            # Get subunits connected by selected bonds.
            subunit_1 = [
                i
                for i in subunits if substructure_ids[0] in subunits[i]
            ][0]
            subunit_2 = [
                i
                for i in subunits if substructure_ids[1] in subunits[i]
            ][0]
            # Choose subunit to move out of the two connected by the
            # bond randomly.
            moving_su = random.choice([subunit_1, subunit_2])
            moving_su_atom_ids = tuple(i for i in subunits[moving_su])
            # Random number from -1 to 1 for multiplying translation.
            rand = (random.random() - 0.5) * 2
            test_move = random_substructure.get_move(
                position_matrix=position_matrix,
                multiplier=self._step_size * rand,
                movable_atom_ids=moving_su_atom_ids,
            )

        elif chosen_move == 'com_translation':
            # Choose a random subunit.
            moving_su_atom_ids = tuple(random.choice(subunits))
            # Define molecule centroid.
            cent = mol.get_centroid()
            su_cent_vector = (
                mol.get_centroid(atom_ids=moving_su_atom_ids)-cent
            )
            # Random number from -1 to 1 for multiplying translation.
            rand = (random.random() - 0.5) * 2
            test_move = Translation(
                vector=su_cent_vector * self._step_size * rand,
                movable_atom_ids=moving_su_atom_ids,
            )

        # Perform move.
        mol = test_move.perform_move(mol)

        new_system_potential, new_nonbonded_potential = (
            self._potential_function.compute_potential(mol)
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
            mol = test_move.reverse_move(mol)

        # Update properties at each step.
        step_result = MCStepResult(
            step=step,
            position_matrix=mol.get_position_matrix(),
            passed=passed,
            system_potential=system_potential,
            nonbonded_potential=nonbonded_potential,
            chosen_move=chosen_move,
            log=(
                f"{step} "
                f"{system_potential} "
                f"{nonbonded_potential} "
                f'{chosen_move} {updated}\n'
            ),
        )

        return mol, step_result

    def get_trajectory(self, mol):
        """
        Get trajectory of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.Result`
            The result of the optimization including all steps.

        """
        subunits = mol.get_subunits()
        substructures = tuple(mol.get_substructures())

        result = Result(start_time=time.time())

        result.update_log(self._output_top_lines())
        result.update_log(
            f'There are {len(substructures)} substructures to'
            f' optimize.\n'
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
            'step system_potential nonbond_potential '
            'opt_bbs chosen_move updated?\n'
        )
        mol, step_result = self._run_first_step(mol=mol)
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()
        result.add_step_result(step_result=step_result)

        for step in range(1, self._num_steps):
            mol, step_result = self._run_step(
                mol=mol,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )

            system_potential = step_result.get_system_potential()
            nonbonded_potential = step_result.get_nonbonded_potential()
            result.add_step_result(step_result=step_result)
            result.update_log(step_result.get_log())

        num_passed = result.get_number_passed()
        move_types = result.get_all_chosen_moves()
        result.update_log(
            string=(
                '\n============================================\n'
                'Optimisation done:\n'
                f'{num_passed} steps passed: '
                f'{(num_passed/self._num_steps)*100}'
                '%\n'
                f'move types counter: '
                f'{Counter(move_types)}'
                '\n'
                f'Total optimisation time: '
                f'{round(result.get_timing(time.time()), 4)}s\n'
                '============================================\n'
            ),
        )

        return mol, result

    def get_result(self, mol):
        """
        Get final result of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.MCStepResult`
            The result of the final optimization step.

        """
        print('add angular terms')
        print('reinstate rotation options')

        mol, step_result = self._run_first_step(mol)
        system_potential = step_result.get_system_potential()
        nonbonded_potential = step_result.get_nonbonded_potential()

        for step in range(1, self._num_steps):
            mol, step_result = self._run_step(
                mol=mol,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )
            system_potential = step_result.get_system_potential()
            nonbonded_potential = step_result.get_nonbonded_potential()

        return mol, step_result
