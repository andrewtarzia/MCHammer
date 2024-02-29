"""Optimizer for minimising intermolecular distances."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import pdist

from .mc_operations import test_move, translate_atoms_along_vector
from .results import MCStepResult, Result
from .utilities import get_atom_distance, get_bond_vector

if TYPE_CHECKING:
    from .molecule import Molecule


class Optimizer:
    """Optimize target bonds using MC algorithm.

    A Metropolis MC algorithm is applied to perform rigid
    translations of the subunits separatred by the target bonds.

    """

    def __init__(  # noqa: PLR0913
        self,
        step_size: float,
        target_bond_length: float,
        num_steps: int,
        bond_epsilon: float = 50,
        nonbond_epsilon: float = 20,
        nonbond_sigma: float = 1.2,
        nonbond_mu: float = 3,
        beta: float = 2,
        random_seed: int | None = 1000,
    ) -> None:
        """Initialize a :class:`Optimizer` instance.

        Parameters:
            step_size:
                The relative size of the step to take during step.

            target_bond_length:
                Target equilibrium bond length for long bonds to minimize
                to.

            num_steps:
                Number of MC moves to perform.

            bond_epsilon:
                Value of epsilon used in the bond potential in MC moves.
                Determines strength of the bond potential.
                Defaults to 50.

            nonbond_epsilon:
                Value of epsilon used in the nonbond potential in MC moves.
                Determines strength of the nonbond potential.
                Defaults to 20.

            nonbond_sigma:
                Value of sigma used in the nonbond potential in MC moves.
                Defaults to 1.2.

            nonbond_mu:
                Value of mu used in the nonbond potential in MC moves.
                Determines the steepness of the nonbond potential.
                Defaults to 3.

            beta:
                Value of beta used in the in MC moves. Beta takes the
                place of the inverse boltzmann temperature.
                Defaults to 2.

            random_seed:
                Random seed to use for MC algorithm. Should only be set to
                ``None`` if system-based random seed is desired. Defaults
                to a set seed of 1000, to avoid randomness.

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
            self._generator = np.random.default_rng()
        else:
            self._generator = np.random.default_rng(random_seed)

    def _bond_potential(self, distance: float) -> float:
        """Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """
        potential = (distance - self._target_bond_length) ** 2
        return self._bond_epsilon * potential

    def _nonbond_potential(self, distance: float) -> float:
        """Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """
        return self._nonbond_epsilon * (
            (self._nonbond_sigma / distance) ** self._nonbond_mu
        )

    def _compute_nonbonded_potential(
        self,
        position_matrix: np.ndarray,
    ) -> float:
        # Get all pairwise distances between atoms in each subunut.
        pair_dists = pdist(position_matrix)
        return np.sum(self._nonbond_potential(pair_dists))

    def compute_potential(
        self,
        mol: Molecule,
        bond_pair_ids: list,
    ) -> tuple[float, float]:
        """Compute the potential of a molecule."""
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

    def _output_top_lines(self) -> str:
        return (
            "====================================================\n"
            "                MCHammer optimisation               \n"
            "                ---------------------               \n"
            "                    Andrew Tarzia                   \n"
            "                ---------------------               \n"
            "                                                    \n"
            "Settings:                                           \n"
            f" step size = {self._step_size} \n"
            f" target bond length = {self._target_bond_length} \n"
            f" num. steps = {self._num_steps} \n"
            f" bond epsilon = {self._bond_epsilon} \n"
            f" nonbond epsilon = {self._nonbond_epsilon} \n"
            f" nonbond sigma = {self._nonbond_sigma} \n"
            f" nonbond mu = {self._nonbond_mu} \n"
            f" beta = {self._beta} \n"
            "====================================================\n\n"
        )

    def _run_first_step(
        self,
        mol: Molecule,
        bond_pair_ids: list,
    ) -> tuple[Molecule, MCStepResult]:
        system_potential, nonbonded_potential = self.compute_potential(
            mol=mol, bond_pair_ids=bond_pair_ids
        )

        # Update properties at each step.
        max_bond_distance = max(
            [
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bond_pair_ids
            ]
        )
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
                "-- --\n"
            ),
        )

        return mol, step_result

    def _run_step(  # noqa: PLR0913
        self,
        mol: Molecule,
        bond_pair_ids: list,
        subunits: dict,
        step: int,
        system_potential: float,
        nonbonded_potential: float,
    ) -> tuple[Molecule, MCStepResult]:
        position_matrix = mol.get_position_matrix()

        # Randomly select a bond to optimize from bonds.
        bond_ids = self._generator.choice(bond_pair_ids)
        bond_vector = get_bond_vector(
            position_matrix=position_matrix, bond_pair=bond_ids
        )

        # Get subunits connected by selected bonds.
        subunit_1 = next(i for i in subunits if bond_ids[0] in subunits[i])
        subunit_2 = next(i for i in subunits if bond_ids[1] in subunits[i])

        # Choose subunit to move out of the two connected by the
        # bond randomly.
        moving_su = self._generator.choice([subunit_1, subunit_2])
        moving_su_atom_ids = tuple(i for i in subunits[moving_su])

        # Random number from -1 to 1 for multiplying translation.
        rand = (self._generator.random() - 0.5) * 2
        # Define translation along long bond vector where
        # direction is from force, magnitude is randomly
        # scaled.
        bond_translation = -bond_vector * self._step_size * rand

        # Define subunit COM vector to molecule COM.
        cent = mol.get_centroid()
        su_cent_vector = mol.get_centroid(atom_ids=moving_su_atom_ids) - cent
        com_translation = su_cent_vector * self._step_size * rand

        # Randomly choose between translation along long bond
        # vector or along BB-COM vector.
        translation_vector = self._generator.choice(
            [bond_translation, com_translation]  # type: ignore[arg-type]
        )

        # Translate building block.
        # Update atom position of building block.
        mol = translate_atoms_along_vector(
            mol=mol,
            atom_ids=moving_su_atom_ids,
            vector=translation_vector,  # type: ignore[arg-type]
        )

        (
            new_system_potential,
            new_nonbonded_potential,
        ) = self.compute_potential(mol=mol, bond_pair_ids=bond_pair_ids)

        if test_move(
            beta=self._beta,
            curr_pot=system_potential,
            new_pot=new_system_potential,
            generator=self._generator,
        ):
            updated = "T"
            passed = True
            system_potential = new_system_potential
            nonbonded_potential = new_nonbonded_potential
        else:
            updated = "F"
            passed = False
            # Reverse move.
            mol = translate_atoms_along_vector(
                mol=mol,
                atom_ids=moving_su_atom_ids,
                vector=-translation_vector,  # type: ignore[operator]
            )

        # Update properties at each step.
        max_bond_distance = max(
            [
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bond_pair_ids
            ]
        )
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
                f"{bond_ids} {updated}\n"
            ),
        )

        return mol, step_result

    def get_trajectory(
        self,
        mol: Molecule,
        bond_pair_ids: list[tuple],
        subunits: dict,
    ) -> tuple[Molecule, Result]:
        """Get trajectory of optimization run on `mol`.

        Parameters:
            mol:
                The molecule to be optimized.

            bond_pair_ids:
                :class:`iterable` of :class:`tuple` of :class:`ints`
                Iterable of pairs of atom ids with bond between them to
                optimize.

            subunits:
                The subunits of `mol` split by bonds defined by
                `bond_pair_ids`. Key is subunit identifier, Value is
                :class:`iterable` of atom ids in subunit.

        Returns:
            mol:
                The optimized molecule.

            result:
                The result of the optimization including all steps.

        """
        result = Result(start_time=time.time())

        result.update_log(self._output_top_lines())
        result.update_log(
            f"There are {len(bond_pair_ids)} bonds to optimize.\n"
        )
        result.update_log(
            f"There are {len(subunits)} sub units with N atoms:\n"
            f"{[len(subunits[i]) for i in subunits]}\n"
        )
        result.update_log(
            "====================================================\n"
            "                 Running optimisation!              \n"
            "====================================================\n\n"
        )
        result.update_log(
            "step system_potential nonbond_potential max_dist "
            "opt_bbs updated?\n"
        )
        mol, step_result = self._run_first_step(mol, bond_pair_ids)
        system_potential = step_result.system_potential
        nonbonded_potential = step_result.nonbonded_potential
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

            system_potential = step_result.system_potential
            nonbonded_potential = step_result.nonbonded_potential
            result.add_step_result(step_result=step_result)

        num_passed = result.get_number_passed()
        result.update_log(
            string=(
                "\n============================================\n"
                "Optimisation done:\n"
                f"{num_passed} steps passed: "
                f"{(num_passed/self._num_steps)*100}"
                "%\n"
                f"Total optimisation time: "
                f"{round(result.get_timing(time.time()), 4)}s\n"
                "============================================\n"
            ),
        )

        return mol, result

    def get_result(
        self,
        mol: Molecule,
        bond_pair_ids: list,
        subunits: dict,
    ) -> tuple[Molecule, MCStepResult]:
        """Get final result of optimization run on `mol`.

        Parameters:
            mol:
                The molecule to be optimized.

            bond_pair_ids:
                :class:`iterable` of :class:`tuple` of :class:`ints`
                Iterable of pairs of atom ids with bond between them to
                optimize.

            subunits:
                The subunits of `mol` split by bonds defined by
                `bond_pair_ids`. Key is subunit identifier, Value is
                :class:`iterable` of atom ids in subunit.

        Returns:
            mol:
                The optimized molecule.

            result:
                The result of the final optimization step.

        """
        mol, step_result = self._run_first_step(mol, bond_pair_ids)
        system_potential = step_result.system_potential
        nonbonded_potential = step_result.nonbonded_potential

        for step in range(1, self._num_steps):
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                system_potential=system_potential,
                nonbonded_potential=nonbonded_potential,
            )
            system_potential = step_result.system_potential
            nonbonded_potential = step_result.nonbonded_potential

        return mol, step_result
