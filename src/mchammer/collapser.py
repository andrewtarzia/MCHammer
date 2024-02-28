"""Optimizer for minimising intermolecular distances."""

from __future__ import annotations

import time
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

from .results import Result, StepResult
from .utilities import get_atom_distance

if TYPE_CHECKING:
    from collections import abc

    from .molecule import Molecule


class Collapser:
    """Moves rigid-subunits toward center of mass of molecule."""

    def __init__(
        self,
        step_size: float,
        distance_threshold: float,
        scale_steps: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a :class:`Collapser` instance.

        Parameters
        ----------
        step_size : :class:`float`
            The relative size of the step to take during collapse.

        distance_threshold : :class:`float`
            Distance between distinct subunits to use as
            threshold for halting collapse in Angstrom.

        scale_steps : :class:`bool`, optional
            Whether to scale the step of each distict building block
            by their relative distance from the molecules centroid.
            Defaults to ``True``

        """
        self._step_size = step_size
        self._distance_threshold = distance_threshold
        self._scale_steps = scale_steps

    def _has_short_contacts(self, mol: Molecule, subunits: dict) -> bool:
        """Calculate if there are short contants in mol."""
        return any(
            dist < self._distance_threshold
            for dist in self._get_subunit_distances(mol, subunits)
        )

    def _get_subunit_distances(
        self,
        mol: Molecule,
        subunits: dict[int, tuple],
    ) -> abc.Iterable[float]:
        """Yield the distances between subunits in mol.

        Ignores H atoms.

        """
        atom_elements = [i.get_element_string() for i in mol.get_atoms()]
        position_matrix = mol.get_position_matrix()

        for su1, su2 in combinations(subunits, 2):
            if su1 == su2:
                continue
            su1_atom_ids = subunits[su1]
            su2_atom_ids = subunits[su2]
            for atom1_id, atom2_id in product(su1_atom_ids, su2_atom_ids):
                atom1_element = atom_elements[atom1_id]
                atom2_element = atom_elements[atom2_id]
                if atom1_element != "H" and atom2_element != "H":
                    yield get_atom_distance(
                        position_matrix=position_matrix,
                        atom1_id=atom1_id,
                        atom2_id=atom2_id,
                    )

    def _get_new_position_matrix(  # noqa: PLR0913
        self,
        mol: Molecule,
        subunits: dict,
        vectors: dict,
        scales: dict,
        step_size: float | None,
    ) -> np.ndarray:
        """Get the position matrix of the mol after translation."""
        if step_size is None:
            step_size = self._step_size

        new_position_matrix = mol.get_position_matrix()
        for su in subunits:
            for atom_id in subunits[su]:
                pos = mol.get_position_matrix()[atom_id]
                new_position_matrix[atom_id] = (
                    pos - step_size * vectors[su] * scales[su]
                )

        return new_position_matrix

    def _get_subunit_vectors(
        self,
        mol: Molecule,
        subunits: dict,
    ) -> tuple[dict, dict]:
        """Get the subunit to COM vectors.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        subunits : :class:`.dict`
            The subunits of `mol` split by bonds defined by
            `bond_pair_ids`. Key is subunit identifier, Value is
            :class:`iterable` of atom ids in subunit.

        Returns:
        -------
        su_cent_vectors :
            :class:`dict` mapping :class:`int`: to
            :class:`numpy.ndarray`
            Dictionary mapping subunit ids (keys) to centroid
            vectors (values) of each distinct building block in the
            molecule.

        su_cent_scales :
            :class:`dict` mapping :class:`int`: to :class:`float`
            Dictionary mapping subunit ids (keys) to relative
            magnitude of centroid vectors (values) of each distinct
            building block in the molecule.

        """
        centroid = mol.get_centroid()

        # Get subunit centroid to molecule centroid vector.

        su_cent_vectors = {
            i: mol.get_centroid(atom_ids=subunits[i]) - centroid
            for i in subunits
        }

        # Scale the step size based on the different distances of
        # subunits from the COM. Impacts anisotropic topologies.
        if self._scale_steps:
            norms = {
                i: np.linalg.norm(su_cent_vectors[i]) for i in su_cent_vectors
            }
            max_distance = max(list(norms.values()))
            su_cent_scales = {i: float(norms[i] / max_distance) for i in norms}
        else:
            su_cent_scales = {i: 1.0 for i in su_cent_vectors}

        return su_cent_vectors, su_cent_scales

    def _run_step(  # noqa: PLR0913
        self,
        mol: Molecule,
        bond_pair_ids: tuple,
        subunits: dict,
        step: int,
        step_size: float | None = None,
    ) -> tuple[Molecule, StepResult]:
        # Get the subunit to centroid vectors and their relative scale.
        su_cent_vectors, su_cent_scales = self._get_subunit_vectors(
            mol=mol, subunits=subunits
        )

        new_position_matrix = self._get_new_position_matrix(
            mol=mol,
            subunits=subunits,
            vectors=su_cent_vectors,
            scales=su_cent_scales,
            step_size=step_size,
        )

        mol = mol.with_position_matrix(new_position_matrix)
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
        step_result = StepResult(
            step=step,
            position_matrix=new_position_matrix,
            max_bond_distance=max_bond_distance,
            log=f"{step} {max_bond_distance}\n",
        )
        return mol, step_result

    def get_trajectory(
        self,
        mol: Molecule,
        bond_pair_ids: tuple,
        subunits: dict,
    ) -> tuple[Molecule, Result]:
        """Get trajectory of optimization run on `mol`.

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

        Returns:
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.Result`
            The result of the optimization including all steps.

        """
        result = Result(start_time=time.time())
        result.update_log(
            "====================================================\n"
            f"                Collapser algorithm.\n"
            "====================================================\n"
            f"Step size: {self._step_size}\n"
            f"Scale steps?: {self._scale_steps}\n"
            f"Distance threshold: {self._distance_threshold}\n"
        )
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

        step = 0
        while not self._has_short_contacts(mol, subunits):
            step += 1
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
            )
            result.add_step_result(step_result=step_result)

        # Check that we have not gone too far.
        min_dist = min(
            dist for dist in self._get_subunit_distances(mol, subunits)
        )
        if min_dist < self._distance_threshold / 2:
            step += 1
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                step_size=-(self._step_size / 2),
            )
            result.add_step_result(step_result=step_result)

        result.update_log(
            "====================================================\n"
            f"Steps run: {step}\n"
            f"Minimum inter-subunit distance: {min_dist}\n"
            "====================================================\n"
        )

        return mol, result

    def get_result(
        self,
        mol: Molecule,
        bond_pair_ids: tuple,
        subunits: dict,
    ) -> tuple[Molecule, StepResult]:
        """Get final result of the optimization of `mol`.

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

        Returns:
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.StepResult`
            The result of the final optimization step.

        """
        step = 0
        while not self._has_short_contacts(mol, subunits):
            step += 1
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
            )

        # Check that we have not gone too far.
        min_dist = min(
            dist for dist in self._get_subunit_distances(mol, subunits)
        )
        if min_dist < self._distance_threshold / 2:
            step += 1
            mol, step_result = self._run_step(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                subunits=subunits,
                step=step,
                step_size=-(self._step_size / 2),
            )
        return mol, step_result
