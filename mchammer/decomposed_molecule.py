"""
DecomposedMolecule
==================

#. :class:`.DecomposedMolecule`

DecomposedMolecule class for optimisation.

"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter

from .molecule import Molecule
from .atom import Atom
from .bond import Bond
from .angle import Angle


class DecomposedMolecule(Molecule):
    """
    DecomposedMolecule to optimize.

    """

    def __init__(
        self,
        atoms,
        bonds,
        position_matrix,
        decomp_bond_pair_ids,
        translation_dict,
    ):
        """
        Initialize a :class:`Molecule` instance.

        Parameters
        ----------
        atoms : :class:`iterable` of :class:`.Atom`
            Atoms that define the molecule.

        bonds : :class:`iterable` of :class:`.Bond`
            Bonds between atoms that define the molecule.

        position_matrix : :class:`numpy.ndarray`
            A ``(n, 3)`` matrix holding the position of every atom in
            the :class:`.Molecule`.

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)
        self._angles = self._extract_angles()
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )
        self._translation_dict = translation_dict
        self._decomp_bond_pair_ids = decomp_bond_pair_ids

    @classmethod
    def decompose_molecule(cls, molecule, bond_pair_ids):
        """
        Get decomposition of molecule.

        Defines atoms and bonds to maintain:
            1) Centroids of each subunit.
            2) Bonds between subunits.

        """

        subunits = molecule.get_subunits(bond_pair_ids)

        atoms = []
        bonds = []
        position_matrix = []
        decomp_bond_pair_ids = []
        translation_dict = {}
        next_atom_id = 0
        next_bond_id = 0
        # Add atoms where bond pair ids connect.
        atoms_in_bond_pairs = set([
            id_ for pair in bond_pair_ids for id_ in pair
        ])
        for atom_id in atoms_in_bond_pairs:
            atoms.append(Atom(next_atom_id, 'B'))
            position_matrix.append(
                molecule.get_centroid(atom_id)
            )
            translation_dict[(atom_id, 'bonder')] = next_atom_id
            next_atom_id += 1

        # Add subunit centroids.
        for su in subunits:
            if len(subunits[su]) > 1:
                atoms.append(Atom(next_atom_id, 'C'))
                position_matrix.append(
                    molecule.get_centroid(subunits[su])
                )
                translation_dict[(su, 'subunit')] = next_atom_id
                next_atom_id += 1
            else:
                # Handle single atom building blocks.
                # Check translation dict for existing atom.
                atom_id = tuple(subunits[su])[0]
                try:
                    decomp_a1_id = translation_dict[
                        (atom_id, 'bonder')
                    ]
                    # Do not need to atom.
                except KeyError:
                    raise NotImplementedError(
                        'Cannot handle the case where a single atom'
                        'building block does not have a bond.'
                    )

        # Add bonds.
        for bond_pair in bond_pair_ids:
            a1_id, a2_id = bond_pair
            decomp_a1_id = translation_dict[(a1_id, 'bonder')]
            decomp_a2_id = translation_dict[(a2_id, 'bonder')]
            bonds.append(
                Bond(next_bond_id, (decomp_a1_id, decomp_a2_id))
            )
            decomp_bond_pair_ids.append(
                tuple(sorted((decomp_a1_id, decomp_a2_id)))
            )
            next_bond_id += 1

        # Connect subunit centroids with their binding atoms.
        for atom_id in atoms_in_bond_pairs:
            decomp_a1_id = translation_dict[(atom_id, 'bonder')]
            # Get the subunit it is in.
            for su in subunits:
                if atom_id in subunits[su]:
                    subunit_id = su
                    break
            if len(subunits[subunit_id]) == 1:
                continue
            # Get the subunit centroid atom id.
            decomp_a2_id = translation_dict[(subunit_id, 'subunit')]
            # Define a bond.
            bonds.append(
                Bond(next_bond_id, (decomp_a1_id, decomp_a2_id))
            )
            next_bond_id += 1

        return DecomposedMolecule(
            atoms=tuple(atoms),
            bonds=tuple(bonds),
            position_matrix=np.asarray(position_matrix),
            translation_dict=translation_dict,
            decomp_bond_pair_ids=tuple(decomp_bond_pair_ids),
        )



    def with_position_matrix(self, position_matrix):
        """
        Return clone DecomposedMolecule with new position matrix.

        Parameters
        ----------
        position_matrix : :class:`numpy.ndarray`
            A position matrix of the clone. The shape of the matrix
            is ``(n, 3)``.

        """

        clone = self.__class__.__new__(self.__class__)
        DecomposedMolecule.__init__(
            self=clone,
            atoms=self._atoms,
            bonds=self._bonds,
            position_matrix=np.array(position_matrix),
            translation_dict=self._translation_dict,
            decomp_bond_pair_ids=self._decomp_bond_pair_ids,
        )
        return clone

    def get_bond_pair_ids(self):
        return self._decomp_bond_pair_ids

    def recompose_molecule(self, molecule):
        """
        Return recomposed molecule.

        """

        print(self._translation_dict)
        raise NotImplementedError()

        return molecule
