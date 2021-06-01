"""
Molecule
========

#. :class:`.Molecule`

Molecule class for optimisation.

"""

import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter

from .angle import Angle


class Molecule:
    """
    Molecule to optimize.

    """

    def __init__(self, atoms, bonds, position_matrix):
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

    def get_position_matrix(self):
        """
        Return a matrix holding the atomic positions.

        Returns
        -------
        :class:`numpy.ndarray`
            The array has the shape ``(n, 3)``. Each row holds the
            x, y and z coordinates of an atom.

        """

        return np.array(self._position_matrix.T)

    def with_position_matrix(self, position_matrix):
        """
        Return clone Molecule with new position matrix.

        Parameters
        ----------
        position_matrix : :class:`numpy.ndarray`
            A position matrix of the clone. The shape of the matrix
            is ``(n, 3)``.

        """

        clone = self.__class__.__new__(self.__class__)
        Molecule.__init__(
            self=clone,
            atoms=self._atoms,
            bonds=self._bonds,
            position_matrix=np.array(position_matrix),
        )
        return clone

    def _write_xyz_content(self):
        """
        Write basic `.xyz` file content of Molecule.

        """
        coords = self.get_position_matrix()
        content = [0]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\n\n'

        return content

    def write_xyz_file(self, path):
        """
        Write basic `.xyz` file of Molecule to `path`.

        Connectivity is not maintained in this file type!

        """

        content = self._write_xyz_content()

        with open(path, 'w') as f:
            f.write(''.join(content))

    def get_atoms(self):
        """
        Yield the atoms in the molecule, ordered as input.

        Yields
        ------
        :class:`.Atom`
            An atom in the molecule.

        """

        for atom in self._atoms:
            yield atom

    def get_bonds(self):
        """
        Yield the bonds in the molecule, ordered as input.

        Yields
        ------
        :class:`.Bond`
            A bond in the molecule.

        """

        for bond in self._bonds:
            yield bond

    def _extract_angles(self):
        """
        Define angles in molecule.

        """

        angles = []
        count = 0
        for bond1, bond2 in combinations(self._bonds, 2):
            b1_a1 = bond1.get_atom1_id()
            b1_a2 = bond1.get_atom2_id()
            b2_a1 = bond2.get_atom1_id()
            b2_a2 = bond2.get_atom2_id()
            if bond1.get_id() == bond2.get_id():
                continue
            sets = set((b1_a1, b1_a2, b2_a1, b2_a2))
            if len(sets) != 3:
                continue
            counts = Counter((b1_a1, b1_a2, b2_a1, b2_a2))
            for c in counts:
                amount = counts[c]
                if amount == 2:
                    angle_atom2_id = c
                    break
            angle_atom1_id, angle_atom3_id = (
                i for i in sets if i != angle_atom2_id
            )
            atom_ids = (angle_atom1_id, angle_atom2_id, angle_atom3_id)
            id_ = count
            angles.append(Angle(id_, atom_ids))
            count += 1
        return angles

    def get_angles(self):
        """
        Yield the angles in the molecule, ordered as input.

        Yields
        ------
        :class:`.Angle`
            An angle in the molecule.

        """

        for angle in self._angles:
            yield angle

    def get_centroid(self, atom_ids=None):
        """
        Return the centroid.

        Parameters
        ----------
        atom_ids : :class:`iterable` of :class:`int`, optional
            The ids of atoms which are used to calculate the
            centroid. Can be a single :class:`int`, if a single
            atom is to be used, or ``None`` if all atoms are to be
            used.

        Returns
        -------
        :class:`numpy.ndarray`
            The centroid of atoms specified by `atom_ids`.

        Raises
        ------
        :class:`ValueError`
            If `atom_ids` has a length of ``0``.

        """

        if atom_ids is None:
            atom_ids = range(len(self._atoms))
        elif isinstance(atom_ids, int):
            atom_ids = (atom_ids, )
        elif not isinstance(atom_ids, (list, tuple)):
            atom_ids = list(atom_ids)

        if len(atom_ids) == 0:
            raise ValueError('atom_ids was of length 0.')

        return np.divide(
            self._position_matrix[:, atom_ids].sum(axis=1),
            len(atom_ids)
        )

    def get_nx_graph(self, bond_pair_ids=None):
        """
        Get networkX graph of molecule.

        Parameters
        ----------
        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        Returns
        -------
        graph : :class:`networkx.graph`
            The graph of `mol`.

        """

        if bond_pair_ids is None:
            bond_pair_ids = (None, )

        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        for atom in self.get_atoms():
            mol_graph.add_node(atom.get_id())

        # Add edges.
        for bond in self.get_bonds():
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            if pair_ids not in bond_pair_ids:
                mol_graph.add_edge(*pair_ids)

        return mol_graph

    def get_subunits(self, bond_pair_ids):
        """
        Get connected graphs based on Molecule separated by bonds.

        Parameters
        ----------
        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        Returns
        -------
        subunits : :class:`.dict`
            The subunits of `mol` split by bonds defined by
            `bond_pair_ids`. Key is subunit identifier, Value is
            :class:`iterable` of atom ids in subunit.

        """

        mol_graph = self.get_nx_graph(bond_pair_ids)

        # Get atom ids in disconnected subgraphs.
        subunits = {
            i: sg
            for i, sg in enumerate(nx.connected_components(mol_graph))
        }

        return subunits

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
