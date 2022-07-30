"""
Molecule
========

#. :class:`.Molecule`

Molecule class for optimisation.

"""

import numpy as np
import networkx as nx


class Molecule:
    """
    Molecule to optimize.

    """

    def __init__(
        self,
        atoms,
        bonds,
        position_matrix,
        subunit_factories=None,
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

        subunit_factories

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )
        self._subunit_factories = subunit_factories

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

    def _write_pdb_content(self):
        """
        Write basic `.pdb` file content of Molecule.

        """

        content = []
        atom_counts = {}
        hetatm = 'HETATM'
        alt_loc = ''
        res_name = 'UNL'
        chain_id = ''
        res_seq = '1'
        i_code = ''
        occupancy = '1.00'
        temp_factor = '0.00'

        coords = self.get_position_matrix()
        # This set will be used by bonds.
        atoms = set()
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            atom_id = atom.get_id()
            atoms.add(atom_id)
            serial = atom_id+1
            element = atom.get_element_string()
            charge = 0
            atom_counts[element] = atom_counts.get(element, 0) + 1
            name = f'{element}{atom_counts[element]}'

            content.append(
                f'{hetatm:<6}{serial:>5} {name:<4}'
                f'{alt_loc:<1}{res_name:<3} {chain_id:<1}'
                f'{res_seq:>4}{i_code:<1}   '
                f' {x:>7.3f} {y:>7.3f} {z:>7.3f}'
                f'{occupancy:>6}{temp_factor:>6}          '
                f'{element:>2}{charge:>2}\n'
            )

        conect = 'CONECT'
        for bond in self.get_bonds():
            a1 = bond.get_atom1_id()
            a2 = bond.get_atom2_id()
            if a1 in atoms and a2 in atoms:
                content.append(
                    f'{conect:<6}{a1+1:>5}{a2+1:>5}               \n'
                )

        content.append('END\n')

        return content

    def write_pdb_file(self, path):
        """
        Write basic `.pdb` file of Molecule to `path`.

        """

        content = self._write_pdb_content()

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

        # Get atom ids in disconnected subgraphs.
        subunits = {
            i: sg
            for i, sg in enumerate(nx.connected_components(mol_graph))
        }

        return subunits

    def get_subunit_molecules(self, bond_pair_ids):
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
        subunit_molecules : :class:`.dict`
            The subunits of `mol` split by bonds defined by
            `bond_pair_ids`. Key is subunit identifier, Value is
            :class:`mch.Molecule`.

        """

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

        # Get atom ids in disconnected subgraphs.
        subunits = {}
        for i, c in enumerate(nx.connected_components(mol_graph)):
            c_ids = sorted(c)
            in_atoms = [
                i for i in self._atoms
                if i.get_id() in c
            ]
            in_bonds = [
                i for i in self._bonds
                if i.get_atom1_id() in c and i.get_atom2_id() in c
            ]
            new_pos_matrix = self._position_matrix[:, list(c_ids)].T
            subunits[i] = Molecule(in_atoms, in_bonds, new_pos_matrix)

        return subunits

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
