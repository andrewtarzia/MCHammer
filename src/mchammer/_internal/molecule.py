"""Molecule class for optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections import abc

    from .atom import Atom
    from .bond import Bond


@dataclass
class Molecule:
    """Molecule to optimize.

    Parameters:
        atoms:
            Atoms that define the molecule.

        bonds:
            Bonds between atoms that define the molecule.

        position_matrix:
            A ``(n, 3)`` matrix holding the position of every atom in
            the :class:`.Molecule`.

    """

    atoms: tuple[Atom, ...]
    bonds: tuple[Bond, ...]
    position_matrix: np.ndarray

    def __post_init__(self) -> None:
        """Post initialization of molecule."""
        self.atoms = tuple(self.atoms)
        self.bonds = tuple(self.bonds)
        self.position_matrix = np.array(
            self.position_matrix.T,
            dtype=np.float64,
        )

    def get_position_matrix(self) -> np.ndarray:
        """Return a matrix holding the atomic positions.

        Returns:
            The array has the shape ``(n, 3)``. Each row holds the
            x, y and z coordinates of an atom.

        """
        return np.array(self.position_matrix.T)

    def with_displacement(self, displacement: np.ndarray) -> Molecule:
        """Return a displaced clone Molecule.

        Parameters:
            displacement:
                The displacement vector to be applied.

        """
        new_position_matrix = self.position_matrix.T + displacement
        return Molecule(
            atoms=tuple(self.atoms),
            bonds=tuple(self.bonds),
            position_matrix=np.array(new_position_matrix),
        )

    def with_position_matrix(self, position_matrix: np.ndarray) -> Molecule:
        """Return clone Molecule with new position matrix.

        Parameters:
            position_matrix:
                A position matrix of the clone. The shape of the matrix
                is ``(n, 3)``.

        """
        return Molecule(
            atoms=tuple(self.atoms),
            bonds=tuple(self.bonds),
            position_matrix=np.array(position_matrix),
        )

    def write_xyz_content(self) -> list[str]:
        """Write basic `.xyz` file content of Molecule."""
        coords = self.get_position_matrix()
        content = ["0"]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(f"{atom.get_element_string()} {x:f} {y:f} {z:f}\n")
        # Set first line to the atom_count.
        content[0] = f"{i}\n\n"

        return content

    def write_xyz_file(self, path: str) -> None:
        """Write basic `.xyz` file of Molecule to `path`.

        Connectivity is not maintained in this file type!

        """
        content = self.write_xyz_content()

        with open(path, "w") as f:
            f.write("".join(content))

    def _write_pdb_content(self) -> list[str]:
        """Write basic `.pdb` file content of Molecule."""
        content = []
        atom_counts: dict[str, int] = {}
        hetatm = "HETATM"
        alt_loc = ""
        res_name = "UNL"
        chain_id = ""
        res_seq = "1"
        i_code = ""
        occupancy = "1.00"
        temp_factor = "0.00"

        coords = self.get_position_matrix()
        # This set will be used by bonds.
        atoms = set()
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            atom_id = atom.get_id()
            atoms.add(atom_id)
            serial = atom_id + 1
            element = atom.get_element_string()
            charge = 0
            atom_counts[element] = atom_counts.get(element, 0) + 1
            name = f"{element}{atom_counts[element]}"

            content.append(
                f"{hetatm:<6}{serial:>5} {name:<4}"
                f"{alt_loc:<1}{res_name:<3} {chain_id:<1}"
                f"{res_seq:>4}{i_code:<1}   "
                f" {x:>7.3f} {y:>7.3f} {z:>7.3f}"
                f"{occupancy:>6}{temp_factor:>6}          "
                f"{element:>2}{charge:>2}\n"
            )

        conect = "CONECT"
        for bond in self.get_bonds():
            a1 = bond.get_atom1_id()
            a2 = bond.get_atom2_id()
            if a1 in atoms and a2 in atoms:
                content.append(
                    f"{conect:<6}{a1+1:>5}{a2+1:>5}               \n"
                )

        content.append("END\n")

        return content

    def write_pdb_file(self, path: str) -> None:
        """Write basic `.pdb` file of Molecule to `path`."""
        content = self._write_pdb_content()

        with open(path, "w") as f:
            f.write("".join(content))

    def get_atoms(self) -> abc.Iterable[Atom]:
        """Yield the atoms in the molecule, ordered as input."""
        yield from self.atoms

    def get_bonds(self) -> abc.Iterable[Bond]:
        """Yield the bonds in the molecule, ordered as input."""
        yield from self.bonds

    def get_num_atoms(self) -> int:
        """Return the number of atoms in the molecule."""
        return len(self.atoms)

    def get_centroid(self, atom_ids: tuple | set | None = None) -> float:
        """Return the centroid.

        Parameters:
            atom_ids: :class:`iterable` of :class:`int`, optional
                The ids of atoms which are used to calculate the
                centroid. Can be a single :class:`int`, if a single
                atom is to be used, or ``None`` if all atoms are to be
                used.

        Returns:
            The centroid of atoms specified by `atom_ids`.

        Raises:
            If `atom_ids` has a length of ``0``.

        """
        if atom_ids is None:
            atom_ids = range(len(self.atoms))  # type: ignore[assignment]
        elif not isinstance(atom_ids, (list, tuple)):
            atom_ids = tuple(atom_ids)

        if len(atom_ids) == 0:  # type: ignore[arg-type]
            msg = "atom_ids was of length 0."
            raise ValueError(msg)

        return np.divide(
            self.position_matrix[:, atom_ids].sum(axis=1),  # type: ignore[index]
            len(atom_ids),  # type: ignore[arg-type]
        )

    def get_subunits(self, bond_pair_ids: tuple) -> dict:
        """Get connected graphs based on Molecule separated by bonds.

        Parameters:
            bond_pair_ids:
                :class:`iterable` of :class:`tuple` of :class:`ints`
                Iterable of pairs of atom ids with bond between them to
                optimize.

        Returns:
            subunits:
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
        return dict(enumerate(nx.connected_components(mol_graph)))

    def __str__(self) -> str:
        """String representation."""
        return repr(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<{self.__class__.__name__}({len(self.atoms)} atoms) "
            f"at {id(self)}>"
        )
