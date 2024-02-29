from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mchammer as mch


def test_molecule_get_position_matrix(
    molecule: mch.Molecule,
    position_matrix: np.ndarray,
) -> None:
    assert np.all(
        np.equal(
            position_matrix,
            molecule.get_position_matrix(),
        )
    )


def test_molecule_with_position_matrix(
    molecule: mch.Molecule,
    position_matrix2: np.ndarray,
) -> None:
    test = molecule.with_position_matrix(position_matrix2)
    assert np.all(
        np.equal(
            position_matrix2,
            test.get_position_matrix(),
        )
    )


def test_molecule_with_displacement(
    molecule: mch.Molecule,
    displacement: np.ndarray,
    displaced_position_matrix: np.ndarray,
) -> None:
    test = molecule.with_displacement(displacement)
    print(test.get_position_matrix())
    assert np.all(
        np.allclose(
            displaced_position_matrix,
            test.get_position_matrix(),
        )
    )


@pytest.fixture()
def path(tmpdir) -> None:  # noqa: ANN001
    return os.path.join(tmpdir, "molecule.xyz")  # noqa: PTH118


def test_molecule_write_xyz_file(molecule: mch.Molecule, path: str) -> None:
    molecule.write_xyz_file(path)
    content = molecule.write_xyz_content()
    with open(path) as f:
        test_lines = f.readlines()

    assert "".join(test_lines) == "".join(content)


def test_molecule_get_atoms(
    molecule: mch.Molecule,
    atoms: list[mch.Atom],
) -> None:
    for test, atom in zip(molecule.get_atoms(), atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()


def test_molecule_get_bonds(
    molecule: mch.Molecule,
    bonds: list[mch.Bond],
) -> None:
    for test, bond in zip(molecule.get_bonds(), bonds):
        assert test.get_id() == bond.get_id()
        assert test.get_atom1_id() == bond.get_atom1_id()
        assert test.get_atom2_id() == bond.get_atom2_id()


def test_molecule_get_centroid(
    molecule: mch.Molecule,
    position_matrix: np.ndarray,
    centroid: np.ndarray,
) -> None:
    test = molecule.with_position_matrix(position_matrix)
    assert np.all(
        np.allclose(
            centroid,
            test.get_centroid(),
            atol=1e-6,
        )
    )


def test_molecule_get_subunits(molecule: mch.Molecule, subunits: dict) -> None:
    test = molecule.get_subunits(bond_pair_ids=((0, 3),))
    assert test == subunits


def test_molecule_get_num_atoms(
    molecule: mch.Molecule,
    num_atoms: int,
) -> None:
    assert molecule.get_num_atoms() == num_atoms
