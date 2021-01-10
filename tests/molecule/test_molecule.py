import pytest
import os
import numpy as np


def test_molecule_get_position_matrix(molecule, position_matrix):
    assert np.all(np.equal(
        position_matrix,
        molecule.get_position_matrix(),
    ))


def test_molecule_with_position_matrix(
    molecule, position_matrix2, position_matrix
):
    test = molecule.with_position_matrix(position_matrix2)
    assert np.all(np.equal(
        position_matrix2,
        test.get_position_matrix(),
    ))


@pytest.fixture
def path(request, tmpdir):
    return os.path.join(tmpdir, 'molecule.xyz')


def test_molecule_write_xyz_file(molecule, path):
    molecule.write_xyz_file(path)
    content = molecule._write_xyz_content()
    with open(path, 'r') as f:
        test_lines = f.readlines()

    assert ''.join(test_lines) == ''.join(content)


def test_molecule_get_atoms(molecule, atoms):
    for test, atom in zip(molecule.get_atoms(), atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()


def test_molecule_get_bonds(molecule, bonds):
    for test, bond in zip(molecule.get_bonds(), bonds):
        assert test.get_id() == bond.get_id()
        assert test.get_atom1_id() == bond.get_atom1_id()
        assert test.get_atom2_id() == bond.get_atom2_id()


def test_molecule_get_centroid(
    molecule, position_matrix, centroid
):
    test = molecule.with_position_matrix(position_matrix)
    assert np.all(np.allclose(
        centroid,
        test.get_centroid(),
        atol=1E-6,
    ))


def test_molecule_get_subunits(molecule, subunits):
    test = molecule.get_subunits(bond_pair_ids=((0, 3), ))
    assert test == subunits
