import pytest
import os
from mchammer import Atom, Bond, Molecule
import numpy as np


atoms = [Atom(0, 'C'), Atom(1, 'C'), Atom(2, 'N')]
bonds = [Bond(0, 0, 1), Bond(1, 1, 2)]
position_matrix = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
])
position_matrix2 = np.array([
    [0, -10, 0],
    [0, 1, 0],
    [2, 0, 0],
])
molecule = Molecule(
    atoms=atoms,
    bonds=bonds,
    position_matrix=position_matrix
)
centroid = np.array([0.66666667, -3., 0.])


def test_molecule_get_position_matrix():
    assert np.all(np.equal(
        position_matrix,
        molecule.get_position_matrix(),
    ))


def test_molecule_update_position_matrix():
    molecule.update_position_matrix(position_matrix2)
    assert np.all(np.equal(
        position_matrix2,
        molecule.get_position_matrix(),
    ))


@pytest.fixture
def path(request, tmpdir):
    return os.path.join(tmpdir, 'molecule.xyz')


def test_molecule_write_xyz_file(path):
    molecule.write_xyz_file(path)
    content = molecule._write_xyz_content()
    with open(path, 'r') as f:
        test_lines = f.readlines()

    assert ''.join(test_lines) == ''.join(content)


def test_molecule_get_atoms():
    for test, atom in zip(molecule.get_atoms(), atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()


def test_molecule_get_bonds():
    for test, bond in zip(molecule.get_bonds(), bonds):
        assert test.get_id() == bond.get_id()
        assert test.get_atom1_id() == bond.get_atom1_id()
        assert test.get_atom2_id() == bond.get_atom2_id()


def test_molecule_get_centroid():
    assert np.all(np.allclose(
        centroid,
        molecule.get_centroid(),
        atol=1E-6,
    ))
