import pytest
import os
import numpy as np


def test_molecule_get_position_matrix(m_molecule, m_position_matrix):
    assert np.all(np.equal(
        m_position_matrix,
        m_molecule.get_position_matrix(),
    ))


def test_molecule_update_position_matrix(
    m_molecule, m_position_matrix2, m_position_matrix
):
    m_molecule.update_position_matrix(m_position_matrix2)
    assert np.all(np.equal(
        m_position_matrix2,
        m_molecule.get_position_matrix(),
    ))


@pytest.fixture
def path(request, tmpdir):
    return os.path.join(tmpdir, 'molecule.xyz')


def test_molecule_write_xyz_file(m_molecule, path):
    m_molecule.write_xyz_file(path)
    content = m_molecule._write_xyz_content()
    with open(path, 'r') as f:
        test_lines = f.readlines()

    assert ''.join(test_lines) == ''.join(content)


def test_molecule_get_atoms(m_molecule, m_atoms):
    for test, atom in zip(m_molecule.get_atoms(), m_atoms):
        assert test.get_id() == atom.get_id()
        assert test.get_element_string() == atom.get_element_string()


def test_molecule_get_bonds(m_molecule, m_bonds):
    for test, bond in zip(m_molecule.get_bonds(), m_bonds):
        assert test.get_id() == bond.get_id()
        assert test.get_atom1_id() == bond.get_atom1_id()
        assert test.get_atom2_id() == bond.get_atom2_id()


def test_molecule_get_centroid(
    m_molecule, m_position_matrix, m_centroid
):
    print(m_molecule.get_position_matrix())
    m_molecule.update_position_matrix(m_position_matrix)
    print(m_molecule.get_position_matrix())
    assert np.all(np.allclose(
        m_centroid,
        m_molecule.get_centroid(),
        atol=1E-6,
    ))
