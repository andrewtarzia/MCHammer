import numpy as np
from mchammer import get_atom_distance


def test_opt_get_bond_vector(
    o_optimizer, o_bond_vector, o_position_matrix
):
    assert np.all(np.equal(
        o_bond_vector,
        o_optimizer._get_bond_vector(o_position_matrix, (0, 3)),
    ))


def test_opt_bond_potential(o_optimizer, o_bond_potentials):
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = o_optimizer._bond_potential(distance=d)
        assert np.isclose(test, o_bond_potentials[i], atol=1E-5)


def test_opt_nonbond_potential(o_optimizer, o_nonbond_potentials):
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = o_optimizer._nonbond_potential(distance=d)
        assert np.isclose(test, o_nonbond_potentials[i], atol=1E-5)


def test_opt_compute_nonbonded_potential(
    o_optimizer, o_position_matrix, o_nonbonded_potential
):
    test = o_optimizer._compute_nonbonded_potential(o_position_matrix)
    assert test == o_nonbonded_potential


def test_opt_compute_potential(
    o_optimizer, o_molecule, o_nonbonded_potential, o_system_potential
):
    test_system_potential, test_nonbond_potential = (
        o_optimizer._compute_potential(
            o_molecule, bond_pair_ids=((0, 3), ),
        )
    )
    assert test_nonbond_potential == o_nonbonded_potential
    assert test_system_potential == o_system_potential


def test_opt_translate_atoms_along_vector(
    o_optimizer, o_molecule, o_position_matrix, o_position_matrix2
):
    new_molecule = o_optimizer._translate_atoms_along_vector(
        mol=o_molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, 5, 0])
    )
    assert np.all(np.equal(
        o_position_matrix2,
        new_molecule.get_position_matrix(),
    ))
    new_molecule = o_optimizer._translate_atoms_along_vector(
        mol=o_molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, -5, 0])
    )
    assert np.all(np.equal(
        o_position_matrix,
        new_molecule.get_position_matrix(),
    ))


def test_opt_test_move(o_optimizer):
    # Do not test random component.
    assert o_optimizer._test_move(curr_pot=-1, new_pot=-2)


def test_opt_get_subunits(o_optimizer, o_molecule, o_subunits):
    test = o_optimizer.get_subunits(
        o_molecule, bond_pair_ids=((0, 3), )
    )
    assert test == o_subunits


def test_opt_get_result(o_optimizer, o_molecule):
    original_pos_mat = o_molecule.get_position_matrix()
    subunits = o_optimizer.get_subunits(
        o_molecule, bond_pair_ids=((0, 3), )
    )
    results = o_optimizer.get_result(
        mol=o_molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )

    assert results.get_step_count() == 99
    assert len(tuple(results.get_steps_properties())) == 1

    final_bond_length = np.linalg.norm(
        o_optimizer._get_bond_vector(
            position_matrix=results.get_final_position_matrix(),
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert 1.5 < final_bond_length
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in o_molecule.get_bonds():
        if (bond.get_atom1_id(), bond.get_atom2_id()) != (0, 3):
            test = get_atom_distance(
                position_matrix=results.get_final_position_matrix(),
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            bond_length = get_atom_distance(
                position_matrix=original_pos_mat,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            assert np.isclose(test, bond_length, atol=1E-6)


def test_opt_get_trajectory(o_optimizer, o_molecule):
    original_pos_mat = o_molecule.get_position_matrix()
    subunits = o_optimizer.get_subunits(
        o_molecule, bond_pair_ids=((0, 3), )
    )
    results = o_optimizer.get_trajectory(
        mol=o_molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )

    assert results.get_step_count() == 99
    assert len(tuple(results.get_steps_properties())) == 100

    final_bond_length = np.linalg.norm(
        o_optimizer._get_bond_vector(
            position_matrix=results.get_final_position_matrix(),
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert 1.5 < final_bond_length
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in o_molecule.get_bonds():
        if (bond.get_atom1_id(), bond.get_atom2_id()) != (0, 3):
            test = get_atom_distance(
                position_matrix=results.get_final_position_matrix(),
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            bond_length = get_atom_distance(
                position_matrix=original_pos_mat,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            assert np.isclose(test, bond_length, atol=1E-6)
