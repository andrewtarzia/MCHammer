import numpy as np
from mchammer import get_atom_distance


def test_opt_get_bond_vector(
    optimizer, bond_vector, position_matrix
):
    assert np.all(np.equal(
        bond_vector,
        optimizer._get_bond_vector(position_matrix, (0, 3)),
    ))


def test_opt_bond_potential(optimizer, bond_potentials):
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._bond_potential(distance=d)
        assert np.isclose(test, bond_potentials[i], atol=1E-5)


def test_opt_nonbond_potential(optimizer, nonbond_potentials):
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._nonbond_potential(distance=d)
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)


def test_opt_compute_nonbonded_potential(
    optimizer, position_matrix, nonbonded_potential
):
    test = optimizer._compute_nonbonded_potential(position_matrix)
    assert test == nonbonded_potential


def test_opt_compute_potential(
    optimizer, molecule, nonbonded_potential, system_potential
):
    test_system_potential, test_nonbond_potential = (
        optimizer._compute_potential(
            molecule, bond_pair_ids=((0, 3), ),
        )
    )
    assert test_nonbond_potential == nonbonded_potential
    assert test_system_potential == system_potential


def test_opt_translate_atoms_along_vector(
    optimizer, molecule, position_matrix, position_matrix3
):
    molecule = molecule.with_position_matrix(position_matrix)
    new_molecule = optimizer._translate_atoms_along_vector(
        mol=molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, 5, 0])
    )
    assert np.all(np.equal(
        position_matrix3,
        new_molecule.get_position_matrix(),
    ))
    new_molecule = optimizer._translate_atoms_along_vector(
        mol=new_molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, -5, 0])
    )
    print(position_matrix, new_molecule.get_position_matrix())
    assert np.all(np.equal(
        position_matrix,
        new_molecule.get_position_matrix(),
    ))


def test_opt_test_move(optimizer):
    # Do not test random component.
    assert optimizer._test_move(curr_pot=-1, new_pot=-2)


def test_opt_get_result(optimizer, molecule):
    original_pos_mat = molecule.get_position_matrix()
    subunits = molecule.get_subunits(bond_pair_ids=((0, 3), ))
    mol, results = optimizer.get_result(
        mol=molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )

    assert results.get_step_count() == 99
    assert len(tuple(results.get_steps_properties())) == 1

    final_bond_length = np.linalg.norm(
        optimizer._get_bond_vector(
            position_matrix=results.get_final_position_matrix(),
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert 1.5 < final_bond_length
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in molecule.get_bonds():
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


def test_opt_get_trajectory(optimizer, molecule):
    original_pos_mat = molecule.get_position_matrix()
    subunits = molecule.get_subunits(bond_pair_ids=((0, 3), ))
    mol, results = optimizer.get_trajectory(
        mol=molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )

    assert results.get_step_count() == 99
    assert len(tuple(results.get_steps_properties())) == 100

    final_bond_length = np.linalg.norm(
        optimizer._get_bond_vector(
            position_matrix=results.get_final_position_matrix(),
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert 1.5 < final_bond_length
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in molecule.get_bonds():
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
