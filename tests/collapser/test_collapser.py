import numpy as np
from mchammer import get_atom_distance


def test_c_get_subunit_distances(
    collapser,
    coll_molecule,
    coll_subunits,
    coll_su_dists,
    coll_position_matrix,
):
    test = coll_molecule.with_position_matrix(coll_position_matrix)
    for i, dist in enumerate(collapser._get_subunit_distances(
        mol=test,
        subunits=coll_subunits,
    )):
        print(i)
        assert np.isclose(dist, coll_su_dists[i], atol=1E-6)


def test_c_get_new_position_matrix(
    collapser,
    coll_position_matrix,
    coll_position_matrix2,
    coll_molecule,
    coll_subunits,
    coll_step,
    coll_vectors,
    coll_scales,
):
    test_mol = coll_molecule.with_position_matrix(coll_position_matrix)
    test_pos_mat = collapser._get_new_position_matrix(
        mol=test_mol,
        subunits=coll_subunits,
        step_size=coll_step,
        vectors=coll_vectors,
        scales=coll_scales,
    )

    for i in range(len(coll_position_matrix2)):
        test = test_pos_mat[i]
        given = coll_position_matrix2[i]
        assert np.all(np.isclose(test, given, atol=1E-6))


def test_c_get_bb_vectors(
    collapser, coll_molecule, coll_subunits, su_vectors, su_scales,
):
    test_vectors, test_scales = collapser._get_subunit_vectors(
        coll_molecule, coll_subunits
    )
    for t, s in zip(test_vectors, su_vectors):
        test = test_vectors[t]
        su_v = su_vectors[s]
        assert np.all(np.equal(test, su_v))
        test = test_scales[t]
        su_s = su_scales[s]
        assert np.all(np.equal(test, su_s))


def test_c_get_result(
    collapser, coll_molecule, coll_final_position_matrix
):
    original_pos_mat = coll_molecule.get_position_matrix()
    subunits = coll_molecule.get_subunits(bond_pair_ids=((0, 3), ))
    test_mol, results = collapser.get_result(
        mol=coll_molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )

    assert results.get_step_count() == 35
    assert len(tuple(results.get_steps_properties())) == 1

    final_min_distance = min(
        dist for dist in collapser._get_subunit_distances(
            test_mol, subunits
        )
    )
    # Give it some wiggle room.
    assert np.isclose(1.4947505, final_min_distance, atol=1E-8)

    # Check position matrix because this code is deterministic.
    test_pos_mat = test_mol.get_position_matrix()
    for i in range(len(coll_final_position_matrix)):
        test = test_pos_mat[i]
        given = coll_final_position_matrix[i]
        assert np.all(np.isclose(test, given, atol=1E-8))

    # Test all other bond lengths are equivalent.
    for bond in coll_molecule.get_bonds():
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


def test_c_get_trajectory(
    collapser, coll_molecule, coll_final_position_matrix
):
    original_pos_mat = coll_molecule.get_position_matrix()
    subunits = coll_molecule.get_subunits(bond_pair_ids=((0, 3), ))
    test_mol, results = collapser.get_trajectory(
        mol=coll_molecule,
        bond_pair_ids=((0, 3), ),
        subunits=subunits,
    )
    test_mol = coll_molecule.with_position_matrix(
        results.get_final_position_matrix()
    )
    assert results.get_step_count() == 35
    assert len(tuple(results.get_steps_properties())) == 35

    final_min_distance = min(
        dist for dist in collapser._get_subunit_distances(
            test_mol, subunits
        )
    )
    # Give it some wiggle room.
    assert np.isclose(1.4947505, final_min_distance, atol=1E-8)

    # Check position matrix because this code is deterministic.
    test_pos_mat = test_mol.get_position_matrix()
    for i in range(len(coll_final_position_matrix)):
        test = test_pos_mat[i]
        given = coll_final_position_matrix[i]
        assert np.all(np.isclose(test, given, atol=1E-8))

    # Test all other bond lengths are equivalent.
    for bond in coll_molecule.get_bonds():
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
