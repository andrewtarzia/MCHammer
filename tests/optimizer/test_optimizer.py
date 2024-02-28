from __future__ import annotations

import mchammer as mch
import numpy as np


def test_opt_get_bond_vector(
    bond_vector: np.ndarray,
    position_matrix: np.ndarray,
) -> None:
    assert np.all(
        np.equal(
            bond_vector,
            mch.get_bond_vector(position_matrix, (0, 3)),
        )
    )


def test_opt_bond_potential(
    optimizer: mch.Optimizer,
    bond_potentials: list[int],
) -> None:
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._bond_potential(distance=d)  # noqa: SLF001
        assert np.isclose(test, bond_potentials[i], atol=1e-5)


def test_opt_nonbond_potential(
    optimizer: mch.Optimizer,
    nonbond_potentials: list[float],
) -> None:
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._nonbond_potential(distance=d)  # noqa: SLF001
        assert np.isclose(test, nonbond_potentials[i], atol=1e-5)


def test_opt_compute_nonbonded_potential(
    optimizer: mch.Optimizer,
    position_matrix: np.ndarray,
    nonbonded_potential: float,
) -> None:
    test = optimizer._compute_nonbonded_potential(  # noqa: SLF001
        position_matrix
    )
    assert test == nonbonded_potential


def test_opt_compute_potential(
    optimizer: mch.Optimizer,
    molecule: mch.Molecule,
    nonbonded_potential: float,
    system_potential: float,
) -> None:
    (
        test_system_potential,
        test_nonbond_potential,
    ) = optimizer.compute_potential(
        molecule,
        bond_pair_ids=((0, 3),),
    )
    assert test_nonbond_potential == nonbonded_potential
    assert test_system_potential == system_potential


def test_opt_translate_atoms_along_vector(
    molecule: mch.Molecule,
    position_matrix: np.ndarray,
    position_matrix3: np.ndarray,
) -> None:
    molecule = molecule.with_position_matrix(position_matrix)
    new_molecule = mch.translate_atoms_along_vector(
        mol=molecule, atom_ids=(3, 4, 5), vector=np.array([0, 5, 0])
    )
    assert np.all(
        np.equal(
            position_matrix3,
            new_molecule.get_position_matrix(),
        )
    )
    new_molecule = mch.translate_atoms_along_vector(
        mol=new_molecule, atom_ids=(3, 4, 5), vector=np.array([0, -5, 0])
    )
    print(position_matrix, new_molecule.get_position_matrix())
    assert np.all(
        np.equal(
            position_matrix,
            new_molecule.get_position_matrix(),
        )
    )


def test_opt_test_move() -> None:
    assert mch.test_move(
        beta=2,
        curr_pot=-1,
        new_pot=-2,
        generator=np.random.default_rng(),
    )


def test_opt_get_result(
    optimizer: mch.Optimizer,
    molecule: mch.Molecule,
) -> None:
    original_pos_mat = molecule.get_position_matrix()
    subunits = molecule.get_subunits(bond_pair_ids=((0, 3),))
    mol, results = optimizer.get_result(
        mol=molecule,
        bond_pair_ids=((0, 3),),
        subunits=subunits,
    )

    final_bond_length = np.linalg.norm(
        mch.get_bond_vector(
            position_matrix=results.position_matrix,
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert final_bond_length > 1.5
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in molecule.get_bonds():
        if (bond.get_atom1_id(), bond.get_atom2_id()) != (0, 3):
            test = mch.get_atom_distance(
                position_matrix=results.position_matrix,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            bond_length = mch.get_atom_distance(
                position_matrix=original_pos_mat,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            assert np.isclose(test, bond_length, atol=1e-6)


def test_opt_get_trajectory(
    optimizer: mch.Optimizer,
    molecule: mch.Molecule,
) -> None:
    original_pos_mat = molecule.get_position_matrix()
    subunits = molecule.get_subunits(bond_pair_ids=((0, 3),))
    mol, results = optimizer.get_trajectory(
        mol=molecule,
        bond_pair_ids=((0, 3),),
        subunits=subunits,
    )

    assert results.get_step_count() == 99
    assert len(tuple(results.get_steps_properties())) == 100

    final_bond_length = np.linalg.norm(
        mch.get_bond_vector(
            position_matrix=results.get_final_position_matrix(),
            bond_pair=(0, 3),
        ),
    )
    # Give it some wiggle room.
    assert final_bond_length > 1.5
    assert final_bond_length < 2.5

    # Test all other bond lengths are equivalent.
    for bond in molecule.get_bonds():
        if (bond.get_atom1_id(), bond.get_atom2_id()) != (0, 3):
            test = mch.get_atom_distance(
                position_matrix=results.get_final_position_matrix(),
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            bond_length = mch.get_atom_distance(
                position_matrix=original_pos_mat,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            assert np.isclose(test, bond_length, atol=1e-6)
