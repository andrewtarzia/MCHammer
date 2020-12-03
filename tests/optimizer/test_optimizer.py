from mchammer import Atom, Bond, Molecule, Optimizer, get_atom_distance
import numpy as np
import os

atoms = [
    Atom(0, 'C'), Atom(1, 'C'), Atom(2, 'C'),
    Atom(3, 'C'), Atom(4, 'C'), Atom(5, 'C'),
]
bonds = [
    Bond(0, 0, 1), Bond(1, 0, 2), Bond(2, 0, 3),
    Bond(3, 3, 4), Bond(4, 3, 5)
]
position_matrix = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [0, 10, 0],
    [1, 10, 0],
    [-1, 10, 0],
])
molecule = Molecule(
    atoms=atoms,
    bonds=bonds,
    position_matrix=position_matrix
)
bond_vector = np.array([0, 9, 0])
bond_potentials = [50, 0, 50, 200, 450, 800, 1250]
nonbond_potentials = [
    34.559999999999995, 4.319999999999999, 1.2799999999999998,
    0.5399999999999999, 0.27647999999999995, 0.15999999999999998,
    0.10075801749271138,
]
nonbonded_potential = 147.2965949864993
system_potential = 2597.2965949864993
subnits = {0: {0, 1, 2}, 1: {3, 4, 5}}
position_matrix2 = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [-1, 1, 0],
    [0, 5, 0],
    [1, 5, 0],
    [-1, 5, 0],
])

optimizer = Optimizer(
    output_dir=os.path.join(os.getcwd(), 'test_opt_output'),
    step_size=0.1,
    target_bond_length=2.0,
    num_steps=100
)


def test_opt_get_bond_vector():

    assert np.all(np.equal(
        bond_vector,
        optimizer._get_bond_vector(position_matrix, (0, 3)),
    ))


def test_opt_bond_potential():
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._bond_potential(distance=d)
        assert np.isclose(test, bond_potentials[i], atol=1E-5)


def test_opt_nonbond_potential():
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        test = optimizer._nonbond_potential(distance=d)
        assert np.isclose(test, nonbond_potentials[i], atol=1E-5)


def test_opt_compute_nonbonded_potential():
    test = optimizer._compute_nonbonded_potential(position_matrix)
    assert test == nonbonded_potential


def test_opt_compute_potenmtial():
    test_system_potential, test_nonbond_potential = (
        optimizer._compute_potential(
            molecule, bond_pair_ids=((0, 3), ),
        )
    )
    assert test_system_potential == system_potential
    assert test_nonbond_potential == nonbonded_potential


def test_opt_translate_atoms_along_vector():
    new_molecule = optimizer._translate_atoms_along_vector(
        mol=molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, 5, 0])
    )
    assert np.all(np.equal(
        position_matrix2,
        new_molecule.get_position_matrix(),
    ))
    new_molecule = optimizer._translate_atoms_along_vector(
        mol=molecule,
        atom_ids=(3, 4, 5),
        vector=np.array([0, -5, 0])
    )
    assert np.all(np.equal(
        position_matrix,
        new_molecule.get_position_matrix(),
    ))


def test_opt_test_move():
    # Do not test random component.
    assert optimizer._test_move(curr_pot=-1, new_pot=-2)


def test_opt_get_subunits():
    test = optimizer._get_subunits(molecule, bond_pair_ids=((0, 3), ))
    assert test == subnits


def test_opt_optimize():

    original_pos_mat = molecule.get_position_matrix()
    new_molecule = optimizer.optimize(
        mol=molecule,
        bond_pair_ids=((0, 3), )
    )
    final_bond_length = np.linalg.norm(
        optimizer._get_bond_vector(
            position_matrix=new_molecule.get_position_matrix(),
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
                position_matrix=new_molecule.get_position_matrix(),
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            bond_length = get_atom_distance(
                position_matrix=original_pos_mat,
                atom1_id=bond.get_atom1_id(),
                atom2_id=bond.get_atom2_id(),
            )
            assert np.isclose(test, bond_length, atol=1E-6)
