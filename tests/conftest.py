import pytest
import numpy as np
import mchammer as mch


@pytest.fixture(
    params=(
        (mch.Atom(id=0, element_string='N'), 0, 'N'),
        (mch.Atom(id=65, element_string='P'), 65, 'P'),
        (mch.Atom(id=2, element_string='C'), 2, 'C'),
    )
)
def atom_info(request):
    return request.param


@pytest.fixture(
    params=(
        (mch.Bond(id=0, atom1_id=0, atom2_id=1), 0, 0, 1),
        (mch.Bond(id=65, atom1_id=2, atom2_id=3), 65, 2, 3),
        (mch.Bond(id=2, atom1_id=3, atom2_id=4), 2, 3, 4),
        (mch.Bond(id=3, atom1_id=0, atom2_id=9), 3, 0, 9),
    )
)
def bond_info(request):
    return request.param


@pytest.fixture
def atoms():
    return [
        mch.Atom(0, 'C'), mch.Atom(1, 'C'), mch.Atom(2, 'C'),
        mch.Atom(3, 'C'), mch.Atom(4, 'C'), mch.Atom(5, 'C'),
    ]


@pytest.fixture
def bonds():
    return [
        mch.Bond(0, 0, 1), mch.Bond(1, 0, 2), mch.Bond(2, 0, 3),
        mch.Bond(3, 3, 4), mch.Bond(4, 3, 5)
    ]


@pytest.fixture
def position_matrix():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 10, 0],
        [1, 10, 0],
        [-1, 10, 0],
    ])


@pytest.fixture
def position_matrix2():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 20, 0],
        [1, 20, 0],
        [-1, 20, 0],
    ])


@pytest.fixture
def centroid():
    return np.array([0, 5.5, 0])


@pytest.fixture
def molecule(atoms, bonds, position_matrix):
    return mch.Molecule(
        atoms=atoms,
        bonds=bonds,
        position_matrix=position_matrix
    )


@pytest.fixture
def bond_vector():
    return np.array([0, 9, 0])


@pytest.fixture
def bond_potentials():
    return [50, 0, 50, 200, 450, 800, 1250]


@pytest.fixture
def nonbond_potentials():
    return [
        34.559999999999995, 4.319999999999999, 1.2799999999999998,
        0.5399999999999999, 0.27647999999999995, 0.15999999999999998,
        0.10075801749271138,
    ]


@pytest.fixture
def nonbonded_potential():
    return 147.2965949864993


@pytest.fixture
def system_potential():
    return 2597.2965949864993


@pytest.fixture
def subunits():
    return {0: {0, 1, 2}, 1: {3, 4, 5}}


@pytest.fixture
def position_matrix3():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 5, 0],
        [1, 5, 0],
        [-1, 5, 0],
    ])


@pytest.fixture
def optimizer():
    return mch.Optimizer(
        step_size=0.1,
        target_bond_length=2.0,
        num_steps=100
    )


@pytest.fixture
def coll_atoms():
    return [
        mch.Atom(0, 'C'), mch.Atom(1, 'C'), mch.Atom(2, 'C'),
        mch.Atom(3, 'C'), mch.Atom(4, 'C'), mch.Atom(5, 'C'),
    ]


@pytest.fixture
def coll_bonds():
    return [
        mch.Bond(0, 0, 1), mch.Bond(1, 0, 2), mch.Bond(2, 0, 3),
        mch.Bond(3, 3, 4), mch.Bond(4, 3, 5)
    ]


@pytest.fixture
def coll_position_matrix():
    return np.array([
        [0, 1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [0, 10, 0],
        [1, 10, 0],
        [-1, 10, 0],
    ])


@pytest.fixture
def coll_vectors(request):
    return {0: np.array([2, -1.5, 0]), 1: np.array([0, 2.5, 0])}


@pytest.fixture
def coll_su_dists(request):
    return [
        9, 9.055385138137417, 9.055385138137417, 9.055385138137417,
        9, 9.219544457292887, 9.055385138137417, 9.219544457292887,
        9,
    ]


@pytest.fixture
def coll_scales(request):
    return {0: 0.2, 1: 1.0}


@pytest.fixture
def coll_step(request):
    return 1.5


@pytest.fixture
def coll_position_matrix2():
    return np.array([
        [0, 1, 0],
        [0, 1, 0],
        [-1, 1, 0],
        [0, 6, 0],
        [1, 6, 0],
        [-1, 6, 0],
    ])


@pytest.fixture
def su_vectors(request):
    return {0: np.array([0, -4.5, 0]), 1: np.array([0, 4.5, 0])}


@pytest.fixture
def su_scales(request):
    return {0: 1.0, 1: 1.0}


@pytest.fixture
def coll_molecule(coll_atoms, coll_bonds, coll_position_matrix):
    return mch.Molecule(
        atoms=coll_atoms,
        bonds=coll_bonds,
        position_matrix=coll_position_matrix
    )


@pytest.fixture
def coll_subunits():
    return {0: {0, 1, 2}, 1: {3, 4, 5}}


@pytest.fixture
def coll_final_position_matrix():
    return np.array([
        [0., 4.75262477, 0.],
        [1., 4.75262477, 0.],
        [-1., 4.75262477, 0.],
        [0., 6.24737523, 0.],
        [1., 6.24737523, 0.],
        [-1., 6.24737523, 0.],
    ])


@pytest.fixture
def collapser():
    return mch.Collapser(
        step_size=0.05,
        distance_threshold=1.5,
        scale_steps=True,
    )
