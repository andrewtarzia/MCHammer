import pytest
import numpy as np
import os
import mchammer as mch


# @pytest.fixture
# def benzene_build():
#     """
#     Benzene fixture with distorted geometry.
#
#     """
#
#     path_tcurrent_file = os.path.realpath(__file__)
#     current_directory = os.path.split(path_tcurrent_file)[0]
#     benzene = os.path.join(current_directory, "benzene.mol")
#     mol = stk.BuildingBlock.init_from_file(benzene)
#
#     return mol

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
