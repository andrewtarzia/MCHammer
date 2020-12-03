from mchammer import get_atom_distance
import numpy as np


def test_get_atom_distance():
    position_matrix = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [2, 0, 0],
    ])
    assert get_atom_distance(position_matrix, 0, 1) == 1
    assert get_atom_distance(position_matrix, 0, 2) == 2
