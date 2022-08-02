"""
This module defines general-purpose objects, functions and classes.

"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import euclidean
import rdkit.Chem.AllChem as rdkit


periodic_table = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C',
    7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg',
    13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co',
    28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge',
    33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb',
    38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
    48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te',
    53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm',
    63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
    68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf',
    73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir',
    78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
    83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr',
    88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U',
    93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk',
    98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No',
    103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh',
    108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
    113: 'Uut', 114: 'Fl', 115: 'Uup', 116: 'Lv',
    117: 'Uus', 118: 'Uuo'
}


def get_atom_angle(position_matrix, atom_ids):
    """
    Get angle between atom1-atom2 and atom2-atom3.

    """

    atom1_pos = position_matrix[atom_ids[0]]
    atom2_pos = position_matrix[atom_ids[1]]
    atom3_pos = position_matrix[atom_ids[2]]
    v1 = atom1_pos - atom2_pos
    v2 = atom3_pos - atom2_pos
    return np.degrees(vector_angle(v1, v2))


def get_atom_distance(position_matrix, atom1_id, atom2_id):
    """
    Return the distance between two atoms.

    """

    return float(euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id]
    ))


def get_atom_ids(query, molecule):
    """
    Yield the ids of atoms in `molecule` which match `query`.

    Multiple substructures in `molecule` can match `query` and
    therefore each set is yielded as a group.

    Parameters
    ----------
    query : :class:`str`
        A SMARTS string used to query atoms.

    molecule : :class:`.Molecule`
        A molecule whose atoms should be queried.

    Yields
    ------
    :class:`tuple` of :class:`int`
        The ids of atoms in `molecule` which match `query`.

    """

    rdkit_mol = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_mol)
    yield from rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(query),
    )


def get_atomic_number(element_string):
    for i in periodic_table:
        if periodic_table[i] == element_string:
            return i


def rotation_matrix_arbitrary_axis(angle, axis):
    """
    Returns a rotation matrix of `angle` radians about `axis`.

    Parameters
    ----------
    angle : :class:`float`
        The size of the rotation in radians.

    axis : :class:`numpy.ndarray`
        A 3 element aray which represents a vector. The vector is the
        axis about which the rotation is carried out. Must be of
        unit magnitude.

    Returns
    -------
    :class:`numpy.ndarray`
        A ``3x3`` array representing a rotation matrix.
    """

    a = np.cos(angle/2)
    b, c, d = axis * np.sin(angle/2)

    e11 = np.square(a) + np.square(b) - np.square(c) - np.square(d)
    e12 = 2*(b*c - a*d)
    e13 = 2*(b*d + a*c)

    e21 = 2*(b*c + a*d)
    e22 = np.square(a) + np.square(c) - np.square(b) - np.square(d)
    e23 = 2*(c*d - a*b)

    e31 = 2*(b*d - a*c)
    e32 = 2*(c*d + a*b)
    e33 = np.square(a) + np.square(d) - np.square(b) - np.square(c)

    # Initialize as a scipy Rotation object, which normalizes the
    # matrix and allows for returns as quaternion or alternative
    # type in the future.
    return Rotation.from_matrix(np.array([
        [e11, e12, e13],
        [e21, e22, e23],
        [e31, e32, e33]
    ])).as_matrix()


def vector_angle(vector1, vector2):
    """
    Returns the angle between two vectors in radians.

    Parameters
    ----------
    vector1 : :class:`numpy.ndarray`
        The first vector.

    vector2 : :class:`numpy.ndarray`
        The second vector.

    Returns
    -------
    :class:`float`
        The angle between `vector1` and `vector2` in radians.

    """

    if np.all(np.equal(vector1, vector2)):
        return 0.

    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    # This if statement prevents returns of NaN due to floating point
    # inaccuracy.
    term = numerator/denominator
    if term >= 1.:
        return 0.0
    if term <= -1.:
        return np.pi
    return np.arccos(term)


def orthogonal_vector(vector):
    ortho = np.array([0., 0., 0.])
    for m, val in enumerate(vector):
        if not np.allclose(val, 0, atol=1e-8):
            n = (m+1) % 3
            break
    ortho[n] = vector[m]
    ortho[m] = -vector[n]
    return ortho


def normalize_vector(vector):
    """
    Normalizes the given vector.

    A new vector is returned, the original vector is not modified.

    Parameters
    ----------
    vector : :class:`np.ndarray`
        The vector to be normalized.

    Returns
    -------
    :class:`np.ndarray`
        The normalized vector.

    """

    return np.divide(vector, np.linalg.norm(vector))


def rotation_matrix(vector1, vector2):
    """
    Returns a rotation matrix which transforms `vector1` to `vector2`.

    Multiplying `vector1` by the rotation matrix returned by this
    function yields `vector2`.

    Parameters
    ----------
    vector1 : :class:`numpy.ndarray`
        The vector which needs to be transformed to `vector2`.

    vector2 : :class:`numpy.ndarray`
        The vector onto which `vector1` needs to be transformed.

    Returns
    -------
    :class:`numpy.ndarray`
        A rotation matrix which transforms `vector1` to `vector2`.

    References
    ----------
    http://tinyurl.com/kybj9ox
    http://tinyurl.com/gn6e8mz

    """

    # Make sure both inputs are unit vectors.
    vector1 = normalize_vector(vector1)
    vector2 = normalize_vector(vector2)

    # Hande the case where the input and output vectors are equal.
    if np.allclose(vector1, vector2, atol=1e-8):
        return np.identity(3)

    # Handle the case where the rotation is 180 degrees.
    if np.allclose(vector1, np.multiply(vector2, -1), atol=1e-8):
        return rotation_matrix_arbitrary_axis(
            angle=np.pi,
            axis=orthogonal_vector(vector1)
        )

    v = np.cross(vector1, vector2)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    s = np.linalg.norm(v)
    c = np.dot(vector1, vector2)
    i = np.identity(3)
    mult_factor = (1-c)/np.square(s)

    # Initialize as a scipy Rotation object, which normalizes the
    # matrix and allows for returns as quaternion or alternative
    # type in the future.
    return Rotation.from_matrix(
        i + vx + np.multiply(np.dot(vx, vx), mult_factor)
    ).as_matrix()
