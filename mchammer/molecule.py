"""
Molecule
========

#. :class:`.Molecule`

Molecule class for optimisation.

"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Molecule:
    """
    Molecule to optimize.


    """

    def __init__(self, atoms, bonds, position_matrix):
        """
        Initialize a :class:`Molecule` instance.

        Parameters
        ----------

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)
        self._position_matrix = np.array(
            position_matrix.T,
            dtype=np.float64,
        )

    def get_position_matrix(self):
        return np.array(self._position_matrix.T)

    def update_position_matrix(self, position_matrix):
        self._position_matrix = np.array(position_matrix.T)

    def write_xyz_file(self, path):

        coords = self.get_position_matrix()
        content = [0]
        for i, atom in enumerate(self.get_atoms(), 1):
            x, y, z = (i for i in coords[atom.get_id()])
            content.append(
                f'{atom.get_element_string()} {x:f} {y:f} {z:f}\n'
            )
        # Set first line to the atom_count.
        content[0] = f'{i}\n\n'
        content = ''.join(content)

        with open(path, 'w') as f:
            f.write(''.join(content))

    def get_atoms(self):
        for atom in self._atoms:
            yield atom

    def get_bonds(self):
        for bond in self._bonds:
            yield bond

    def get_centroid(self, atom_ids=None):
        """
        Return the centroid.

        Parameters
        ----------
        atom_ids : :class:`iterable` of :class:`int`, optional
            The ids of atoms which are used to calculate the
            centroid. Can be a single :class:`int`, if a single
            atom is to be used, or ``None`` if all atoms are to be
            used.

        Returns
        -------
        :class:`numpy.ndarray`
            The centroid of atoms specified by `atom_ids`.

        Raises
        ------
        :class:`ValueError`
            If `atom_ids` has a length of ``0``.

        """

        if atom_ids is None:
            atom_ids = range(len(self._atoms))
        elif isinstance(atom_ids, int):
            atom_ids = (atom_ids, )
        elif not isinstance(atom_ids, (list, tuple)):
            atom_ids = list(atom_ids)

        if len(atom_ids) == 0:
            raise ValueError('atom_ids was of length 0.')

        return np.divide(
            self._position_matrix[:, atom_ids].sum(axis=1),
            len(atom_ids)
        )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
