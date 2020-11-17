"""
Molecule
========

#. :class:`.Molecule`

Molecule class for optimisation.

"""

import logging

logger = logging.getLogger(__name__)


class Molecule:
    """
    Molecule to optimize.


    """

    def __init__(self, atoms, bonds):
        """
        Initialize a :class:`Molecule` instance.

        Parameters
        ----------

        """

        self._atoms = tuple(atoms)
        self._bonds = tuple(bonds)

    def get_atoms(self):
        for atom in self._atoms:
            yield atom

    def get_bonds(self):
        for bond in self._bonds:
            yield bond

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
