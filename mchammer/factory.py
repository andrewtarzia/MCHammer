"""
Factory
=======

#. :class:`.Factory`

Factory class for optimisation.

"""

from .substructure import BondSubstructure, AngleSubstructure
from .utilities import get_atom_ids


class Factory:
    """
    Factory to define subunits and potentials.

    """

    def __init__(self):
        """
        Initialize a :class:`Factory` instance.

        """

    def get_substructures(self, molecule):
        """
        Get substructures in molecule.

        """
        raise NotImplementedError()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'


class BondFactory(Factory):

    def __init__(self, smarts, disconnectors, target):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        disconnectors : :class:`tuple` of :class:`int`

        target : :class:`float`

        """

        self._smarts = smarts
        self._disconnectors = disconnectors
        self._target = target

    def get_substructures(self, molecule):
        """
        Get substructures in molecule.

        """

        ids = get_atom_ids(self._smarts, molecule)
        for atom_ids in ids:
            yield BondSubstructure(
                atom_ids=atom_ids,
                disconnectors=self._disconnectors,
                target=self._target,
            )


class LongBondFactory(Factory):

    def __init__(self, smarts):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """

        super().__init__(smarts)


class AngleFactory(Factory):

    def __init__(self, smarts, disconnectors, target):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        disconnectors : :class:`tuple` of :class:`int`

        target : :class:`float`

        """

        self._smarts = smarts
        self._disconnectors = disconnectors
        self._target = target

    def get_substructures(self, molecule):
        """
        Get substructures in molecule.

        """

        ids = get_atom_ids(self._smarts, molecule)
        for atom_ids in ids:
            yield AngleSubstructure(
                atom_ids=atom_ids,
                disconnectors=self._disconnectors,
                target=self._target,
            )


class RotatableBondFactory(Factory):

    def __init__(self):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """
