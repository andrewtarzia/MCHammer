"""
Factory
=======

#. :class:`.Factory`

Factory class for optimisation.

"""

from .substructure import (
    BondSubstructure,
    AngleSubstructure,
    RotatableSubstructure,
)
from .utilities import get_atom_ids, get_atom_distance


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

        """

        # From rdkit: rdkit/Chem/Lipinski.py, but includes neighbours.
        # Also removes the limit on ring bonds because molecules in
        # cages have ring bonds.
        self._rotatable_smarts = (
            '[*]~[!$(*#*)&!D1]-[!$(*#*)&!D1]~[*]'
        )

    def get_substructures(self, molecule):
        """
        Get substructures in molecule.

        """

        position_matrix = molecule.get_position_matrix()

        ids = get_atom_ids(self._rotatable_smarts, molecule)
        collected_ids = set()
        for atom_ids in ids:
            rotatable_ids = tuple(sorted(atom_ids[1: 3]))
            if rotatable_ids not in collected_ids:
                yield RotatableSubstructure(
                    atom_ids=atom_ids,
                    disconnectors=((1, 2), ),
                    target=None,
                )
                collected_ids.add(rotatable_ids)
                yield BondSubstructure(
                    atom_ids=atom_ids[1: 3],
                    disconnectors=((0, 1), ),
                    target=get_atom_distance(
                        position_matrix=position_matrix,
                        atom1_id=atom_ids[1],
                        atom2_id=atom_ids[2],
                    ),
                )
