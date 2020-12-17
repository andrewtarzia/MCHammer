"""
Bond
====

#. :class:`.Bond`

Bond class.

"""


class AtomIDOrderingError(Exception):
    ...


class Bond:
    """
    Bond between two atoms.

    """

    def __init__(self, id, atom1_id, atom2_id):
        """
        Initialize a :class:`Bond` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to bond.

        atom1_id : :class:`int`
            ID of atom 1 in bond.


        atom2_id : :class:`int`
            ID of atom 2 in bond.

        """

        self._id = id
        if atom1_id > atom2_id:
            raise AtomIDOrderingError(
                f'Atom 1 ID ({atom1_id}) must be less than Atom 2 ID'
                f'({atom2_id}).'
            )

        self._atom1_id = atom1_id
        self._atom2_id = atom2_id

    def get_id(self):
        """
        Get bond ID.

        """

        return self._id

    def get_atom1_id(self):
        """
        Get ID of atom 1 in bond.

        """

        return self._atom1_id

    def get_atom2_id(self):
        """
        Get ID of atom 2 in bond.

        """

        return self._atom2_id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(id={self.get_id()}, '
            f'atom1_id={self.get_atom1_id}, '
            f'atom2_id={self.get_atom2_id})'
        )
