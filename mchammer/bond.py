"""
Bond
====

#. :class:`.Bond`

Bond class.

"""


class Bond:
    """
    Bond between two atoms.

    """

    def __init__(self, id, atom_ids):
        """
        Initialize a :class:`Bond` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to bond.

        atom_ids : :class:`iterable` of :class:`int`
            IDs of atom 1 and atom 2 in bond, where atom 1 is always
            the smaller number and the IDs cannot be the same.

        """

        self._id = id
        if len(set(atom_ids)) == 0:
            raise ValueError('Two distict atom ids are required.')
        self._atom1_id, self._atom2_id = sorted(atom_ids)

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
