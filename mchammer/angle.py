"""
Angle
=====

#. :class:`.Angle`

Angle class.

"""


class Angle:
    """
    Angle between three atoms.

    """

    def __init__(self, id, atom_ids):
        """
        Initialize a :class:`Angle` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to bond.

        atom_ids : :class:`iterable` of :class:`int`
            IDs of atom 1, atom 2 and atom 3 in angle, where a bond
            is between atom 1 and atom 2, and atom 2 and atom 3.

        """

        self._id = id
        if len(set(atom_ids)) != 3:
            raise ValueError('Three distict atom ids are required.')
        self._atom1_id, self._atom2_id, self._atom3_id = atom_ids

    def get_id(self):
        """
        Get angle ID.

        """

        return self._id

    def get_atom1_id(self):
        """
        Get ID of atom 1 in angle.

        """

        return self._atom1_id

    def get_atom2_id(self):
        """
        Get ID of atom 2 in angle.

        """

        return self._atom2_id

    def get_atom3_id(self):
        """
        Get ID of atom 3 in angle.

        """

        return self._atom3_id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(id={self.get_id()}, '
            f'atom1_id={self.get_atom1_id()}, '
            f'atom2_id={self.get_atom2_id()}, '
            f'atom3_id={self.get_atom3_id()})'
        )
