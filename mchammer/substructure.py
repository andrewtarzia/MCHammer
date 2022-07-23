"""
Substructure
============

#. :class:`.Substructure`
#. :class:`.SubstructureFactory`

"""


class Substructure:
    """
    Substructure.

    """

    def __init__(self, atom_ids):
        """
        Initialize a :class:`Substructure` instance.

        Parameters
        ----------
        atom_ids : :class:`tuple` of :class:`int`
            Atom ids in substructure.

        """

        self._atom_ids = atom_ids

    def get_atom_ids(self):
        """
        Get atom ID.

        """
        return self._atom_ids

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'{self.__class__.__name__}(ids={self.get_atom_ids()})'
