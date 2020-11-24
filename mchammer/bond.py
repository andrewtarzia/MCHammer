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

    def __init__(self, id, atom1_id, atom2_id):
        """
        Initialize a :class:`Bond` instance.

        Parameters
        ----------

        """

        self._id = id
        self._atom1_id = atom1_id
        self._atom2_id = atom2_id

    def get_id(self):
        return self._id

    def get_atom1_id(self):
        return self._atom1_id

    def get_atom2_id(self):
        return self._atom2_id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
