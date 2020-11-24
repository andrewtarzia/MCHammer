"""
Atom
====

#. :class:`.Atom`

Atom class.

"""


class Atom:
    """
    Atom.

    """

    def __init__(self, id, element_string):
        """
        Initialize a :class:`Atom` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to atom.

        element_string : :class:`str`
            Atom element symbol as string.

        """

        self._id = id
        self._element_string = element_string

    def get_id(self):
        """
        Get atom ID.

        """
        return self._id

    def get_element_string(self):
        """
        Get atom element symbol.

        """
        return self._element_string

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
