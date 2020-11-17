"""
Atom
====

#. :class:`.Atom`

Atom class.

"""

import logging


logger = logging.getLogger(__name__)


class Atom:
    """
    Atom.


    """

    def __init__(self, id, element_string):
        """
        Initialize a :class:`Atom` instance.

        Parameters
        ----------

        """

        self._id = id
        self._element_string = element_string

    def get_id(self):
        return self._id

    def get_element_string(self):
        return self._element_string

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
