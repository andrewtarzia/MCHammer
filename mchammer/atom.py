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

    def __init__(self, id):
        """
        Initialize a :class:`Atom` instance.

        Parameters
        ----------

        """

        self._id = id

    def get_id(self):
        return self._id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'
