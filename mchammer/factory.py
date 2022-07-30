"""
Factory
=======

#. :class:`.Factory`

Factory class for optimisation.

"""


class Factory:
    """
    Factory to define subunits and potentials.

    """

    def __init__(self, smarts):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """

        self._smarts = tuple(smarts)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} at {id(self)}>'


class BondFactory(Factory):

    def __init__(self, smarts):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """

        super().__init__(smarts)


class AngleFactory(Factory):

    def __init__(self, smarts):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """

        super().__init__(smarts)


class RotatableBondFactory(Factory):

    def __init__(self, smarts):
        """
        Initialize a :class:`Factory` instance.

        Parameters
        ----------
        smarts : :class:`str`

        """

        super().__init__(smarts)
