"""Atom class."""

from .radii import get_radius


class Atom:
    """Atom."""

    def __init__(self, id: int, element_string: str) -> None:  # noqa: A002
        """Initialize a :class:`Atom` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to atom.

        element_string : :class:`str`
            Atom element symbol as string.

        """
        self._id = id
        self._element_string = element_string
        self._radius = get_radius(element_string)

    def get_id(self) -> int:
        """Get atom ID."""
        return self._id

    def get_element_string(self) -> str:
        """Get atom element symbol."""
        return self._element_string

    def get_radius(self) -> float:
        """Get atomic radius (STREUSEL)."""
        return self._radius

    def __str__(self) -> str:
        """String representation of Atom."""
        return repr(self)

    def __repr__(self) -> str:
        """String representation of Atom."""
        return f"{self.get_element_string()}(id={self.get_id()})"
