"""Atom class."""

from __future__ import annotations

from dataclasses import dataclass

from .radii import get_radius


@dataclass
class Atom:
    """Atom.

    Attributes:
        id:
            ID to be assigned to atom.

        element_string:
            Atom element symbol as string.

        radius:
            Radius (Default is from STREUSSEL) in angstrom.

        sigma:
            Value of sigma for custom potentials.

        epsilon:
            Value of epsilon for custom potentials.

        charge:
            Value of atomic charge for custom potentials.

    """

    id: int
    element_string: str
    radius: float | None = None
    sigma: float | None = None
    epsilon: float | None = None
    charge: float | None = None

    def __post_init__(self) -> None:
        """Post initialization of atom."""
        if self.radius is None:
            self.radius = get_radius(self.element_string)

    def get_id(self) -> int:
        """Get atom ID."""
        return self.id

    def get_element_string(self) -> str:
        """Get atom element symbol."""
        return self.element_string

    def get_radius(self) -> float | None:
        """Get atomic radius (STREUSEL)."""
        return self.radius

    def __str__(self) -> str:
        """String representation of Atom."""
        return repr(self)

    def __repr__(self) -> str:
        """String representation of Atom."""
        return f"{self.get_element_string()}(id={self.get_id()})"
