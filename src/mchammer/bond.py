"""Bond class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bond:
    """Bond between two atoms.

    Attributes:
        id : :class:`int`
            ID to be assigned to bond.

        atom_ids : :class:`iterable` of :class:`int`
            IDs of atom 1 and atom 2 in bond, where atom 1 is always
            the smaller number and the IDs cannot be the same.

    """

    id: int
    atom_ids: tuple[int, int]

    def __post_init__(self) -> None:
        """Post initialization of bond."""
        if len(set(self.atom_ids)) != 2:  # noqa: PLR2004
            msg = "Two distinct atom ids are required."
            raise ValueError(msg)
        self.atom1_id, self.atom2_id = sorted(self.atom_ids)

    def get_id(self) -> int:
        """Get bond ID."""
        return self.id

    def get_atom1_id(self) -> int:
        """Get ID of atom 1 in bond."""
        return self.atom1_id

    def get_atom2_id(self) -> int:
        """Get ID of atom 2 in bond."""
        return self.atom2_id

    def __str__(self) -> str:
        """String representation of Bond."""
        return repr(self)

    def __repr__(self) -> str:
        """String representation of Bond."""
        return (
            f"{self.__class__.__name__}(id={self.get_id()}, "
            f"atom1_id={self.get_atom1_id()}, "
            f"atom2_id={self.get_atom2_id()})"
        )
