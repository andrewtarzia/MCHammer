"""Bond class."""

from __future__ import annotations


class Bond:
    """Bond between two atoms."""

    def __init__(
        self,
        id: int,  # noqa: A002
        atom_ids: tuple[int, int],
    ) -> None:
        """Initialize a :class:`Bond` instance.

        Parameters
        ----------
        id : :class:`int`
            ID to be assigned to bond.

        atom_ids : :class:`iterable` of :class:`int`
            IDs of atom 1 and atom 2 in bond, where atom 1 is always
            the smaller number and the IDs cannot be the same.

        """
        self._id = id
        if len(set(atom_ids)) == 0:
            msg = "Two distict atom ids are required."
            raise ValueError(msg)
        self._atom1_id, self._atom2_id = sorted(atom_ids)

    def get_id(self) -> int:
        """Get bond ID."""
        return self._id

    def get_atom1_id(self) -> int:
        """Get ID of atom 1 in bond."""
        return self._atom1_id

    def get_atom2_id(self) -> int:
        """Get ID of atom 2 in bond."""
        return self._atom2_id

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
