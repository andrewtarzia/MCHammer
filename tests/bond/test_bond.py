def test_bond_get_id(bond_info: tuple) -> None:
    assert bond_info[0].get_id() == bond_info[1]
    assert bond_info[0].id == bond_info[1]


def test_bond_get_atom1_id(bond_info: tuple) -> None:
    assert bond_info[0].get_atom1_id() == bond_info[2]
    assert bond_info[0].atom1_id == bond_info[2]


def test_bond_get_atom2_id(bond_info: tuple) -> None:
    assert bond_info[0].get_atom2_id() == bond_info[3]
    assert bond_info[0].atom2_id == bond_info[3]
