
def test_bond_get_id(bond_info):
    assert bond_info[0].get_id() == bond_info[1]


def test_bond_get_atom1_id(bond_info):
    assert bond_info[0].get_atom1_id() == bond_info[2]


def test_bond_get_atom2_id(bond_info):
    assert bond_info[0].get_atom2_id() == bond_info[3]
