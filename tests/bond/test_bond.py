from mchammer import Bond

bond = Bond(id=0, atom1_id=0, atom2_id=1)


def test_bond_get_id():
    assert bond.get_id() == 0


def test_bond_get_atom1_id():
    assert bond.get_atom1_id() == 0


def test_bond_get_atom2_id():
    assert bond.get_atom2_id() == 1
