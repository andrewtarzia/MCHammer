def test_atom_get_id(atom_info: tuple) -> None:
    assert atom_info[0].get_id() == atom_info[1]


def test_atom_get_element_string(atom_info: tuple) -> None:
    assert atom_info[0].get_element_string() == atom_info[2]


def test_atom_get_radius(atom_info: tuple) -> None:
    assert atom_info[0].get_radius() == atom_info[3]
