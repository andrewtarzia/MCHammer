def test_atom_get_id(atom_info: tuple) -> None:
    assert atom_info[0].get_id() == atom_info[1]
    assert atom_info[0].id == atom_info[1]


def test_atom_get_element_string(atom_info: tuple) -> None:
    assert atom_info[0].get_element_string() == atom_info[2]
    assert atom_info[0].element_string == atom_info[2]


def test_atom_get_radius(atom_info: tuple) -> None:
    assert atom_info[0].get_radius() == atom_info[3]
    assert atom_info[0].radius == atom_info[3]
