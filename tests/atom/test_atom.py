from mchammer import Atom

atom = Atom(id=0, element_string='N')


def test_atom_get_id():
    assert atom.get_id() == 0


def test_atom_get_element_string():
    assert atom.get_element_string() == 'N'
