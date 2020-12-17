import stk
import mchammer as mch


benzene = stk.BuildingBlock.init_from_file('benzene.mol')
benzene_atoms = [
    (atom.get_id(), atom.__class__.__name__)
    for atom in benzene.get_atoms()
]
benzene_bonds = []
for i, bond in enumerate(benzene.get_bonds()):
    # Must ensure that bond atom ids are ordered by atom id.
    b_ids = (
        (bond.get_atom1().get_id(), bond.get_atom2().get_id())
        if bond.get_atom1().get_id() < bond.get_atom2().get_id()
        else (bond.get_atom2().get_id(), bond.get_atom1().get_id())
    )
    benzene_bonds.append((i, b_ids))

mch_mol = mch.Molecule(
    atoms=(
        mch.Atom(id=i[0], element_string=i[1]) for i in benzene_atoms
    ),
    bonds=(
        mch.Bond(id=i[0], atom1_id=i[1][0], atom2_id=i[1][1])
        for i in benzene_bonds
    ),
    position_matrix=benzene.get_position_matrix(),
)

optimizer = mch.Optimizer(
    output_dir='benzene_mch',
    step_size=0.25,
    target_bond_length=1.2,
    num_steps=100,
)
subunits = optimizer.get_subunits(
    mol=mch_mol,
    bond_pair_ids=((2, 3), (1, 5)),
)
mch_mol = optimizer.optimize(
    mol=mch_mol,
    bond_pair_ids=((2, 3), (1, 5)),
    subunits=subunits,
)
benzene = benzene.with_position_matrix(mch_mol.get_position_matrix())
benzene.write('benzene_opt.mol')
