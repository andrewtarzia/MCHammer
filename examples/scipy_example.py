import stk
import mchammer as mch


triangle = stk.BuildingBlock.init_from_file('triangle.mol')
triangle_atoms = [
    (atom.get_id(), atom.__class__.__name__)
    for atom in triangle.get_atoms()
]
triangle_bonds = []
for i, bond in enumerate(triangle.get_bonds()):
    b_ids = (bond.get_atom1().get_id(), bond.get_atom2().get_id())
    triangle_bonds.append((i, b_ids))

mch_mol = mch.Molecule(
    atoms=(
        mch.Atom(id=i[0], element_string=i[1]) for i in triangle_atoms
    ),
    bonds=(
        mch.Bond(id=i[0], atom_ids=i[1])
        for i in triangle_bonds
    ),
    position_matrix=triangle.get_position_matrix(),
)

target_bond_length = 1.2
bond_pair_ids = ((6, 32), (11, 12), (16, 33))
optimizer = mch.ScipyOptimizer(
    step_size=0.25,
    target_bond_length=target_bond_length,
    num_steps=100,
)
subunits = mch_mol.get_subunits(
    bond_pair_ids=bond_pair_ids,
)
# Get all steps.
mch_mol, mch_result = optimizer.get_trajectory(
    mol=mch_mol,
    bond_pair_ids=bond_pair_ids,
    subunits=subunits,
)
triangle = triangle.with_position_matrix(mch_mol.get_position_matrix())
triangle.write('triangle_opt.mol')

with open('triangle_opt.out', 'w') as f:
    f.write(mch_result.get_log())
