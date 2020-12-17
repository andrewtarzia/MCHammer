import stk
import os
import mchammer as mch


def get_long_bond_ids(mol):
    """
    Find long bonds in stk.ConstructedMolecule.

    """
    long_bond_ids = []
    for bond_infos in mol.get_bond_infos():
        if bond_infos.get_building_block() is None:
            ids = (
                bond_infos.get_bond().get_atom1().get_id(),
                bond_infos.get_bond().get_atom2().get_id(),
            )
            long_bond_ids.append(ids)

    return tuple(long_bond_ids)


def merge_subunits_by_buildingblockid(mol, subunits):
    """
    Merge subunits in stk.Molecule by building block ids.

    """

    subunit_building_block_ids = {i: set() for i in subunits}
    for su in subunits:
        su_ids = subunits[su]
        for i in su_ids:
            atom_info = next(mol.get_atom_infos(atom_ids=i))
            subunit_building_block_ids[su].add(
                atom_info.get_building_block_id()
            )

    new_subunits = {}
    taken_subunits = set()
    for su in subunits:
        bb_ids = subunit_building_block_ids[su]
        if len(bb_ids) > 1:
            raise ValueError(
                'Subunits not made up of singular BuildingBlock'
            )
        bb_id = list(bb_ids)[0]
        if su in taken_subunits:
            continue

        compound_subunit = subunits[su]
        has_same_bb_id = [
            (su_id, bb_id) for su_id in subunits
            if list(subunit_building_block_ids[su_id])[0] == bb_id
            and su_id != su
        ]

        for su_id, bb_id in has_same_bb_id:
            for i in subunits[su_id]:
                compound_subunit.add(i)
            taken_subunits.add(su_id)
        new_subunits[su] = compound_subunit

    return new_subunits


# Building a cage from the examples on the stk docs.
bb1 = stk.BuildingBlock(
    smiles='O=CC(C=O)C=O',
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock(
    smiles='O=CC(Cl)(C=O)C=O',
    functional_groups=[stk.AldehydeFactory()],
)
bb3 = stk.BuildingBlock('NCCN', [stk.PrimaryAminoFactory()])
bb4 = stk.BuildingBlock.init_from_file(
    'some_complex.mol',
    functional_groups=[stk.PrimaryAminoFactory()],
)
bb5 = stk.BuildingBlock('NCCCCN', [stk.PrimaryAminoFactory()])

cage = stk.ConstructedMolecule(
    topology_graph=stk.cage.FourPlusSix(
        # building_blocks is now a dict, which maps building
        # blocks to the id of the vertices it should be placed
        # on. You can use ranges to specify the ids.
        building_blocks={
            bb1: range(2),
            bb2: (2, 3),
            bb3: 4,
            bb4: 5,
            bb5: range(6, 10),
        },
    ),
)
cage.write('poc.mol')
stk_long_bond_ids = get_long_bond_ids(cage)
mch_mol = mch.Molecule(
    atoms=(
        mch.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        ) for atom in cage.get_atoms()
    ),
    bonds=(
        mch.Bond(
            id=i,
            atom1_id=bond.get_atom1().get_id(),
            atom2_id=bond.get_atom2().get_id()
        ) for i, bond in enumerate(cage.get_bonds())
    ),
    position_matrix=cage.get_position_matrix(),
)

optimizer = mch.Collapser(
    step_size=0.05,
    distance_threshold=1.2,
    scale_steps=True,
)
subunits = mch_mol.get_subunits(
    bond_pair_ids=stk_long_bond_ids,
)
# Just get final step.
mch_result = optimizer.get_trajectory(
    mol=mch_mol,
    bond_pair_ids=stk_long_bond_ids,
    subunits=merge_subunits_by_buildingblockid(cage, subunits),
)

cage = cage.with_position_matrix(
    mch_result.get_final_position_matrix()
)
cage.write('coll_poc_opt.mol')

with open('coll_opt.out', 'w') as f:
    f.write(mch_result.get_log())

# Output trajectory as separate xyz files for visualisation.
if not os.path.exists('coll_poc_traj'):
    os.mkdir('coll_poc_traj')
for step, new_pos_mat in mch_result.get_trajectory():
    mch_mol.update_position_matrix(new_pos_mat)
    mch_mol.write_xyz_file(f'coll_poc_traj/traj_{step}.xyz')
