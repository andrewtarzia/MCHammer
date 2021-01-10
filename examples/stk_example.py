from copy import deepcopy
from collections import defaultdict
import stk
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


def get_subunits(mol):
    """
    Get connected graphs based on building block ids.

    Returns
    -------
    subunits : :class:`.dict`
        The subunits of `mol` split by building block id. Key is
        subunit identifier, Value is :class:`iterable` of atom ids in
        subunit.

    """

    subunits = defaultdict(list)
    for atom_info in mol.get_atom_infos():
        subunits[atom_info.get_building_block_id()].append(
            atom_info.get_atom().get_id()
        )

    return subunits


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
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            )
        ) for i, bond in enumerate(cage.get_bonds())
    ),
    position_matrix=cage.get_position_matrix(),
)
mch_mol_nci = deepcopy(mch_mol)

optimizer = mch.Optimizer(
    step_size=0.25,
    target_bond_length=1.2,
    num_steps=500,
)
subunits = mch_mol.get_subunits(
    bond_pair_ids=stk_long_bond_ids,
)
# Just get final step.
mch_mol, mch_result = optimizer.get_result(
    mol=mch_mol,
    bond_pair_ids=stk_long_bond_ids,
    subunits=subunits,
)


optimizer = mch.Optimizer(
    step_size=0.25,
    target_bond_length=1.2,
    num_steps=500,
)
subunits = get_subunits(mol=cage)
# Just get final step.
mch_mol_nci, mch_result_nci = optimizer.get_result(
    mol=mch_mol_nci,
    bond_pair_ids=stk_long_bond_ids,
    # Can merge subunits to match distinct BuildingBlocks in stk
    # ConstructedMolecule.
    subunits=subunits,
)
print(mch_result_nci)
cage = cage.with_position_matrix(mch_mol.get_position_matrix())
cage.write('poc_opt.mol')
cage = cage.with_position_matrix(mch_mol_nci.get_position_matrix())
cage.write('poc_opt_nci.mol')
