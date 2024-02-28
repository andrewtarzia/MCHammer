import os  # noqa: INP001, D100
from collections import defaultdict

import mchammer as mch
import stk


def get_long_bond_ids(mol: mch.Molecule) -> tuple:
    """Find long bonds in stk.ConstructedMolecule."""
    long_bond_ids = []
    for bond_infos in mol.get_bond_infos():
        if bond_infos.get_building_block() is None:
            ids = (
                bond_infos.get_bond().get_atom1().get_id(),
                bond_infos.get_bond().get_atom2().get_id(),
            )
            long_bond_ids.append(ids)

    return tuple(long_bond_ids)


def get_subunits(mol: mch.Molecule) -> dict:
    """Get connected graphs based on building block ids.

    Returns:
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
    smiles="O=CC(C=O)C=O",
    functional_groups=[stk.AldehydeFactory()],
)
bb2 = stk.BuildingBlock(
    smiles="O=CC(Cl)(C=O)C=O",
    functional_groups=[stk.AldehydeFactory()],
)
bb3 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])
bb4 = stk.BuildingBlock.init_from_file(
    "some_complex.mol",
    functional_groups=[stk.PrimaryAminoFactory()],
)
bb5 = stk.BuildingBlock("NCCCCN", [stk.PrimaryAminoFactory()])

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
cage.write("poc.mol")
stk_long_bond_ids = get_long_bond_ids(cage)
mch_mol = mch.Molecule(
    atoms=(
        mch.Atom(
            id=atom.get_id(),
            element_string=atom.__class__.__name__,
        )
        for atom in cage.get_atoms()
    ),
    bonds=(
        mch.Bond(
            id=i,
            atom_ids=(
                bond.get_atom1().get_id(),
                bond.get_atom2().get_id(),
            ),
        )
        for i, bond in enumerate(cage.get_bonds())
    ),
    position_matrix=cage.get_position_matrix(),
)

optimizer = mch.Collapser(
    step_size=0.05,
    distance_threshold=1.2,
    scale_steps=True,
)
subunits = get_subunits(mol=cage)
# Get all steps.
mch_mol, mch_result = optimizer.get_trajectory(
    mol=mch_mol,
    bond_pair_ids=stk_long_bond_ids,
    subunits=subunits,
)

cage = cage.with_position_matrix(mch_result.get_final_position_matrix())
cage.write("coll_poc_opt.mol")

with open("coll_opt.out", "w") as f:
    f.write(mch_result.get_log())

# Output trajectory as separate xyz files for visualisation.
if not os.path.exists("coll_poc_traj"):  # noqa: PTH110
    os.mkdir("coll_poc_traj")  # noqa: PTH102
for step, new_pos_mat in mch_result.get_trajectory():
    new_mol = mch_mol.with_position_matrix(new_pos_mat)
    new_mol.write_xyz_file(f"coll_poc_traj/traj_{step}.xyz")
