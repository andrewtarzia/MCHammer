import pathlib

import mchammer as mch
import numpy as np
import stk

from .utilities import get_long_bond_ids, get_mch_bonds, get_subunits


def test_stk_collapser() -> None:
    bb1 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])
    bb2 = stk.BuildingBlock("O=CCC=O", [stk.AldehydeFactory()])

    polymer_opt = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(bb1, bb2),
            repeating_unit="AB",
            num_repeating_units=2,
            optimizer=stk.Collapser(),
        ),
    )

    polymer_unopt = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(bb1, bb2),
            repeating_unit="AB",
            num_repeating_units=2,
        ),
    )
    mch_mol = mch.Molecule(
        atoms=(
            mch.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
            )
            for atom in polymer_unopt.get_atoms()
        ),
        bonds=get_mch_bonds(polymer_unopt),
        position_matrix=polymer_unopt.get_position_matrix(),
    )

    # Run optimization.
    optimizer = mch.Collapser(
        step_size=0.1,
        distance_threshold=1.5,
        scale_steps=True,
    )
    mch_mol, result = optimizer.get_result(
        mol=mch_mol,
        bond_pair_ids=tuple(get_long_bond_ids(polymer_unopt)),
        subunits=get_subunits(polymer_unopt),
    )
    new_polymer = polymer_unopt.with_position_matrix(
        position_matrix=mch_mol.get_position_matrix()
    )
    print("test", polymer_opt.get_position_matrix())
    print("new", new_polymer.get_position_matrix())
    assert np.all(
        np.equal(
            polymer_opt.get_position_matrix(),
            new_polymer.get_position_matrix(),
        )
    )

    known = stk.BuildingBlock.init_from_file(
        pathlib.Path(__file__).resolve().parent / "collapser.mol"
    )
    print("test", known.get_position_matrix())
    print("new", new_polymer.get_position_matrix())
    assert np.all(
        np.equal(
            known.get_position_matrix(),
            new_polymer.get_position_matrix(),
        )
    )


def test_stk_mchammer() -> None:
    bb1 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])
    bb2 = stk.BuildingBlock("O=CCC=O", [stk.AldehydeFactory()])

    polymer_opt = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(bb1, bb2),
            repeating_unit="AB",
            num_repeating_units=6,
            optimizer=stk.MCHammer(),
        ),
    )

    polymer_unopt = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(bb1, bb2),
            repeating_unit="AB",
            num_repeating_units=6,
        ),
    )
    mch_mol = mch.Molecule(
        atoms=(
            mch.Atom(
                id=atom.get_id(),
                element_string=atom.__class__.__name__,
            )
            for atom in polymer_unopt.get_atoms()
        ),
        bonds=tuple(get_mch_bonds(polymer_unopt)),
        position_matrix=polymer_unopt.get_position_matrix(),
    )

    # Run optimization.
    optimizer = mch.Optimizer(
        step_size=0.25,
        target_bond_length=1.2,
        num_steps=500,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=1000,
    )
    mch_mol, _ = optimizer.get_result(
        mol=mch_mol,
        bond_pair_ids=tuple(get_long_bond_ids(polymer_unopt)),
        subunits=get_subunits(polymer_unopt),
    )
    new_polymer = polymer_unopt.with_position_matrix(
        position_matrix=mch_mol.get_position_matrix()
    )
    print("test", polymer_opt.get_position_matrix())
    print("new", new_polymer.get_position_matrix())
    assert np.all(
        np.equal(
            polymer_opt.get_position_matrix(),
            new_polymer.get_position_matrix(),
        )
    )

    known = stk.BuildingBlock.init_from_file(
        pathlib.Path(__file__).resolve().parent / "mchammer.mol"
    )
    print("test", known.get_position_matrix())
    print("new", new_polymer.get_position_matrix())
    assert np.all(
        np.equal(
            known.get_position_matrix(),
            new_polymer.get_position_matrix(),
        )
    )
