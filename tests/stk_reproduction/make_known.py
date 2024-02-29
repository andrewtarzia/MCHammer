import stk

bb1 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])
bb2 = stk.BuildingBlock("O=CCC=O", [stk.AldehydeFactory()])

polymer = stk.ConstructedMolecule(
    topology_graph=stk.polymer.Linear(
        building_blocks=(bb1, bb2),
        repeating_unit="AB",
        num_repeating_units=6,
        optimizer=stk.MCHammer(),
    ),
)
polymer.write("mchammer.mol")


bb1 = stk.BuildingBlock("NCCN", [stk.PrimaryAminoFactory()])
bb2 = stk.BuildingBlock("O=CCC=O", [stk.AldehydeFactory()])

polymer = stk.ConstructedMolecule(
    topology_graph=stk.polymer.Linear(
        building_blocks=(bb1, bb2),
        repeating_unit="AB",
        num_repeating_units=2,
        optimizer=stk.Collapser(),
    ),
)
polymer.write("collapser.mol")
