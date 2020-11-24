MCHammer
========

:author: Andrew Tarzia

A Monte Carlo-based molecular optimizer for optimizing the length of specified bonds in a molecule toward a target using cheap and unphysical potentials.

Please contact me with any questions (<andrew.tarzia@gmail.com>) or submit an issue!

Installation
------------

Install using pip:

    pip install XXXXX

Algorithm
---------

.. image:: https://raw.githubusercontent.com/andrewtarzia/MCHammer/main/docs/workflow.png?sanitize=true

Examples
--------

This code was originally written for use with *stk* (<https://stk.readthedocs.io/>), which assembles structures with long bonds that we wanted to optimize quickly.
Now it has been generalized to take any molecule (defined by atoms and bonds) and a set of bonds to optimize to some target bond length.
The algorithm is unphysical in that the bonded and nonbonded potential we apply is meaningless, other than to give a reasonable structure and avoid steric clashes!

In this example, we use *stk* for I/O only with the input file available in examples/


.. code-block:: python

    # Load in molecule with stk.
    benzene = stk.BuildingBlock.init_from_file('benzene.mol')
    # Define atoms and bonds.
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

    # Define MCHammer molecule.
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

    # Define optimizer.
    optimizer = mch.Optimizer(
        output_dir='benzene_mch',
        step_size=0.25,
        target_bond_length=1.2,
        num_steps=100,
    )
    mch_mol = optimizer.optimize(
        mol=mch_mol,
        bond_pair_ids=((2, 3), (1, 5)),
    )

    # Update stk molecule and write to file.
    benzene = benzene.with_position_matrix(mch_mol.get_position_matrix())
    benzene.write('benzene_opt.mol')



Contributors
------------

Lukas Turcani, Steven Bennett

License
-------

The project is licensed under the MIT license.
