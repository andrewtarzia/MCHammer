import matplotlib.pyplot as plt
import os
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

target_bond_length = 1.2
optimizer = mch.Optimizer(
    step_size=0.25,
    target_bond_length=target_bond_length,
    num_steps=100,
)
subunits = optimizer.get_subunits(
    mol=mch_mol,
    bond_pair_ids=((2, 3), (1, 5)),
)
mch_result = optimizer.get_trajectory(
    mol=mch_mol,
    bond_pair_ids=((2, 3), (1, 5)),
    subunits=subunits,
)
benzene = benzene.with_position_matrix(
    mch_result.get_final_position_matrix()
)
benzene.write('benzene_opt.mol')

with open('benzene_opt.out', 'w') as f:
    f.write(mch_result.get_log())

# Output trajectory as separate xyz files for visualisation.
if not os.path.exists('benzene_traj'):
    os.mkdir('benzene_traj')
for step, new_pos_mat in mch_result.get_trajectory():
    mch_mol.update_position_matrix(new_pos_mat)
    mch_mol.write_xyz_file(f'benzene_traj/traj_{step}.xyz')

# Plot properties for parameterisation.
data = {
    'steps': [],
    'max_bond_distances': [],
    'system_potentials': [],
    'nonbonded_potentials': [],
}
for step, prop in mch_result.get_steps_properties():
    data['steps'].append(step)
    data['max_bond_distances'].append(prop['max_bond_distance'])
    data['system_potentials'].append(prop['system_potential'])
    data['nonbonded_potentials'].append(prop['nonbonded_potential'])

# Show plotting from results to viauslise progress.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    data['steps'],
    data['max_bond_distances'],
    c='k', lw=2
)
# Set number of ticks for x-axis
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(0, None)
ax.set_xlabel('step', fontsize=16)
ax.set_ylabel('max long bond length [angstrom]', fontsize=16)
ax.axhline(y=target_bond_length, c='r', linestyle='--')
fig.tight_layout()
fig.savefig(
    f'benzene_maxd_vs_step.pdf',
    dpi=360,
    bbox_inches='tight'
)
plt.close()
# Plot energy vs timestep.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    data['steps'],
    data['system_potentials'],
    c='k', lw=2, label='system potential'
)
ax.plot(
    data['steps'],
    data['nonbonded_potentials'],
    c='r', lw=2, label='nonbonded potential'
)
# Set number of ticks for x-axis
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(0, None)
ax.set_xlabel('step', fontsize=16)
ax.set_ylabel('potential', fontsize=16)
ax.legend(fontsize=16)
fig.tight_layout()
fig.savefig(
    f'benzene_pot_vs_step.pdf',
    dpi=360,
    bbox_inches='tight'
)
plt.close()
