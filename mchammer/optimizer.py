"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimise intermolecular distances.

"""

import logging
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
import networkx as nx
import uuid
import os
import shutil

from .utilities import (
    get_atom_distance,
    rotation_matrix_arbitrary_axis,
    normalize_vector,
)


logger = logging.getLogger(__name__)


class Optimizer:
    """
    Collapse molecule to decrease enlarged bonds using MC algorithm.

    Smarter optimisation than Collapser using simple Monte Carlo
    algorithm to perform rigid translations of building blocks.

    This optimizer aims to bring extended bonds closer together for
    further optimisation.

    .. code-block:: python

        import stk
        import stko

        bb1 = stk.BuildingBlock('NCCN', [stk.PrimaryAminoFactory()])
        bb2 = stk.BuildingBlock(
            smiles='O=CC(C=O)C=O',
            functional_groups=[stk.AldehydeFactory()],
        )
        cage1 = stk.ConstructedMolecule(
            topology_graph=stk.cage.FourPlusSix((bb1, bb2)),
        )

        # Perform collapser optimisation.
        output_dir = f'cage_opt_{cage_name}_coll'
        optimizer = stko.Collapser(
            output_dir=output_dir,
            step_size=0.05,
            distance_cut=2.0,
            scale_steps=True,
        )
        cage1 = optimizer.optimize(mol=cage1)


    """

    def __init__(
        self,
        output_dir,
        step_size,
        rotation_step_size,
        target_bond_length,
        num_steps,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=None,
        use_neighbour_list=True,
    ):
        """
        Initialize a :class:`Collapser` instance.

        Parameters
        ----------
        output_dir : :class:`str`
            The name of the directory into which files generated during
            the calculation are written, if ``None`` then
            :func:`uuid.uuid4` is used.

        step_size : :class:`float`
            The relative size of the step to take during step.

        rotation_step_size : :class:`float`
            Maximum rotation step to take in radians.

        target_bond_length : :class:`float`
            Target equilibrium bond length for long bonds to minimize
            to.

        num_steps : :class:`int`
            Number of MC moves to perform.

        bond_epsilon : :class:`float`, optional
            Value of epsilon used in the bond potential in MC moves.
            Determines strength of the bond potential.
            Defaults to 50.

        nonbond_epsilon : :class:`float`, optional
            Value of epsilon used in the nonbond potential in MC moves.
            Determines strength of the nonbond potential.
            Defaults to 20.

        nonbond_sigma : :class:`float`, optional
            Value of sigma used in the nonbond potential in MC moves.
            Defaults to 1.2.

        nonbond_mu : :class:`float`, optional
            Value of mu used in the nonbond potential in MC moves.
            Determines the steepness of the nonbond potential.
            Defaults to 3.

        beta : :class:`float`, optional
            Value of beta used in the in MC moves. Beta takes the
            place of the inverse boltzmann temperature.
            Defaults to 2.

        random_seed : :class:`int`, optional
            Random seed to use for MC algorithm. Should only be set
            if exactly reproducible results are required, otherwise
            a system-based random seed should be used for proper
            sampling.

        use_neighbour_list : :class:`bool`, optional
            ``True`` to use neighbour list defined by immediately
            connected sub units, which are defined by bonds to be
            optimized. Neighbour lists will speed up the algorithm
            but may lead to steric clashes in complex systems.
            Defaults to ``True``.

        """

        self._output_dir = output_dir
        self._step_size = step_size
        self._rotation_step_size = rotation_step_size
        self._target_bond_length = target_bond_length
        self._num_steps = num_steps
        self._bond_epsilon = bond_epsilon
        self._nonbond_epsilon = nonbond_epsilon
        self._nonbond_sigma = nonbond_sigma
        self._nonbond_mu = nonbond_mu
        self._beta = beta
        if random_seed is None:
            random.seed()
        else:
            random.seed(random_seed)
        self._use_neighbour_list = use_neighbour_list

    def _get_bond_vector(self, position_matrix, bond_ids):
        atom1_pos = position_matrix[bond_ids[0]]
        atom2_pos = position_matrix[bond_ids[1]]
        return atom2_pos - atom1_pos

    def _get_cent_to_lb_vector(
        self,
        mol,
        bb_centroids,
        long_bond_infos
    ):
        """
        Returns dict of long bond atom to bb centroid vectors.

        """

        position_matrix = mol.get_position_matrix()
        centroid_to_lb_vectors = {}
        for bb in bb_centroids:
            cent = bb_centroids[bb]
            for b_atom_ids, bond_info in long_bond_infos.items():
                for atom_id in b_atom_ids:
                    atom_info, = mol.get_atom_infos(atom_ids=atom_id)
                    atom_pos = position_matrix[atom_id]
                    if atom_info.get_building_block_id() == bb:
                        centroid_to_lb_vectors[(bb, atom_id)] = (
                            atom_pos - cent,
                        )
                        break

        return centroid_to_lb_vectors

    def _bond_potential(self, distance):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        potential = (distance - self._target_bond_length) ** 2
        potential = self._bond_epsilon * potential

        return potential

    def _non_bond_potential(self, distance):
        """
        Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (self._nonbond_sigma/distance) ** self._nonbond_mu
            )
        )

    def _compute_non_bonded_potential(self, mol):

        # Get all pairwise distances.
        pair_dists = pdist(mol.get_position_matrix())
        non_bonded_potential = self._non_bond_potential(pair_dists)
        t1 = time.time()
        print(
            f'nbp1: '
            f'{time.time() - t1}'
        )

        return sum(non_bonded_potential)

    def _compute_potential(self, mol, bonds, neighour_lists):

        system_potential = self._compute_non_bonded_potential(
            mol,
            neighour_lists
        )
        position_matrix = mol.get_position_matrix()
        for bond in bonds:
            system_potential += self._bond_potential(
                distance=get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
            )

        return system_potential

    def _translate_atoms_along_vector(self, mol, atom_ids, vector):

        new_position_matrix = mol.get_position_matrix()
        for atom in mol.get_atoms():
            if atom.get_id() not in atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - vector

        mol.update_position_matrix(new_position_matrix)
        return mol

    def _rotate_atoms_by_angle(
        self,
        mol,
        atom_ids,
        angle,
        axis,
        origin
    ):
        ori_position_matrix = mol.get_position_matrix()
        new_position_matrix = mol.get_position_matrix()
        # Set the origin of the rotation to "origin".
        new_position_matrix = new_position_matrix - origin
        # Perform rotation.
        rot_mat = rotation_matrix_arbitrary_axis(angle, axis)
        # Apply the rotation matrix on the position matrix, to get the
        # new position matrix.
        new_position_matrix = (rot_mat @ new_position_matrix.T).T
        # Return the centroid of the molecule to the original position.
        new_position_matrix = new_position_matrix + origin

        # Reinstate NOT atom_ids with original position matrix.
        for atom in mol.get_atoms():
            if atom.get_id() not in atom_ids:
                new_position_matrix[atom.get_id()] = (
                    ori_position_matrix[atom.get_id()]
                )

        mol.update_position_matrix(new_position_matrix)
        return mol

    def _test_move(self, curr_pot, new_pot):

        if new_pot < curr_pot:
            return True
        else:
            exp_term = np.exp(-self._beta*(new_pot-curr_pot))
            rand_number = random.random()

            if exp_term > rand_number:
                return True
            else:
                return False

    def _output_top_lines(self):

        string = (
            '====================================================\n'
            '                Collapser optimisation              \n'
            '                ----------------------              \n'
            '                                                    \n'
            f' step size = {self._step_size} \n'
            f' rotation step size = {self._rotation_step_size} \n'
            f' target bond length = {self._target_bond_length} \n'
            f' num. steps = {self._num_steps} \n'
            f' bond epsilon = {self._bond_epsilon} \n'
            f' nonbond epsilon = {self._nonbond_epsilon} \n'
            f' nonbond sigma = {self._nonbond_sigma} \n'
            f' nonbond mu = {self._nonbond_mu} \n'
            f' beta = {self._beta} \n'
            f' neighbour lists used? = {self._use_neighbour_list} \n'
            '====================================================\n\n'
        )

        return string

    def _plot_progess(self, system_properties, output_dir):

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            system_properties['steps'],
            system_properties['max_bond_distance'],
            c='k', lw=2
        )
        # Set number of ticks for x-axis
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('step', fontsize=16)
        ax.set_ylabel('max long bond length [angstrom]', fontsize=16)
        ax.axhline(y=self._target_bond_length, c='r', linestyle='--')
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f'maxd_vs_step.pdf'),
            dpi=360,
            bbox_inches='tight'
        )
        plt.close()
        # Plot energy vs timestep.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            system_properties['steps'],
            system_properties['total_potential'],
            c='k', lw=2, label='system potential'
        )
        ax.plot(
            system_properties['steps'],
            system_properties['nbond_potential'],
            c='r', lw=2, label='nonbonded potential'
        )
        # Set number of ticks for x-axis
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('step', fontsize=16)
        ax.set_ylabel('potential', fontsize=16)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f'pot_vs_step.pdf'),
            dpi=360,
            bbox_inches='tight'
        )
        plt.close()

    def _get_subunits(self, mol, bonds):
        """
        Get connected graphs based on mol separated by bonds.

        """

        # Produce a graph from the molecule that does not include edges
        # where the bonds to be optimized are.
        mol_graph = nx.Graph()
        for atom in mol.get_atoms():
            mol_graph.add_node(atom.get_id())

        # Add edges.
        for bond in mol.get_bonds():
            pair_ids = (bond.get_atom1_id(), bond.get_atom2_id())
            if pair_ids not in bonds:
                mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        subunits = {
            i: sg
            for i, sg in enumerate(nx.connected_components(mol_graph))
        }

        return subunits

    def _run_optimization(self, mol, bonds, output_dir, f):

        f.write(self._output_top_lines())
        f.write(f'There are {len(bonds)} long bonds.\n')
        # Find rigid subunits based on bonds to optimize.
        subunits = self._get_subunits(mol, bonds)

        # Define neighbour list based on connected subunits.
        if self._use_neighbour_list:
            neighour_lists = {}
            for su in subunits:
                connected_subunits = []
                for bond_ids in bonds:
                    if bond_ids[0] in subunits[su]:
                        connected_subunits.append(
                            [
                                i
                                for i in subunits
                                if bond_ids[1] in subunits[i]
                            ][0]
                        )
                    elif bond_ids[1] in subunits[su]:
                        connected_subunits.append(
                            [
                                i
                                for i in subunits
                                if bond_ids[0] in subunits[i]
                            ][0]
                        )
                neighour_lists[su] = connected_subunits

        else:
            neighour_lists = None
        f.write(
            f'There are {len(subunits)} sub units of sizes: '
            f'{[len(subunits[i]) for i in subunits]}\n'
        )

        if self._use_neighbour_list:
            f.write(f'Neighbour lists: {neighour_lists}\n\n\n')

        system_potential = self._compute_potential(
            mol=mol, bonds=bonds, neighour_lists=neighour_lists,
        )

        # Write structures at each step to file.
        mol.write_xyz_file(os.path.join(output_dir, f'coll_0.xyz'))

        # Update properties at each step.
        system_properties = {
            'steps': [0],
            'passed': [],
            'total_potential': [system_potential],
            'nbond_potential': [
                self._compute_non_bonded_potential(
                    mol=mol, neighour_lists=neighour_lists
                )
            ],
            'max_bond_distance': [max([
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bonds
            ])],
        }
        f.write(
            'step system_potential nonbond_potential max_dist '
            'opt_bbs updated?\n'
        )
        f.write(
            f"{system_properties['steps'][-1]} "
            f"{system_properties['total_potential'][-1]} "
            f"{system_properties['nbond_potential'][-1]} "
            f"{system_properties['max_bond_distance'][-1]} "
            '-- --\n'
        )

        for step in range(1, self._num_steps):
            step_begin_time = time.time()
            position_matrix = mol.get_position_matrix()

            # Randomly select a bond to optimize from bonds.
            bond_ids = random.choice(bonds)
            bond_vector = self._get_bond_vector(
                position_matrix=position_matrix,
                bond_ids=bond_ids
            )

            # Get subunits connected by selected bonds.
            subunit_1 = [
                i for i in subunits if bond_ids[0] in subunits[i]
            ][0]
            subunit_2 = [
                i for i in subunits if bond_ids[1] in subunits[i]
            ][0]

            # Choose subunit to move out of the two connected by the
            # bond randomly.
            moving_su = random.choice([subunit_1, subunit_2])
            moving_su_atom_ids = tuple(i for i in subunits[moving_su])

            # Random number from -1 to 1 for multiplying translation.
            rand = (random.random() - 0.5) * 2
            # Define translation along long bond vector where
            # direction is from force, magnitude is randomly
            # scaled.
            bond_translation = -bond_vector * self._step_size * rand

            # Define subunit COM vector to molecule COM.
            cent = mol.get_centroid()
            su_cent_vector = (
                mol.get_centroid(atom_ids=moving_su_atom_ids)-cent
            )
            com_translation = su_cent_vector * self._step_size * rand

            # Randomly choose between translation along long bond
            # vector or along BB-COM vector.
            translation_vector = random.choice([
                bond_translation,
                com_translation,
            ])

            # Translate building block.
            # Update atom position of building block.
            mol = self._translate_atoms_along_vector(
                mol=mol,
                atom_ids=moving_su_atom_ids,
                vector=translation_vector,
            )

            # Define a random rotation of a random subunit out of the
            # two options.
            # Random number from -1 to 1 for multiplying rotation.
            rand = (random.random() - 0.5) * 2
            rotation_angle = self._rotation_step_size * rand
            rotation_axis = normalize_vector(np.array(su_cent_vector))

            # Rotate the subunit.
            mol = self._rotate_atoms_by_angle(
                mol=mol,
                atom_ids=moving_su_atom_ids,
                angle=rotation_angle,
                axis=rotation_axis,
                origin=mol.get_centroid(atom_ids=moving_su_atom_ids),
            )
            new_system_potential = self._compute_potential(
                mol=mol, bonds=bonds, neighour_lists=neighour_lists,
            )
            print(step, subunit_1, subunit_2, new_system_potential)

            if self._test_move(system_potential, new_system_potential):
                updated = 'T'
                system_potential = new_system_potential
                system_properties['passed'].append(step)
            else:
                updated = 'F'
                # Reverse move.
                mol = self._translate_atoms_along_vector(
                    mol=mol,
                    atom_ids=moving_su_atom_ids,
                    vector=-translation_vector,
                )

            # Write structures at each step to file.
            mol.write_xyz_file(
                os.path.join(output_dir, f'coll_{step}.xyz')
            )

            # Update properties at each step.
            system_properties['steps'].append(step)
            system_properties['total_potential'].append(
                system_potential
            )
            system_properties['nbond_potential'].append(
                self._compute_non_bonded_potential(
                    mol=mol, neighour_lists=neighour_lists
                )
            )
            system_properties['max_bond_distance'].append(max([
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bonds
            ]))
            f.write(
                f"{system_properties['steps'][-1]} "
                f"{system_properties['total_potential'][-1]} "
                f"{system_properties['nbond_potential'][-1]} "
                f"{system_properties['max_bond_distance'][-1]} "
                f'{bond_ids} {updated}\n'
            )
            step += 1

        f.write('\n============================================\n')
        f.write(
            'Optimisation done:\n'
            f"{len(system_properties['passed'])} steps passed: "
            f"{len(system_properties['passed'])/self._num_steps}"
        )

        print(
            f'Step time: {time.time() - step_begin_time}'
        )

        self._plot_progess(system_properties, output_dir)

        return mol

    def optimize(self, mol, bonds):
        """
        Optimize `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_ids :
            Bonds to optimize.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        """

        begin_time = time.time()

        # Handle output dir.
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        with open(os.path.join(output_dir, f'coll.out'), 'w') as f:
            mol = self._run_optimization(mol, bonds, output_dir, f)

        print(f'Total optimisation time: {time.time() - begin_time}')

        return mol
