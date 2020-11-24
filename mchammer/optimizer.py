"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimise intermolecular distances.

"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
import networkx as nx
import uuid
import os
import shutil

from .utilities import get_atom_distance


class Optimizer:
    """
    Optimize target bonds using MC algorithm.

    A Metropolis MC algorithm is applied to perform rigid
    translations of the subunits separatred by the target bonds.

    """

    def __init__(
        self,
        output_dir,
        step_size,
        target_bond_length,
        num_steps,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=1000,
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

        random_seed : :class:`int` or :class:`NoneType`, optional
            Random seed to use for MC algorithm. Should only be set to
            ``None`` if system-based random seed is desired. Defaults
            to 1000.

        """

        self._output_dir = output_dir
        self._step_size = step_size
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

    def _get_bond_vector(self, position_matrix, bond_pair):
        """
        Get vector from atom1 to atom2 in bond.

        """

        atom1_pos = position_matrix[bond_pair[0]]
        atom2_pos = position_matrix[bond_pair[1]]
        return atom2_pos - atom1_pos

    def _bond_potential(self, distance):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        potential = (distance - self._target_bond_length) ** 2
        potential = self._bond_epsilon * potential

        return potential

    def _nonbond_potential(self, distance):
        """
        Define an arbitrary repulsive nonbonded potential.

        This potential has no relation to an empircal forcefield.

        """

        return (
            self._nonbond_epsilon * (
                (self._nonbond_sigma/distance) ** self._nonbond_mu
            )
        )

    def _compute_nonbonded_potential(self, position_matrix):
        # Get all pairwise distances between atoms in each subunut.
        pair_dists = pdist(position_matrix)
        nonbonded_potential = np.sum(
            self._nonbond_potential(pair_dists)
        )

        return nonbonded_potential

    def _compute_potential(self, mol, bond_pair_ids):
        position_matrix = mol.get_position_matrix()
        nonbonded_potential = self._compute_nonbonded_potential(
            position_matrix=position_matrix,
        )
        system_potential = nonbonded_potential
        for bond in bond_pair_ids:
            system_potential += self._bond_potential(
                distance=get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
            )

        return system_potential, nonbonded_potential

    def _translate_atoms_along_vector(self, mol, atom_ids, vector):

        new_position_matrix = mol.get_position_matrix()
        for atom in mol.get_atoms():
            if atom.get_id() not in atom_ids:
                continue
            pos = mol.get_position_matrix()[atom.get_id()]
            new_position_matrix[atom.get_id()] = pos - vector

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
            '                MCHammer optimisation               \n'
            '                ---------------------               \n'
            '                    Andrew Tarzia                   \n'
            '                ---------------------               \n'
            '                                                    \n'
            'Settings:                                           \n'
            f' step size = {self._step_size} \n'
            f' target bond length = {self._target_bond_length} \n'
            f' num. steps = {self._num_steps} \n'
            f' bond epsilon = {self._bond_epsilon} \n'
            f' nonbond epsilon = {self._nonbond_epsilon} \n'
            f' nonbond sigma = {self._nonbond_sigma} \n'
            f' nonbond mu = {self._nonbond_mu} \n'
            f' beta = {self._beta} \n'
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
        ax.set_xlim(0, None)
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
        ax.set_xlim(0, None)
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

    def _get_subunits(self, mol, bond_pair_ids):
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
            if pair_ids not in bond_pair_ids:
                mol_graph.add_edge(*pair_ids)

        # Get atom ids in disconnected subgraphs.
        subunits = {
            i: sg
            for i, sg in enumerate(nx.connected_components(mol_graph))
        }

        return subunits

    def _run_optimization(self, mol, bond_pair_ids, output_dir, f):

        begin_time = time.time()

        f.write(self._output_top_lines())
        f.write(f'There are {len(bond_pair_ids)} bonds to optimize.\n')
        # Find rigid subunits based on bonds to optimize.
        subunits = self._get_subunits(mol, bond_pair_ids)
        f.write(
            f'There are {len(subunits)} sub units with N atoms:\n'
            f'{[len(subunits[i]) for i in subunits]}\n'
        )
        f.write(
            '====================================================\n'
            '                 Running optimisation!              \n'
            '====================================================\n\n'
        )

        system_potential, nonbonded_potential = (
            self._compute_potential(
                mol=mol,
                bond_pair_ids=bond_pair_ids
            )
        )

        # Write structures at each step to file.
        mol.write_xyz_file(os.path.join(output_dir, f'coll_0.xyz'))

        # Update properties at each step.
        system_properties = {
            'steps': [0],
            'passed': [],
            'total_potential': [system_potential],
            'nbond_potential': [nonbonded_potential],
            'max_bond_distance': [max([
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bond_pair_ids
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
            position_matrix = mol.get_position_matrix()

            # Randomly select a bond to optimize from bonds.
            bond_ids = random.choice(bond_pair_ids)
            bond_vector = self._get_bond_vector(
                position_matrix=position_matrix,
                bond_pair=bond_ids
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

            new_system_potential, new_nonbonded_potential = (
                self._compute_potential(
                    mol=mol,
                    bond_pair_ids=bond_pair_ids
                )
            )

            if self._test_move(system_potential, new_system_potential):
                updated = 'T'
                system_potential = new_system_potential
                nonbonded_potential = new_nonbonded_potential
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
                nonbonded_potential
            )
            system_properties['max_bond_distance'].append(max([
                get_atom_distance(
                    position_matrix=mol.get_position_matrix(),
                    atom1_id=bond[0],
                    atom2_id=bond[1],
                )
                for bond in bond_pair_ids
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
            f"{100*(len(system_properties['passed'])/self._num_steps)}"
            "%\n"
            f'Total optimisation time: '
            f'{round(time.time() - begin_time, 4)}s\n'
        )
        f.write('============================================\n')

        self._plot_progess(system_properties, output_dir)

        return mol

    def optimize(self, mol, bond_pair_ids):
        """
        Optimize `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_pair_ids : :class:`iterable` of :class:`ints`
            Pair of atom ids with bond between them to optimize.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        """

        # Handle output dir.
        if self._output_dir is None:
            output_dir = str(uuid.uuid4().int)
        else:
            output_dir = self._output_dir
        output_dir = os.path.abspath(output_dir)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        with open(os.path.join(output_dir, f'mch.out'), 'w') as f:
            mol = self._run_optimization(
                mol=mol,
                bond_pair_ids=bond_pair_ids,
                output_dir=output_dir,
                f=f
            )

        return mol
