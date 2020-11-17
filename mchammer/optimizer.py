"""
MCHammer Optimizer
==================

#. :class:`.Optimizer`

Optimizer for minimise intermolecular distances.

"""

import logging
from itertools import combinations
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import random
import networkx as nx
import uuid
import os
import shutil

from .utilities import get_atom_distance


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
        target_bond_length,
        num_steps,
        bond_epsilon=50,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
        beta=2,
        random_seed=None,
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

        random_seed : :class:`int`, optional
            Random seed to use for MC algorithm. Should only be set
            if exactly reproducible results are required, otherwise
            a system-based random seed should be used for proper
            sampling.

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

    def _get_inter_bb_distance(self, mol):
        """
        Yield The distances between building blocks in mol.

        Ignores H atoms.

        """

        position_matrix = mol.get_position_matrix()

        for atom1, atom2 in combinations(mol.get_atom_infos(), 2):
            chk1 = (
                atom1.get_atom().get_id() != atom2.get_atom().get_id()
            )
            chk2 = (
                atom1.get_atom().get_atomic_number() != 1
                and atom2.get_atom().get_atomic_number() != 1
            )
            chk3 = (
                atom1.get_building_block_id() !=
                atom2.get_building_block_id()
            )
            if chk1 and chk2 and chk3:
                dist = get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=atom1.get_atom().get_id(),
                    atom2_id=atom2.get_atom().get_id()
                )
                yield dist

    def _has_short_contacts(self, mol):
        """
        Calculate if there are short contants in mol.

        """

        return any(
            dist < self._distance_cut
            for dist in self._get_inter_bb_distance(mol)
        )

    def _get_new_position_matrix(self, mol, step, vectors, scales):
        """
        Get the position matrix of the mol after translation.

        """

        new_position_matrix = mol.get_position_matrix()
        for atom in mol.get_atom_infos():
            bb_id = atom.get_building_block_id()
            _id = atom.get_atom().get_id()
            pos = mol.get_position_matrix()[_id]
            new_position_matrix[_id] = (
                pos - step*vectors[bb_id]*scales[bb_id]
            )

        return new_position_matrix

    def _get_bb_vectors(self, mol, bb_atom_ids):
        """
        Get the building block to COM vectors.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bb_atom_ids : :class:`dict` mapping :class:`int`: :class:`list`
            Dictionary mapping building block ids (keys) to a list of
            atom ids (values) in each distinct building block in the
            molecule.

        Returns
        -------
        bb_cent_vectors :
            :class:`dict` mapping :class:`int`: :class:`numpy.ndarray`
            Dictionary mapping building block ids (keys) to centroid
            vectors (values) of each distinct building block in the
            molecule.

        bb_cent_scales :
            :class:`dict` mapping :class:`int`: :class:`float`
            Dictionary mapping building block ids (keys) to relative
            magnitude of centroid vectors (values) of each distinct
            building block in the molecule.

        """

        cent = mol.get_centroid()

        # Get bb COM vector to molecule COM.
        bb_cent_vectors = {
            i: mol.get_centroid(atom_ids=bb_atom_ids[i])-cent
            for i in bb_atom_ids
        }

        # Scale the step size based on the different distances of
        # bbs from the COM. Impacts anisotropic topologies.
        if self._scale_steps:
            norms = {
                i: np.linalg.norm(bb_cent_vectors[i])
                for i in bb_cent_vectors
            }
            max_distance = max(list(norms.values()))
            bb_cent_scales = {
                i: norms[i]/max_distance
                for i in norms
            }
        else:
            bb_cent_scales = {
                i: 1
                for i in bb_cent_vectors
            }

        return bb_cent_vectors, bb_cent_scales

    def _get_bb_atom_ids(self, mol):
        bb_atom_ids = defaultdict(list)
        for i in mol.get_atom_infos():
            bb_atom_ids[i.get_building_block_id()].append(
                i.get_atom().get_id()
            )

        return bb_atom_ids

    def _get_bond_vector(self, position_matrix, bond_ids):
        atom1_pos = position_matrix[bond_ids[0]]
        atom2_pos = position_matrix[bond_ids[1]]
        return atom2_pos - atom1_pos

    def _get_bb_centroids(self, mol, bb_atom_ids):
        """
        Returns dict of building block centroids.

        """

        bb_centroids = {
            i: mol.get_centroid(atom_ids=bb_atom_ids[i])
            for i in bb_atom_ids
        }

        return bb_centroids

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

        return sum(non_bonded_potential)

    def _compute_potential(self, mol, bonds):

        system_potential = self._compute_non_bonded_potential(mol)
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

    def _rotate_atoms_onto_vector(
        self,
        mol,
        atom_ids,
        start_vector,
        target_vector,
        axis
    ):
        raise NotImplementedError()

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
        print(bonds)
        subunits = self._get_subunits(mol, bonds)
        subunit_centroids = {
            i: mol.get_centroid(atom_ids=subunits[i])
            for i in subunits
        }
        print(subunit_centroids)

        f.write(
            f'There are {len(subunits)} sub units of sizes: '
            f'{[len(subunits[i]) for i in subunits]}\n'
        )

        # Define neighbour lists here and implement this in the below
        # method.

        system_potential = self._compute_potential(
            mol=mol, bonds=bonds,
        )

        # Define bb centroid - long bond atom vectors.
        # These are to be maintained during optimisation.
        # centroid_to_lb_vectors = self._get_cent_to_lb_vector(
        #     mol, bb_centroids, long_bond_infos
        # )
        f.write('################################################\n')
        f.write('WARNING: centroid_to_lb_vectors not maintained.\n')
        f.write('################################################\n')

        # Write structures at each step to file.
        mol.write_xyz_file(os.path.join(output_dir, f'coll_0.xyz'))

        # Update properties at each step.
        system_properties = {
            'steps': [0],
            'passed': [],
            'total_potential': [system_potential],
            'nbond_potential': [
                self._compute_non_bonded_potential(mol=mol)
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
            moving_bb = random.choice([subunit_1, subunit_2])
            moving_bb_atom_ids = tuple(i for i in subunits[moving_bb])

            # Random number from -1 to 1 for multiplying translation.
            rand = (random.random() - 0.5) * 2
            # Define translation along long bond vector where
            # direction is from force, magnitude is randomly
            # scaled.
            bond_translation = -bond_vector * self._step_size * rand

            # Define subunit COM vector to molecule COM.
            cent = mol.get_centroid()
            su_cent_vector = (
                mol.get_centroid(atom_ids=moving_bb_atom_ids)-cent
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
                atom_ids=moving_bb_atom_ids,
                vector=translation_vector,
            )

            ###################################################
            # Here I want to add rotations
            # To maintain cent_vectors relative orientations.
            # cent_vector_1 = centroid_to_lb_vectors[
            #     (bb_id_1, lb_ids[0])
            # ]
            # cent_vector_2 = centroid_to_lb_vectors[
            #     (bb_id_2, lb_ids[1])
            # ]
            ###################################################

            new_system_potential = self._compute_potential(
                mol=mol, bonds=bonds,
            )
            print(subunit_1, subunit_2, new_system_potential)

            if self._test_move(system_potential, new_system_potential):
                updated = 'T'
                system_potential = new_system_potential
                system_properties['passed'].append(step)
            else:
                updated = 'F'
                # Reverse move.
                mol = self._translate_atoms_along_vector(
                    mol=mol,
                    atom_ids=moving_bb_atom_ids,
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
                self._compute_non_bonded_potential(mol=mol)
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

        return mol
