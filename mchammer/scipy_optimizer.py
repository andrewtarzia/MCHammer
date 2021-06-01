"""
Scipy Optimizer
===============

#. :class:`.ScipyOptimizer`

Optimizer for minimising intermolecular distances.

"""

import numpy as np
import time
import scipy
from scipy.spatial.distance import pdist

from .results import Result, MCStepResult
from .utilities import get_atom_distance, get_angle
from .optimizer import Optimizer


class OptimizerConvergenceException(Exception):
    ...


class ScipyOptimizer(Optimizer):
    """
    Optimize target bonds using scipy optimisation algorithm.

    """

    def __init__(
        self,
        target_bond_length,
        num_steps,
        bond_epsilon=50,
        angle_epsilon=500,
        nonbond_epsilon=20,
        nonbond_sigma=1.2,
        nonbond_mu=3,
    ):
        """
        Initialize a :class:`ScipyOptimizer` instance.

        Parameters
        ----------
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

        """

        self._target_bond_length = target_bond_length
        self._num_steps = num_steps
        self._bond_epsilon = bond_epsilon
        self._angle_epsilon = angle_epsilon
        self._nonbond_epsilon = nonbond_epsilon
        self._nonbond_sigma = nonbond_sigma
        self._nonbond_mu = nonbond_mu

    def _bond_potential(self, distance, target_distance, bond_epsilon):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        potential = (distance - target_distance) ** 2
        potential = bond_epsilon * potential

        return potential

    def _angle_potential(self, angle, target_angle, angle_epsilon):
        """
        Define an arbitrary parabolic bond potential.

        This potential has no relation to an empircal forcefield.

        """

        potential = (angle - target_angle) ** 2
        potential = angle_epsilon * potential

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

    def _compute_nonbonded_potential(self, pair_dists):
        nonbonded_potential = np.sum(
            self._nonbond_potential(pair_dists)
        )

        return nonbonded_potential

    def _compute_potential_from_posmat(
        self,
        x,
        bond_targets,
        angle_targets,
    ):
        position_matrix = x.reshape((-1, 3))
        pair_dists = pdist(position_matrix)
        nonbonded_potential = self._compute_nonbonded_potential(
            pair_dists=pair_dists,
        )
        system_potential = nonbonded_potential
        m = position_matrix.shape[0]
        for bond in bond_targets:
            i = bond[0]
            j = bond[1]
            id_ = int(m * i + j - ((i + 2) * (i + 1)) // 2.)
            system_potential += self._bond_potential(
                distance=pair_dists[id_],
                target_distance=bond[2],
                bond_epsilon=bond[3],
            )

        for angle in angle_targets:
            system_potential += self._angle_potential(
                angle=get_angle(
                    position_matrix=position_matrix,
                    atom1_id=angle[0],
                    atom2_id=angle[1],
                    atom3_id=angle[2],
                ),
                target_angle=angle[3],
                angle_epsilon=angle[4],
            )

        return system_potential

    def _get_bond_targets(self, mol, bond_pair_ids):
        bond_targets = []
        position_matrix = mol.get_position_matrix()
        for bond in mol.get_bonds():
            a1_id, a2_id = sorted(
                [bond.get_atom1_id(), bond.get_atom2_id()]
            )
            if (a1_id, a2_id) in bond_pair_ids:
                target = self._target_bond_length
                epsilon = 1*self._bond_epsilon
            else:
                target = get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=a1_id,
                    atom2_id=a2_id,
                )
                epsilon = 3*self._bond_epsilon
            bond_targets.append((a1_id, a2_id, target, epsilon))
        return bond_targets

    def _get_angle_targets(self, mol, bond_pair_ids):
        angle_targets = []
        position_matrix = mol.get_position_matrix()
        for angle in mol.get_angles():
            a1_id, a2_id, a3_id = (
                angle.get_atom1_id(),
                angle.get_atom2_id(),
                angle.get_atom3_id(),
            )
            b1 = sorted((a1_id, a2_id))
            b2 = sorted((a2_id, a3_id))
            if b1 not in bond_pair_ids and b2 not in bond_pair_ids:
                target = get_angle(
                    position_matrix=position_matrix,
                    atom1_id=a1_id,
                    atom2_id=a2_id,
                    atom3_id=a3_id,
                )
                epsilon = self._angle_epsilon
            angle_targets.append(
                (a1_id, a2_id, a3_id, target, epsilon)
            )
        return angle_targets

    def _get_x(self, mol):
        return mol.get_position_matrix().reshape(-1)

    def get_trajectory(self, mol, bond_pair_ids):
        """
        Get trajectory of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.Result`
            The result of the optimization including all steps.

        """

        result = Result(start_time=time.time())

        result.update_log(
            '====================================================\n'
            "                    Scipy algorithm.\n"
            '====================================================\n'
        )
        result.update_log(
            f'There are {len(bond_pair_ids)} bonds to optimize.\n'
        )
        result.update_log(
            '====================================================\n'
            '                 Running optimisation!              \n'
            '====================================================\n\n'
        )

        bond_targets = self._get_bond_targets(
            mol=mol,
            bond_pair_ids=bond_pair_ids,
        )
        angle_targets = self._get_angle_targets(
            mol=mol,
            bond_pair_ids=bond_pair_ids,
        )

        x0 = self._get_x(mol)
        print('init', self._compute_potential_from_posmat(
            mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))

        # Define the optimizer based on minimizing the potential.
        def func(x, *args):
            """
            Function to optimize, where `x` is position matrix.

            """

            return self._compute_potential_from_posmat(
                x, args[0], args[1]
            )

        opt_result = scipy.optimize.minimize(
            fun=func,
            x0=x0,
            method='BFGS',
            options={'maxiter': self._num_steps},
            args=(bond_targets, angle_targets),
        )
        print(opt_result['message'])
        # Update position matrix.
        new_position_matrix = opt_result['x'].reshape((-1, 3))
        mol = mol.with_position_matrix(new_position_matrix)
        print('final', self._compute_potential_from_posmat(
            mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))

        result.update_log(
            string=(
                '\n============================================\n'
                'Optimisation done:\n'
                f'Total optimisation time: '
                f'{round(result.get_timing(time.time()), 4)}s\n'
                '============================================\n'
            ),
        )

        return mol, result

    def get_result(self, mol, bond_pair_ids):
        """
        Get final result of optimization run on `mol`.

        Parameters
        ----------
        mol : :class:`.Molecule`
            The molecule to be optimized.

        bond_pair_ids :
            :class:`iterable` of :class:`tuple` of :class:`ints`
            Iterable of pairs of atom ids with bond between them to
            optimize.

        Returns
        -------
        mol : :class:`.Molecule`
            The optimized molecule.

        result : :class:`.MCStepResult`
            The result of the final optimization step.

        """

        result = Result(start_time=time.time())

        bond_targets = self._get_bond_targets(
            mol=mol,
            bond_pair_ids=bond_pair_ids,
        )
        angle_targets = self._get_angle_targets(
            mol=mol,
            bond_pair_ids=bond_pair_ids,
        )

        x0 = self._get_x(mol)
        print('init', self._compute_potential_from_posmat(
            mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))

        # Define the optimizer based on minimizing the potential.
        def func(x, *args):
            """
            Function to optimize, where `x` is position matrix.

            """

            return self._compute_potential_from_posmat(
                x, args[0], args[1]
            )

        opt_result = scipy.optimize.minimize(
            fun=func,
            x0=x0,
            method='BFGS',
            options={'maxiter': self._num_steps},
            args=(bond_targets, angle_targets),
        )
        print(opt_result['message'])
        # Update position matrix.
        new_position_matrix = opt_result['x'].reshape((-1, 3))
        mol = mol.with_position_matrix(new_position_matrix)
        print('final', self._compute_potential_from_posmat(
            mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))

        return mol, result
