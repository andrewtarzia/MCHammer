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
from .decomposed_molecule import DecomposedMolecule
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

    def _squared_difference(self, x, target_x):
        """
        Return squared difference between x and target.

        """

        return (x - target_x) ** 2

    def _compute_mean_square_error(
        self,
        x,
        bond_targets,
        angle_targets,
    ):
        position_matrix = x.reshape((-1, 3))
        pair_dists = pdist(position_matrix)
        differences = []
        m = position_matrix.shape[0]
        for bond in bond_targets:
            i = bond[0]
            j = bond[1]
            id_ = int(m * i + j - ((i + 2) * (i + 1)) // 2.)
            differences.append(self._squared_difference(
                x=pair_dists[id_],
                target_x=bond[2],
            ))

        for angle in angle_targets:
            differences.append(self._squared_difference(
                x=get_angle(
                    position_matrix=position_matrix,
                    atom1_id=angle[0],
                    atom2_id=angle[1],
                    atom3_id=angle[2],
                ),
                target_x=angle[3],
            ))

        return np.mean(differences)

    def _get_bond_targets(self, mol, bond_pair_ids):
        bond_targets = []
        position_matrix = mol.get_position_matrix()
        for bond in mol.get_bonds():
            a1_id, a2_id = sorted(
                [bond.get_atom1_id(), bond.get_atom2_id()]
            )
            if (a1_id, a2_id) in bond_pair_ids:
                target = self._target_bond_length
            else:
                target = get_atom_distance(
                    position_matrix=position_matrix,
                    atom1_id=a1_id,
                    atom2_id=a2_id,
                )
            bond_targets.append((a1_id, a2_id, target))
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
            angle_targets.append(
                (a1_id, a2_id, a3_id, target)
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

        raise NotImplementedError('get result mate.')

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

        decomposed_mol = DecomposedMolecule.decompose_molecule(
            molecule=mol,
            bond_pair_ids=bond_pair_ids,
        )
        decomposed_bond_pair_ids = decomposed_mol.get_bond_pair_ids()

        decomposed_mol.write_pdb_file('decomp.pdb')

        bond_targets = self._get_bond_targets(
            mol=decomposed_mol,
            bond_pair_ids=decomposed_bond_pair_ids,
        )
        angle_targets = self._get_angle_targets(
            mol=decomposed_mol,
            bond_pair_ids=decomposed_bond_pair_ids,
        )

        x0 = self._get_x(decomposed_mol)
        print('init', self._compute_mean_square_error(
            decomposed_mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))

        # Define the optimizer based on minimizing the potential.
        def func(x, *args):
            """
            Function to optimize, where `x` is position matrix.

            """

            return self._compute_mean_square_error(
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
        decomposed_mol = (
            decomposed_mol.with_position_matrix(new_position_matrix)
        )
        print('final', self._compute_mean_square_error(
            decomposed_mol.get_position_matrix(),
            bond_targets,
            angle_targets,
        ))
        decomposed_mol.write_pdb_file('finallol.pdb')

        mol = decomposed_mol.recompose_molecule(molecule=mol)
        import sys
        sys.exit()

        return mol, result
