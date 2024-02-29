"""Define classes holding result information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import abc

    import numpy as np


@dataclass
class StepResult:
    """Results of a step.

    Parameters:
        step:
            Step number.

        position_matrix:
            A position matrix after performing this step. The shape of
            the matrix is ``(n, 3)``.

        max_bond_distance:
            Max length of bonds to be optimized in Angstrom.

        log:
            String log of this step.

    """

    step: int
    position_matrix: np.ndarray
    max_bond_distance: float
    log: str

    def get_properties(self) -> dict:
        """Get step properties."""
        return {"max_bond_distance": self.max_bond_distance}


@dataclass
class MCStepResult(StepResult):
    """Results of a step.

    Parameters:
        step:
            Step number.

        position_matrix:
            A position matrix after performing this step. The shape of
            the matrix is ``(n, 3)``.

        passed:
            Flag for whether the MC move passed, or was reverted to
            the previous step.

        system_potential:
            System potential of the structure after this step.

        nonbonded_potential:
            Nonbonded potential of the structure after this step.

        max_bond_distance:
            Max length of bonds to be optimized in Angstrom.

        log:
            String log of this step.

    """

    step: int
    log: str
    position_matrix: np.ndarray
    system_potential: float
    nonbonded_potential: float
    max_bond_distance: float
    passed: bool | None = None

    def get_properties(self) -> dict:
        """Get step properties."""
        return {
            "max_bond_distance": self.max_bond_distance,
            "system_potential": self.system_potential,
            "nonbonded_potential": self.nonbonded_potential,
            "passed": self.passed,
        }


class Result:
    """Result of optimization."""

    def __init__(self, start_time: float) -> None:
        """Initialize a :class:`Result` instance.

        Parameters:
            start_time:
                Start of run timing.

        """
        self._start_time = start_time
        self._log = ""
        self._step_results: dict[int, StepResult] = {}
        self._step_count = 0

    def add_step_result(self, step_result: StepResult) -> None:
        """Add StepResult."""
        self._step_results[step_result.step] = step_result
        self.update_log(step_result.log)
        self._step_count = step_result.step

    def update_log(self, string: str) -> None:
        """Update result log."""
        self._log += string

    def get_log(self) -> str:
        """Get result log."""
        return self._log

    def get_number_passed(self) -> int:
        """Get Number of steps that passed."""
        return len(
            [
                1
                for step in self._step_results
                if self._step_results[step].passed  # type: ignore[attr-defined]
            ]
        )

    def get_final_position_matrix(self) -> np.ndarray:
        """Get final molecule in result."""
        return self._step_results[self._step_count].position_matrix

    def get_timing(self, time: float) -> float:
        """Get run timing."""
        return time - self._start_time

    def get_step_count(self) -> int:
        """Get step count."""
        return self._step_count

    def get_steps_properties(self) -> abc.Iterable[tuple[int, dict]]:
        """Yield properties of all steps."""
        for step in self._step_results:
            yield step, self._step_results[step].get_properties()

    def get_trajectory(self) -> abc.Iterable[tuple[int, np.ndarray]]:
        """Yield .Molecule of all steps."""
        for step in self._step_results:
            yield step, self._step_results[step].position_matrix
