"""
Results
=======

#. :class:`.Result`
#. :class:`.StepResult`

Classes for maintaining optimization results.

"""


class StepResult:
    """
    Results of a step.

    """

    def __init__(self, step):
        """
        Initialize a :class:`StepResult` instance.

        Parameters
        ----------
        step : :class:`int`
            Step number.

        """

        self._step = step
        self._log = ''

    def get_step(self):
        return self._step

    def get_log(self):
        return self._log

    def set_position_matrix(self, position_matrix):
        self._position_matrix = position_matrix

    def set_passed(self, passed):
        self._passed = passed

    def set_system_potential(self, system_potential):
        self._system_potential = system_potential

    def set_nonbonded_potential(self, nonbonded_potential):
        self._nonbonded_potential = nonbonded_potential

    def set_max_bond_distance(self, max_bond_distance):
        self._max_bond_distance = max_bond_distance

    def get_position_matrix(self):
        return self._position_matrix

    def get_passed(self):
        return self._passed

    def get_system_potential(self):
        return self._system_potential

    def get_nonbonded_potential(self):
        return self._nonbonded_potential

    def get_max_bond_distance(self):
        return self._max_bond_distance

    def get_properties(self):
        return {
            'max_bond_distance': self._max_bond_distance,
            'system_potential': self._system_potential,
            'nonbonded_potential': self._nonbonded_potential,
            'passed': self._passed,
        }

    def update_log(self, string):
        """
        Update result log.

        """
        self._log += string


class Result:
    """
    Result of optimization.

    """

    def __init__(self, start_time):
        """
        Initialize a :class:`Result` instance.

        Parameters
        ----------
        start_time : :class:`time.time()`
            Start of run timing.

        """

        self._start_time = start_time
        self._log = ''
        self._step_results = {}
        self._step_count = None

    def add_step_result(self, step_result):
        """
        Add StepResult.

        """

        self._step_results[step_result.get_step()] = step_result
        self.update_log(step_result.get_log())
        self._step_count = step_result.get_step()

    def update_log(self, string):
        """
        Update result log.

        """
        self._log += string

    def get_log(self):
        """
        Get result log.

        """
        return self._log

    def get_number_passed(self):
        """
        Get Number of steps that passed.

        """
        return len([
            1 for step in self._step_results
            if self._step_results[step].get_passed()
        ])

    def get_final_position_matrix(self):
        """
        Get final molecule in result.

        """

        return (
            self._step_results[self._step_count].get_position_matrix()
        )

    def get_timing(self, time):
        """
        Get run timing.

        """
        return time - self._start_time

    def get_step_count(self):
        """
        Get step count.

        """
        return self._step_count

    def get_steps_properties(self):
        """
        Yield properties of all steps.

        """

        for step in self._step_results:
            yield step, self._step_results[step].get_properties()

    def get_trajectory(self):
        """
        Yield .Molecule of all steps.

        """

        for step in self._step_results:
            yield step, self._step_results[step].get_position_matrix()
