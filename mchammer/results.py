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

    def __init__(self, step, start_time):
        """
        Initialize a :class:`StepResult` instance.

        Parameters
        ----------

        """

        self._step = step
        self._start_time = start_time
        self._log = ''

    def get_step(self):
        return self._step

    def get_log(self):
        return self._log

    def get_properties(self):
        return self._properties

    def get_position_matrix(self):
        return self._position_matrix

    def get_system_potential(self):
        return self._properties['total_potential']

    def get_nonbonded_potential(self):
        return self._properties['nbond_potential']

    def get_max_bond_distance(self):
        return self._properties['max_bond_distance']

    def add_position_matrix(self, position_matrix):
        self._position_matrix = position_matrix

    def add_properties(self, step_properties):
        self._properties = step_properties

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
            if self._step_results[step].get_properties()['passed']
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
