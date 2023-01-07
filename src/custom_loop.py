from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from typing import List, Union, Callable
from emukit.core.loop.model_updaters import ModelUpdater
from emukit.core.loop.user_function import UserFunction, UserFunctionWrapper
from emukit.core.loop.stopping_conditions import StoppingCondition, FixedIterationsStoppingCondition, ConvergenceStoppingCondition
from emukit.core.loop.candidate_point_calculators import CandidatePointCalculator
from emukit.core.loop.loop_state import LoopState
from emukit.core.event_handler import EventHandler
import logging
_log = logging.getLogger(__name__)
import numpy as np

class CustomLoop(OuterLoop):
    def generate_neighbors(self):
        space_step = np.array(self.space_step)[:,None]
        transformation_matrix = np.array([[-2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], \
                                          [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]])
        neighbors = space_step * transformation_matrix
        return neighbors
    
    def find_neighbors(self, x_new, user_function):
        final_points = arr = np.empty((0,3))
        neighbors = self.generate_neighbors()
        for x in x_new:
            true_value = user_function.evaluate(np.array([x]))
            if x[2] == 0.0:
                final_points = np.append(final_points, np.array([x]), axis=0)
                continue
            neighbors_3d = np.concatenate((neighbors, np.zeros([1,neighbors.shape[1]])), axis=0)
            new_points = np.array([n+x for n in neighbors_3d.T]) # neighbors_3d+x
            new_points = new_points[[((point>=-1.0).all() and (point<=1.0).all()) for point in new_points]]
            results = user_function.evaluate(new_points)
            filtered_points = [(r.Y == true_value[0].Y).all() for r in results]
            new_points = new_points[filtered_points]
            final_points = np.append(final_points, new_points, axis=0)
        return final_points
            

    def __init__(self, candidate_point_calculator: CandidatePointCalculator,
                 model_updaters: Union[ModelUpdater, List[ModelUpdater]], loop_state: LoopState = None, space_step=[0.1,0.1]) -> None:
        """
        :param candidate_point_calculator: Finds next points to evaluate by optimizing the acquisition function
        :param model_updaters: Updates the data in the model(s) and the model hyper-parameters when we observe new data
        :param loop_state: Object that keeps track of the history of the loop.
                           Default: None, resulting in empty initial state
        """
        self.candidate_point_calculator = candidate_point_calculator
        self.space_step = space_step

        if isinstance(model_updaters, list):
            self.model_updaters = model_updaters
        else:
            self.model_updaters = [model_updaters]
        self.loop_state = loop_state
        if self.loop_state is None:
            self.loop_state = LoopState([])

        self.loop_start_event = EventHandler()
        self.iteration_end_event = EventHandler()

    def run_loop(self, user_function: Union[UserFunction, Callable],
                stopping_condition: Union[StoppingCondition, int],
                context: dict = None) -> None:
        """
        :param user_function: The function that we are emulating
        :param stopping_condition: If integer - a number of iterations to run, or an object - a stopping
                        condition object that decides whether we should stop collecting more points.
                        Note that stopping conditions can be logically combined (&, |)
                        to represent complex stopping criteria.
        :param context: The context is used to force certain parameters of the inputs to the function of interest to
                        have a given value. It is a dictionary whose keys are the parameter names to fix and the values
                        are the values to fix the parameters to.
        """

        is_int = isinstance(stopping_condition, int)
        is_single_condition = isinstance(stopping_condition, StoppingCondition)

        if not (is_int or is_single_condition):
            raise ValueError("Expected stopping_condition to be an int or a StoppingCondition instance,"
                            "but received {}".format(type(stopping_condition)))

        if not isinstance(user_function, UserFunction):
            user_function = UserFunctionWrapper(user_function)

        if isinstance(stopping_condition, int):
            stopping_condition = FixedIterationsStoppingCondition(stopping_condition + self.loop_state.iteration)

        _log.info("Starting outer loop")

        self.loop_start_event(self, self.loop_state)

        while not stopping_condition.should_stop(self.loop_state):
            _log.info("Iteration {}".format(self.loop_state.iteration))

            self._update_models()
            # For us: SequentialPointCalculator with (ModelVariance / cost) acquisition and
            # MultiSourceAcquisitionOptimizer. What it does:
            # 1. Optimizer selects 1000 RANDOM points from space
            # 2. For all points, acquisition evaluates variance / cost in each fidelity and returns the maximum
            # 3. SequentialPointCalculator returns this maximum as next point
            new_x = self.candidate_point_calculator.compute_next_points(self.loop_state, context)
            # find points with same value and add them to acquisition points
            all_points = self.find_neighbors(new_x, user_function) 
            _log.debug("Next suggested point(s): {}".format(all_points))
            results = user_function.evaluate(all_points)
            _log.debug("User function returned: {}".format(results))
            self.loop_state.update(results)
            self.iteration_end_event(self, self.loop_state)

        self._update_models()
        _log.info("Finished outer loop")