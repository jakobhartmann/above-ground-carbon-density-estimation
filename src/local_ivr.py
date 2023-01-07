import abc

import numpy as np
from tqdm import tqdm
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.model_wrappers import GPyMultiOutputWrapper
from scipy.signal import convolve2d
import multiprocessing


class CandidatePointIdentifier(abc.ABC):

    @abc.abstractmethod
    def identify_points(self, variances: np.ndarray) -> np.ndarray:
        """
        :param variances: [num_points_height, num_points_width] a 2d map of variances
        :return: [num_candidate_points, 2] int array with the indices of the candidate points in the variance array.
        These are not the coordinates yet.
        """
        pass

class LatinHypercubeMaximaIdentifier(CandidatePointIdentifier):

    def __init__(self, num_strata: int):
        """
        :param num_strata: The number of strata to divide each dimension into for latin hypercube sampling.
        Note that this means the number of sampled points will be num_strata ^ 2
        """
        self.num_strata = num_strata

    def identify_points(self, variances: np.ndarray) -> np.ndarray:
        # if variances.shape[0] % self.num_strata != 0 or variances.shape[1] % self.num_strata != 0:
        #     raise ValueError(f"num_strata={self.num_strata} is no divisor of variance map of shape {variances.shape}")
        res = np.empty((self.num_strata * self.num_strata, 2), dtype=int)
        strat_width = variances.shape[1] // self.num_strata
        strat_height = variances.shape[0] // self.num_strata
        for y in range(self.num_strata):
            for x in range(self.num_strata):
                y_start = y * strat_height
                y_end = variances.shape[0] if y == self.num_strata - 1 else (y + 1) * strat_height
                x_start = x * strat_width
                x_end = variances.shape[1] if x == self.num_strata - 1 else (x + 1) * strat_width
                # If size not divisible by num_strata, make the strata on the bottom/right higher/wider to include excess
                this_start_width = x_end - x_start
                flat_index = np.argmax(variances[y_start:y_end, x_start:x_end])
                res[y * self.num_strata + x, 0] = y_start + (flat_index // this_start_width)
                res[y * self.num_strata + x, 1] = x_start + (flat_index % this_start_width)
        return res


def local_constant_liar_ivr(train_points: np.ndarray, test_points: np.ndarray, fidelity: int,
                            model, test_all_fidelities: bool, batch_size_per_fidelity):
    """
    :param train_points: [area_size ^ 2, 2] with the grid points we could add
    :param test_points: [(area_size + area_padding * 2) ^ 2, 2] with the points we evaluate the variance on
    :param fidelity: The fidelity to choose points from

    Note that this current method would allow to choose low fidelity points directly next to each other which
    means we would take less points than we could for the fidelity, because the neighbour point might be added
    anyway if it is in the same low fidelity cell. However, this shouldn't happen in practice as the variance at a
    neighbour should be relatively low

    TODO IMPORANT: I'm not quite sure how calculate_variance_reduction() calculates IVR as it doesn't really look like the
     formula we know. However, it does NOT look additive so adding the variance of the different points might lead
     to low-res being preferred most of the time because more variance reductions are calculated and added
    """
    original_data = (model.X, model.Y)
    train_points = np.concatenate((train_points, fidelity * np.ones((train_points.shape[0], 1))), axis=1)
    if test_all_fidelities:
        n_fidelities = len(batch_size_per_fidelity)
        test_points = np.tile(test_points, (n_fidelities, 1))
        test_points = np.concatenate((np.tile(test_points, (n_fidelities, 1)),
                                      np.arange(n_fidelities).repeat(test_points.shape[0], axis=0)[:, None]), axis=1)
    else:
        test_points = np.concatenate((test_points, fidelity * np.ones(test_points.shape[0], 1)), axis=1)
    res = np.empty((batch_size_per_fidelity[fidelity], 3))
    total_variance_reduction = 0
    for i in range(batch_size_per_fidelity[fidelity]):
        best_point_index = None  # array([x1, x2, fidelity])
        best_variance_reduction = -1
        for p_index, point in enumerate(train_points):
            variance_reduction = np.mean(model.calculate_variance_reduction(point[None, :], test_points))
            if variance_reduction > best_variance_reduction:
                best_variance_reduction = variance_reduction
                total_variance_reduction += variance_reduction
                best_point_index = p_index

        new_x = train_points[best_point_index, :][None, :]
        res[i, :] = new_x[0, :]
        new_y = model.predict(new_x)[0]

        # Add new point as fake observation in model
        all_x = np.concatenate([model.X, new_x], axis=0)
        all_y = np.concatenate([model.Y, new_y], axis=0)
        model.set_data(all_x, all_y)
        train_points = np.delete(train_points, best_point_index, axis=0)
    model.set_data(*original_data)
    return total_variance_reduction, res

class LocalBatchPointCalculator(CandidatePointCalculator):
    """
    Modified from: emukit.core.loop.candidate_point_calculators.GreedyBatchPointCalculator

    Batch point calculator. This point calculator calculates the first point in the batch then adds this as a fake
    observation in the model with a Y value equal to the mean prediction. The model is reset with the original data at
    the end of collecting a batch but if you use a model where training the model with the same data leads to different
    predictions, the model behaviour will be modified.
    """

    def __init__(
        self,
        model: GPyMultiOutputWrapper,  # ICalculateVarianceReduction and IModel,
        fidelity_costs,
        coords_domain: np.ndarray,
        num_points_width: int,
        num_points_height: int,
        batch_size: int,
        candidate_identifier: CandidatePointIdentifier,
        area_size: int,
        area_padding: int,
        test_all_fidelities: bool,
        num_processes: int=12
    ):
        """
        Note that the acquisition of variance / cost is baked in here so we can use vectorized numpy instead of loops
        for efficiency.

        :param model: Model that is used by the acquisition function
        :param fidelity_costs: [num_fidelities]: Cost of each fidelity. In particular, this means that one batch will
        give batch_size / fidelity_cost points in the fidelity it decided for
        :param coords_domain: An matrix of shape [num_points_width * num_points_height, 2] that contains the flattened
        version of a high-fidelity resolution grid of coordinates such that the top row comes first, then the
        second row etc.
        :param batch_size: Number of points to calculate in batch
        :param candidate_identifier: The CandidatePointIdentifier that determines which points should be the centers of
        the areas we try IVR on
        :param area_size: How many points into each direction a considered area should go. Should be uneven.
        :param area_padding: While the local IVR can only choose points from an area of sice area_size ^ 2, the variance
        will be evaluated on an area of size (area_padding + area_size + area_padding) ^ 2
        :param test_all_fidelities: If this is set, the local IVR will try to reduce the average variance of all
        fidelities instead of only the one it is sampling points from. This should perform better but also increase
        computational overhead
        :param num_processes: If != 1, a pool of num_processes different processes will compute IVR for the different
        candidate areas in parallel
        """
        if (not isinstance(batch_size, int)) or (batch_size < 1):
            raise ValueError("Batch size should be a positive integer")
        assert coords_domain.shape[0] == num_points_height * num_points_width and coords_domain.shape[1] == 2
        assert area_size % 2 == 1

        self.model = model
        self.fidelity_costs = fidelity_costs
        self.coords_domain = coords_domain
        self.num_points_width = num_points_width
        self.num_points_height = num_points_height
        self.batch_size = batch_size
        self.candidate_identifier = candidate_identifier
        self.area_size = area_size
        self.area_padding = area_padding
        self.test_all_fidelities = test_all_fidelities
        self.num_processes = 12

        # Preallocate/-compute for efficiency/convenience
        self.area_kernel = np.ones((self.area_size, self.area_size))

        self.batch_size_per_fidelity = [int(np.round(batch_size / c)) for c in fidelity_costs]
        self.min_coord_h = np.min(coords_domain[:, 0])
        self.max_coord_h = np.max(coords_domain[:, 0])
        self.min_coord_w = np.min(coords_domain[:, 1])
        self.max_coord_w = np.max(coords_domain[:, 1])

        # Create a meshgrid of shape [area_size * area_size, 2] that can be added to coordinates [x1, x2]
        # to obtain their surrounding coordinates
        _, _, self.area_meshgrid = self._get_subgrid_of_size(area_size)
        half_cell_height, half_cell_width, self.randomized_meshgrid = self._get_subgrid_of_size(area_size + 2 * self.area_padding)

        # We can interpret each point as center of a grid cell. Now we move each point to a random position within this
        # cell which is equivalent to latin hypercube sampling. Note that we use this meshgrid to generate the local
        # area points to evaluate the area on for integrated variance reduction (See Appendix of the paper)
        self.randomized_meshgrid[:, 0] += np.random.uniform(-half_cell_height, half_cell_height,
                                                            size=self.randomized_meshgrid.shape[0])
        self.randomized_meshgrid[:, 1] += np.random.uniform(-half_cell_width, half_cell_width,
                                                            size=self.randomized_meshgrid.shape[0])

    def _get_subgrid_of_size(self, area_size: int):
        area_meshgrid = np.copy(self.coords_domain.reshape(self.num_points_height, self.num_points_width,
                                                           2)[:area_size, :area_size, :])
        # normalize domain to [0, 1] (in our case, it should be [-1, 1] before but we don't rely on that)
        area_meshgrid[:, :, 0] = (area_meshgrid[:, :, 0] - self.min_coord_h) / (self.max_coord_h - self.min_coord_h)
        area_meshgrid[:, :, 1] = (area_meshgrid[:, :, 1] - self.min_coord_w) / (self.max_coord_w - self.min_coord_w)
        area_meshgrid -= area_meshgrid[(area_size - 1) // 2, (area_size - 1) // 2, :]

        half_cell_height = area_meshgrid[1, 1, 0] / 2
        half_cell_width = area_meshgrid[1, 1, 1] / 2

        return half_cell_height, half_cell_width, area_meshgrid.reshape(area_size * area_size, 2)

    def _filter_invalid_points(self, points: np.ndarray):
        """
        :param points: [num_points, 2]
        :return: filtered version without points outside of given range [min, max]
        Note that this assumes min_coord_w=min_coord_h and max_coord_w=max_coord_h for slight effieciency improvements. For general domains, a slight adjustment would be needed
        """
        return points[np.logical_and(np.all(points >= self.min_coord_h, axis=1),
                                     np.all(points <= self.max_coord_h, axis=1)), :]

    def compute_next_points(self, loop_state: LoopState, context: dict = None) -> np.ndarray:
        """
        :param loop_state: Object containing history of the loop
        :param context: Contains variables to fix through optimization of acquisition function. The dictionary key is
                        the parameter name and the value is the value to fix the parameter to.
        :return: 2d array of size (batch_size x input dimensions) of new points to evaluate
        """
        if self.num_processes <= 1:
            return self._compute_next_points_local()

        pool_args = []
        for f in range(len(self.fidelity_costs)):
            _, variance = self.model.predict(np.concatenate((self.coords_domain,
                                                             np.ones((self.coords_domain.shape[0], 1)) * f), axis=1))
            variance = variance.reshape(self.num_points_height, self.num_points_width)
            summed_variances = convolve2d(variance, self.area_kernel, mode="same")

            candidate_point_indices = self.candidate_identifier.identify_points(summed_variances)
            candidate_points = self.coords_domain[candidate_point_indices[:, 0] * self.num_points_width +
                                                  candidate_point_indices[:, 1], :]

            for point in candidate_points:
                # the _filter_invalid_points could also be moved to the processes
                pool_args.append([self._filter_invalid_points(self.area_meshgrid + point),
                                  self._filter_invalid_points(self.randomized_meshgrid + point), f,
                                  self.model, self.test_all_fidelities, self.batch_size_per_fidelity])

        with multiprocessing.Pool(self.num_processes) as pool:
            results = pool.starmap(local_constant_liar_ivr, pool_args)

        return max(results, key=lambda x: x[0])[1]

    def _compute_next_points_local(self):
        num_fidelities = len(self.fidelity_costs)

        best_variance_reduction = -1
        best_points = None  # [batch_size_for_fidelity, 2]

        for f in range(num_fidelities):
            _, variance = self.model.predict(np.concatenate((self.coords_domain,
                                                             np.ones((self.coords_domain.shape[0], 1)) * f), axis=1))
            variance = variance.reshape(self.num_points_height, self.num_points_width)
            summed_variances = convolve2d(variance, self.area_kernel, mode="same")

            candidate_point_indices = self.candidate_identifier.identify_points(summed_variances)
            candidate_points = self.coords_domain[candidate_point_indices[:, 0] * self.num_points_width +
                                                  candidate_point_indices[:, 1], :]

            for point in tqdm(candidate_points):
                # Whereas emukit IntegratedVarianceReduction uses a precalculated set of random points, this might not
                # make much sense for us. Additionally, it would be non-trivial to precalculate due to the varying
                # candidate points, so I just use linspace which is what we did in the tutorials/lecture
                variance_reduction, new_points = local_constant_liar_ivr(
                    self._filter_invalid_points(self.area_meshgrid + point),
                    self._filter_invalid_points(self.randomized_meshgrid + point), f,
                self.model, self.test_all_fidelities, self.batch_size_per_fidelity)

                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_points = new_points
        return best_points





    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: The new training point(s) to evaluate with shape (n_points x 1)
        :return: A numpy array with shape (n_points x 1)
                 containing the values of the acquisition evaluated at each x row
        """

        n_eval_points = x.shape[0]
        integrated_variance = np.zeros((n_eval_points, 1))
        for i in range(n_eval_points):
            # Find variance reduction at each Monte Carlo point
            variance_reduction = self.model.calculate_variance_reduction(x[[i], :], self._x_monte_carlo)
            # Take mean to approximate integral per unit volume
            integrated_variance[i] = np.mean(variance_reduction)

        return integrated_variance
