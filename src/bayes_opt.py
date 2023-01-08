import numpy as np
from emukit.core import ParameterSpace, DiscreteParameter, InformationSourceParameter
from emukit.experimental_design import ExperimentalDesignLoop
import GPy
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.acquisition import Acquisition
from emukit.core.loop import FixedIntervalUpdater, OuterLoop, SequentialPointCalculator
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import gradient_acquisition_optimizer
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.experimental_design.acquisitions import ModelVariance, IntegratedVarianceReduction

# from emukit.test_functions.forrester import multi_fidelity_forrester_function
from data import DataLoad
from constants import *
from custom_loop import CustomLoop
from custom_kernels import WaterRBFKernel
from vegetation import WaterUtils
from src.local_ivr import LocalBatchPointCalculator, LatinHypercubeMaximaIdentifier


def kernel(config):
    print("Using kernels: ", config[KERNELS], " with combination: ", config[KERNEL_COMBINATION])
    combination = []
    if RBF in config[KERNELS].lower():
        combination.append(GPy.kern.RBF(input_dim=2, lengthscale=config[RBF_LENGTHSCALE], variance=config[RBF_VARIANCE]))
    if WHITE in config[KERNELS]:
        combination.append(GPy.kern.White(input_dim=2, variance=config[WHITE_VARIANCE]))
    if PERIODIC in config[KERNELS]:
        combination.append(GPy.kern.StdPeriodic(input_dim=2, lengthscale=config[PERIODIC_LENGTHSCALE], period=config[PERIODIC_PERIOD], variance=config[PERIODIC_VARIANCE]))
    if MATERN32 in config[KERNELS]:
        combination.append(GPy.kern.Matern32(input_dim=2, lengthscale=config[MATERN32_LENGTHSCALE], variance=config[MATERN32_VARIANCE]))
    
    # Return first kernel if only one kernel is used
    if len(combination) == 1:
        return combination[0]

    if config[KERNEL_COMBINATION] == PRODUCT:
        return GPy.kern.Prod(combination)
    elif config[KERNEL_COMBINATION] == SUM:
        return GPy.kern.Add(combination)
    else:
         # Default to sum.
        return GPy.kern.Add(combination)

def geographic_bayes_opt_no_dataloader(target_function, x_space, y_space, X_init, num_iter=10):
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space)])

    Y_init = target_function(X_init)

    kern = GPy.kern.RBF(input_dim=2, lengthscale=0.08, variance=20)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kern, noise_var=1e-10)
    emukit_model = GPyModelWrapper(gpy_model)

    us_acquisition = ModelVariance(emukit_model)
    ivr_acquisition = IntegratedVarianceReduction(emukit_model, space)

    optimizer = GradientAcquisitionOptimizer(space)
    x_new, _ = optimizer.optimize(us_acquisition)
    y_new = target_function(x_new)
    X = np.append(X_init, x_new, axis=0)
    Y = np.append(Y_init, y_new, axis=0)

    emukit_model.set_data(X, Y)
    # print(space.parameters)

    ed = ExperimentalDesignLoop(space=space, model=emukit_model)

    ed.run_loop(target_function, num_iter)
    return emukit_model

def geographic_bayes_opt(dataloader:'DataLoad', x_space, y_space, X_init, logger):
    config = logger.config
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space)])

    Y_init = dataloader.load_values(X_init)

    # Prep plotspace
    x_plot = np.meshgrid(x_space, y_space)
    x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)

    # Load ground truth as dataset
    ground_truth = dataloader.load_data_local()
    ground_truth_reshaped = ground_truth.reshape(dataloader.num_points ** 2, 1)

    # final kernel selected
    kern = kernel(config)
    # kern.plot()
    # print(kern.name)

    gpy_model = GPy.models.GPRegression(X_init, Y_init, kern, noise_var=config[MODEL_NOISE_VARIANCE])
    # gpy_model = GPy.models.GPRegressionGrid(X_init, Y_init, kern)
    emukit_model = GPyModelWrapper(gpy_model, n_restarts=config[OPTIMIZATION_RESTARTS])

    us_acquisition = ModelVariance(emukit_model)
    ivr_acquisition = IntegratedVarianceReduction(emukit_model, space)

    optimizer = GradientAcquisitionOptimizer(space)
    x_new, _ = optimizer.optimize(us_acquisition)
    y_new = dataloader.load_values(x_new)
    X = np.append(X_init, x_new, axis=0)
    Y = np.append(Y_init, y_new, axis=0)

    emukit_model.set_data(X, Y)
    # print(space.parameters)

    ed = ExperimentalDesignLoop(space=space, model=emukit_model)

    def log_metrics_basic(loop, loop_state):
        # print(f'Logging metrics on iteration {loop_state.iteration}')
        # Get predictions
        mu_plot, var_plot = loop.model_updaters[0].model.predict(x_plot)
        std_plot = np.sqrt(var_plot)

        # Separate unseen data for special metrics
        idx_unseen = (x_plot[:, None] != loop.model_updaters[0].model.X).any(-1).all(1)
        x_unseen = x_plot[idx_unseen]
        mu_unseen, var_unseen = loop.model_updaters[0].model.predict(x_unseen)
        std_unseen = np.sqrt(var_unseen)
        ground_truth_unseen = ground_truth_reshaped[idx_unseen]

        logger.log_metrics(ground_truth_reshaped, mu_plot, std_plot, mu_unseen, std_unseen, ground_truth_unseen)
    
    # subscribe to events during loop
    ed.iteration_end_event.append(log_metrics_basic)

    ed.run_loop(dataloader.load_values, config[NUM_ITER])
    return emukit_model

# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]
    
    @property
    def has_gradients(self):
        # return True
        return False # for ModelVariance
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)

def mf_bayes_opt(dataloader1:'DataLoad', dataloader2:'DataLoad', x_space, y_space, X1_init, X2_init, logger):
    config = logger.config
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space),
                            InformationSourceParameter(config[NUM_FIDELITIES])])

    step = [np.abs(x_space[0]-x_space[1]), np.abs(y_space[0]-y_space[1])]

    Y1_init = dataloader1.load_values(X1_init)
    Y2_init = dataloader2.load_values(X2_init)
    Y_init = np.concatenate((Y1_init, Y2_init))
    X_init = np.concatenate((X1_init, X2_init))

    # Prep plotspace
    x_plot = np.meshgrid(x_space, y_space)
    x_plot_high = np.stack((x_plot[1], x_plot[0], np.zeros(x_plot[0].shape)), axis=-1).reshape(-1, 3)
    # x_plot_low = np.stack((x_plot[1], x_plot[0], np.ones(x_plot[0].shape)), axis=-1).reshape(-1, 3)

    # Load ground truth as dataset
    ground_truth_high = dataloader1.load_data_local()
    ground_truth_high_reshaped = ground_truth_high.reshape(dataloader1.num_points ** 2, 1)

    # Custom kernels
    vegetationDataLoader = DataLoad(source = "COPERNICUS/Landcover/100m/Proba-V-C3/Global", center_point = np.array([[-82.8642, 42.33]]), num_points = 101, scale = 250, veg_idx_band = 'discrete_classification', data_load_type = 'optimal')
    water_utils = WaterUtils(dataLoader = vegetationDataLoader, water_value = 80)
    bitmask_land_land, bitmask_land_water, bitmask_water_water = water_utils.get_bitmasks()
    high_fidelity_water_rbf_kernel = WaterRBFKernel(input_dim = 1, variance_land = 20, lengthscale_land = 3, bitmask_land_land = bitmask_land_land, bitmask_water_water = bitmask_water_water)
    kernels = [kernel(config), high_fidelity_water_rbf_kernel] # NOTE: This list must be in order of low to high fidelity

    kernels = [kernel(config), GPy.kern.RBF(input_dim=2, lengthscale=3, variance=20.0)] # NOTE: This list must be in order of low to high fidelity
    linear_mf_kernel = LinearMultiFidelityKernel(kernels)
    gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_init, Y_init, linear_mf_kernel, n_fidelities=config[NUM_FIDELITIES])
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    
    emukit_model = GPyMultiOutputWrapper(gpy_linear_mf_model, config[NUM_FIDELITIES]+1, n_optimization_restarts=config[OPTIMIZATION_RESTARTS], verbose_optimization=True)

    emukit_model.optimize()

    # Acquisition function
    cost_acquisition = Cost([config[HIGH_FIDELITY_COST], config[LOW_FIDELITY_COST]])
    acquisition = MultiInformationSourceEntropySearch(emukit_model, space) / cost_acquisition
    mumbo_acquisition = MUMBO(emukit_model, space, num_samples=5, grid_size=500) / cost_acquisition
    model_variance = ModelVariance(emukit_model) / cost_acquisition
    integrated_variance_reduction = IntegratedVarianceReduction(emukit_model, space) / cost_acquisition
    # Create Outer Loop
    initial_loop_state = create_loop_state(X_init, Y_init)
    acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(space), space)
    candidate_point_calculator = SequentialPointCalculator(model_variance, acquisition_optimizer)
    hypercube_maxima = LatinHypercubeMaximaIdentifier(4)
    # candidate_point_calculator = LocalBatchPointCalculator(emukit_model, [3, 1], x_plot_high[:, :2], x_space.shape[0],
    #                                                        x_space.shape[0], 15, hypercube_maxima, 21, 3, True)
    model_updater = FixedIntervalUpdater(emukit_model)
    loop = OuterLoop(candidate_point_calculator, model_updater, initial_loop_state)
    # loop = CustomLoop(candidate_point_calculator, model_updater, initial_loop_state, step)

    def log_metrics_mf(loop, loop_state):
        # print(f'Logging metrics on iteration {loop_state.iteration}')
        # Get predictions
        mu_plot_high, var_plot_high = loop.model_updaters[0].model.predict(x_plot_high)
        # mu_plot_low, var_plot_low = loop.model_updaters[1].model.predict(x_plot_low)
        std_plot_high = np.sqrt(var_plot_high)

        # Separate unseen data for special metrics
        idx_unseen = (x_plot_high[:, None] != loop.model_updaters[0].model.X).any(-1).all(1)
        x_unseen_high = x_plot_high[idx_unseen]
        mu_unseen_high, var_unseen_high = loop.model_updaters[0].model.predict(x_unseen_high)
        std_unseen_high = np.sqrt(var_unseen_high)
        ground_truth_unseen_high = ground_truth_high_reshaped[idx_unseen]

        # High fidelity metrics
        logger.log_metrics(ground_truth_high_reshaped, mu_plot_high, std_plot_high, mu_unseen_high, std_unseen_high, ground_truth_unseen_high)


    # subscribe to events during loop
    loop.iteration_end_event.append(log_metrics_mf)

    loop_function = MultiSourceFunctionWrapper([
        lambda x: dataloader1.load_values(x),
        lambda x: dataloader2.load_values(x)])

    loop.run_loop(loop_function, config[NUM_ITER])

    return emukit_model
