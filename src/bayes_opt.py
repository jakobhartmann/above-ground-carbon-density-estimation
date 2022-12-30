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

def geographic_bayes_opt(dataloader:'DataLoad', x_space, y_space, X_init, num_iter, config):
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space)])

    Y_init = dataloader.load_values(X_init)

    # RBF_var = 20
    # RBF_lengthscale = 0.08

    k1 = GPy.kern.RBF(input_dim=2, lengthscale=config[RBF_LENGTHSCALE], variance=config[RBF_VARIANCE])
    k2 = GPy.kern.Matern32(input_dim=2, lengthscale=config[MATERN_LENGTHSCALE], variance=config[MATERN_VARIANCE])
    k3 = GPy.kern.StdPeriodic(input_dim=2, lengthscale=config[PERIODIC_LENGTHSCALE], variance=config[PERIODIC_VARIANCE], period=config[PERIODIC_PERIOD])
    k4 = GPy.kern.White(input_dim=2, variance=config[WHITE_VARIANCE])
    
    # product of kernels
    k_prod = k2 * k3 * k4
    # k_prod.plot()

    # Sum of kernels
    k_add = k2 + k3 + k4
    # k_add.plot()
    # hierarchic_comb = GPy.kern.Hierarchical(kern)

    # final kernel selected
    kern = k2
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

    ed.run_loop(dataloader.load_values, num_iter)
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

def mf_bayes_opt(dataloader1:'DataLoad', dataloader2:'DataLoad', x_space, y_space, X1_init, X2_init, num_iter=10, num_fidelities=2, low_fidelity_cost=1, high_fidelity_cost=5, config=None):
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space),
                            InformationSourceParameter(num_fidelities)])

    Y1_init = dataloader1.load_values(X1_init)
    Y2_init = dataloader2.load_values(X2_init)
    Y_init = np.concatenate((Y1_init, Y2_init))
    X_init = np.concatenate((X1_init, X2_init))


    kernels = [GPy.kern.RBF(input_dim=2, lengthscale=0.1, variance=20.0), GPy.kern.RBF(input_dim=2, lengthscale=3, variance=20.0)]
    # kern = GPy.kern.RBF(input_dim=2, lengthscale=0.08, variance=20)
    linear_mf_kernel = LinearMultiFidelityKernel(kernels)
    gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_init, Y_init, linear_mf_kernel, n_fidelities = num_fidelities)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    
    emukit_model = GPyMultiOutputWrapper(gpy_linear_mf_model, num_fidelities+1, n_optimization_restarts=config[OPTIMIZATION_RESTARTS], verbose_optimization=True)

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
    model_updater = FixedIntervalUpdater(emukit_model, config[OPTIMIZER_UPDATE_INTERVAL])
    loop = OuterLoop(candidate_point_calculator, model_updater, initial_loop_state)

    # loop.iteration_end_event.append(plot_acquisition)

    loop_function = MultiSourceFunctionWrapper([
        lambda x: dataloader1.load_values(x),
        lambda x: dataloader2.load_values(x)])

    loop.run_loop(loop_function, num_iter)

    return emukit_model
