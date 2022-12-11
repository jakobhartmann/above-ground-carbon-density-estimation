import numpy as np
from emukit.core import ParameterSpace, DiscreteParameter
from emukit.experimental_design import ExperimentalDesignLoop
import GPy
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer
from data import DataLoad

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

def geographic_bayes_opt(dataloader:'DataLoad', x_space, y_space, X_init, num_iter=10):
    space = ParameterSpace([DiscreteParameter('x', x_space),
                            DiscreteParameter('y', y_space)])

    Y_init = dataloader.load_values(X_init)

    kern = GPy.kern.RBF(input_dim=2, lengthscale=0.08, variance=20)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kern, noise_var=1e-10)
    emukit_model = GPyModelWrapper(gpy_model)

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
