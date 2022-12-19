from bayes_opt import *
from conversions import m_to_deg
from visualization import heatmap_comparison
import ee
import numpy as np
np.random.seed(20)
from data import DataLoad
from emukit.test_functions.forrester import forrester, forrester_low
import GPy
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from matplotlib import pyplot as plt
# https://emukit.github.io/multifidelity-emulation/#references-on-multi-fidelity-gaussian-processes

if __name__ == '__main__':
    # Adapted from https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity.ipynb
    num_fidelities = 2
    
    
    x_plot = np.linspace(0, 1.0, 200)[:,None]
    y_plot_l = forrester_low(x_plot)
    y_plot_h = forrester(x_plot)

    x_train_l = np.atleast_2d(np.random.rand(12)).T
    x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:6])
    y_train_l = forrester_low(x_train_l)
    y_train_h = forrester(x_train_h)
    X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], [y_train_l, y_train_h])

    kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
    linear_mf_kernel = LinearMultiFidelityKernel(kernels)

    gpy_linear_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, linear_mf_kernel, n_fidelities = num_fidelities)
    
    gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(0)


    # gpy_linear_mf_model.optimize()

    # meshgrid of space
    # y_space = np.linspace(0, 1.0, 100)f
    # x_plot = np.meshgrid(x_space, y_space)
    # x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)
    # space = ParameterSpace([DiscreteParameter('x', x_plot)])

    emukit_model = model= GPyMultiOutputWrapper(gpy_linear_mf_model, num_fidelities, n_optimization_restarts=1)
    
    # us_acquisition = ModelVariance(emukit_model)
    # for i in range(10):
    #     # x_new, _ = optimizer.optimize(us_acquisition, Y_metadata={'output_index':np.zeros((x_plot.shape[0],1))[:,None].astype(int)})
    #     x_new, _ = optimizer.optimize(us_acquisition)
    #     y_new = [forrester_low(x_new), forrester(x_new)]
    #     X = np.append(X, [x_new, x_new], axis=0)
    #     Y = np.append(Y, y_new, axis=0)
    # emukit_model.set_data(X, Y)

    emukit_model.optimize()
        # emukit_model.optimize()
    # print(space.parameters)

    # ed = ExperimentalDesignLoop(space=space, model=emukit_model, acquisition=us_acquisition)

    # ed.run_loop(forrester, num_iter)
    X_plot = convert_x_list_to_array([x_plot, x_plot])
    X_plot_l = X_plot[:len(x_plot)]
    X_plot_h = X_plot[len(x_plot):]

    mu_l_plot, var_l_plot = emukit_model.predict(X_plot_l)
    mu_h_plot, var_h_plot = emukit_model.predict(X_plot_h)

    ## Plot the posterior mean and variance
    
    plt.figure(figsize=(12, 8))
    plt.fill_between(x_plot.flatten(), (mu_l_plot - 1.96*np.sqrt(var_l_plot)).flatten(), 
                    (mu_l_plot + 1.96*np.sqrt(var_l_plot)).flatten(), facecolor='g', alpha=0.3)
    plt.fill_between(x_plot.flatten(), (mu_h_plot - 1.96*np.sqrt(var_h_plot)).flatten(), 
                    (mu_h_plot + 1.96*np.sqrt(var_h_plot)).flatten(), facecolor='y', alpha=0.3)

    plt.plot(x_plot, y_plot_l, 'b')
    plt.plot(x_plot, y_plot_h, 'r')
    plt.plot(x_plot, mu_l_plot, '--', color='g')
    plt.plot(x_plot, mu_h_plot, '--', color='y')
    plt.scatter(x_train_l, y_train_l, color='b', s=40)
    plt.scatter(x_train_h, y_train_h, color='r', s=40)
    plt.ylabel('f (x)')
    plt.xlabel('x')
    plt.legend(['Low Fidelity', 'High Fidelity', 'Predicted Low Fidelity', 'Predicted High Fidelity'])
    plt.title('Linear multi-fidelity model fit to low and high fidelity Forrester function')

    plt.show()

    # mu_plot, var_plot = emukit_model.predict(x_plot, Y_metadata={'output_index':np.zeros((x_plot.shape[0],1))[:,None].astype(int)})

    # heatmap_comparison(mu_plot, mu_plot, 100, gpy_linear_mf_model)

