import matplotlib.pyplot as plt

from bayes_opt import *
from conversions import m_to_deg
from custom_logger import CustomLogger, LOGGER # type: ignore
from visualization import heatmap_comparison, plot_2D_vis, plot_variance
from benchmarking import *
import ee
import numpy as np
np.random.seed(20)
from data import DataLoad
import wandb

# Parameters # TODO pass as args?
# Import the MODIS dataset
# dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
# Choose vegetation index band. For our datasets, either 'NDVI' or 'EVI'.
veg_idx_band = 'NDVI'

# Data load option
#   'api', 'local' or 'optimal'(takes local if exist)
data_load_type = 'optimal'

# Function that provides vegetation index, using parameters above
# NOTE: Function is not provided with values as paramters to simplify further refactoring for the data pipeline
# def VI_at(coords):
#     coords = center_point + m_to_deg(coords * (scale * num_points) / 2)
#
#     geom_coords = ee.FeatureCollection(
#         [ee.Geometry.Point(c[0], c[1]) for c in coords])
#     samples = dataset.mean().reduceRegions(**{
#         'collection': geom_coords,
#         'scale': scale,
#         'reducer': 'mean'}).getInfo()
#
#     sample_results = []
#     for sample in samples['features']:
#         sample_results = sample_results + [sample['properties'][veg_idx_band]]
#
#     # values = np.array([new_ds.mean().sample(ee.Geometry.Point(c[0], c[1]), scale).first().get(veg_idx_band).getInfo() for c in coords])[:, None]
#     values = np.array(sample_results)[:, None]
#     return values

# Do bayesian optimization over a given target function in 2 dimensions
def basic_gp_example_old(target_function, num_points, num_iter):
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)

    # Function we are trying to model
    # target_function = VI_at

    # Bayesian optimization
    X_init = np.array([[0.2, 0.4], [0.6, -0.4], [0.9, 0.0]])
    emukit_model = geographic_bayes_opt_no_dataloader(target_function, x_space, y_space, X_init, num_iter=num_iter)
    ground_truth = target_function(x_plot)
    mu_plot, var_plot = emukit_model.predict(x_plot)

    # Show results
    print("x_plot info:", x_plot.shape, np.min(x_plot), np.max(x_plot))
    print("mu_plot info:", mu_plot.shape,
          np.min(mu_plot), np.max(mu_plot))
    print("y_plot info:", ground_truth.shape, np.min(ground_truth), np.max(ground_truth))
    heatmap_comparison(mu_plot, ground_truth, num_points, emukit_model)

# Do bayesian optimization over a given target function in 2 dimensions
def basic_gp(dataloader, num_points, num_iter):
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)

    # Load Data Pipeline
    dataloader.load_data()

    # Bayesian optimization
    X_init = np.array([[0.2, 0.4], [0.6, -0.4], [0.9, 0.0]])
    emukit_model = geographic_bayes_opt(dataloader, x_space, y_space, X_init, num_iter, LOGGER.config)
    ground_truth = dataloader.load_data_local()
    ground_truth_reshaped = ground_truth.reshape(num_points ** 2, 1)
    mu_plot, var_plot = emukit_model.predict(x_plot)


    LOGGER.log(dict(
        mean_plot=heatmap_comparison(mu_plot, ground_truth, num_points, emukit_model),
        variance_plot=plot_variance(var_plot, num_points, emukit_model),
        L1 = l1(mu_plot, ground_truth_reshaped),
        L2 = l2(mu_plot, ground_truth_reshaped),
        MSE = mse(mu_plot, ground_truth_reshaped),
        PSNR = psnr(mu_plot, ground_truth_reshaped),
        SSIM = ssim(mu_plot, ground_truth_reshaped)
    ))
    # close plots
    plt.close('all')
    # TODO Use wandb.summary for these summary statistics instead?

    # Show results
    print("x_plot info:", x_plot.shape, np.min(x_plot), np.max(x_plot))
    print("mu_plot info:", mu_plot.shape,
          np.min(mu_plot), np.max(mu_plot))
    print("y_plot info:", ground_truth.shape, np.min(ground_truth), np.max(ground_truth))


def main():
    center_point = np.array([[LOGGER.config["lat"], LOGGER.config["lon"]]])
    dataloader = DataLoad(center_point, LOGGER.config["num_points"], LOGGER.config["scale"], veg_idx_band, data_load_type)
    basic_gp(dataloader, LOGGER.config["num_points"], LOGGER.config["num_iter"])


if __name__ == '__main__':
    print('Starting basic GP example')
    # ee.Authenticate()
    # wandb.login()
    # basic_gp_example(VI_at, num_points)

    config = dict(
        scale = 250,  # scale in meters
        num_points = 101,  # per direction
        num_points_plot = 101,
        num_iter = 10,  # number of iterations
        lat = 45.77,
        lon= 4.855,
        variance=20.0,
        lengthscale=0.08,
    )
    # Define the kernel parameter search
    sweep_config = {
        'name': 'Kernel parameter search',
        'method': 'random',
        'metric': {
            'name': 'L2',
            'goal': 'minimize',
        },
    }
    parameters_dict = {
        # 'kernels': {
        #     'values': ['rbf', 'matern']
        # },
        'variance': {
            'distribution': 'uniform',
            'min': 0,
            'max': 50,
            # 'values': np.linspace(0.0, 50.0, 10),
        },
        'lengthscale': {
            'distribution': 'log_uniform_values',
            'min': 10**(-5),
            'max': 10**(5),
            # 'values': [i for i in np.logspace(-5, 5, 20, base=10)],
        },
        'Matern_nu': {
            'distribution': 'categorical',
            'values': [0.5, 1.5, 2.5] #, np.inf],
        }
    }
    sweep_config['parameters'] = parameters_dict
    print("starting logger")
    LOGGER: CustomLogger = CustomLogger(use_wandb=True, config=config)
    if sweep_config:
        # self.sweep_id = self._wandb_instance.sweep(sweep_config, project="sensor-placement", entity="camb-mphil")
        LOGGER.sweep_id = LOGGER._wandb_instance.sweep(sweep_config, project="test-sensor-placement", entity="sepand")
        print('sweep initialized')
    main()
    LOGGER._wandb_instance.agent(LOGGER.sweep_id, main, count=5)
    # LOGGER.stop_run()
