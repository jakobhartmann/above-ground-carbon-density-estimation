import matplotlib.pyplot as plt

from bayes_opt import *
from conversions import m_to_deg
from custom_logger import CustomLogger, LOGGER
from visualization import *
from benchmarking import *
import ee
import numpy as np
np.random.seed(20)
from data import DataLoad
import wandb

# Parameters # TODO pass as args?
# Import the MODIS dataset
# dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
# Choose vegetation index band
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
def mf_gp(dataloader_high:'DataLoad', dataloader_low:'DataLoad', num_points, num_iter, num_fidelities, low_fidelity_cost, high_fidelity_cost):
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot_high = np.stack((x_plot[1], x_plot[0], np.zeros(x_plot[0].shape)), axis=-1).reshape(-1, 3)
    x_plot_low = np.stack((x_plot[1], x_plot[0], np.ones(x_plot[0].shape)), axis=-1).reshape(-1, 3)
    
    # Load Data Pipeline
    dataloader_high.load_data()
    dataloader_low.load_data()


    # Bayesian optimization
    X1_init = np.array([(0.2, 0.4, 0.0), (0.6, -0.4, 0.0), (0.9, 0.0, 0.0)])
    # X2_init = np.array([[0.2, 0.4, 1.0], [0.6, -0.4, 1.0], [0.9, 0.0, 1.0]])
    X2_init = np.array([(0.4, 0.2, 1.0), (-0.4, 0.6, 1.0), (0.0, 0.9, 1.0)])
    emukit_model = mf_bayes_opt(dataloader_high, dataloader_low, x_space, y_space, X1_init, X2_init, num_iter=num_iter, num_fidelities=num_fidelities, low_fidelity_cost=low_fidelity_cost, high_fidelity_cost=high_fidelity_cost)
    ground_truth_high = dataloader_high.load_data_local()
    ground_truth_high_reshaped = ground_truth_high.reshape(num_points ** 2, 1)
    ground_truth_low = dataloader_low.load_data_local()

    mu_plot_high, var_plot_high = emukit_model.predict(x_plot_high)
    mu_plot_low, var_plot_low = emukit_model.predict(x_plot_low)
    std_plot_high = np.sqrt(var_plot_high)

    idx_unseen = (x_plot_high[:, None] != emukit_model.X).any(-1).all(1)
    x_unseen_high = x_plot_high[idx_unseen]
    mu_unseen_high, var_unseen_high = emukit_model.predict(x_unseen_high)
    std_unseen_high = np.sqrt(var_unseen_high)
    ground_truth_unseen_high = ground_truth_high_reshaped[idx_unseen]


    LOGGER.log(dict(
        mean_plots = heatmap_comparison_mf(mu_plot_high, mu_plot_low, ground_truth_high, ground_truth_low, num_points, emukit_model),
        # mean_plot_low=heatmap_comparison_mf(mu_plot_low, ground_truth_low, num_points, emukit_model, mf_choose=1.0),
        variance_plot_high=plot_variance(var_plot_high, num_points, emukit_model),
        variance_plot_low=plot_variance(var_plot_low, num_points, emukit_model),
        num_high_fidelity_samples = np.sum(emukit_model.X[:, 2] == 0) - len(X1_init),
        num_low_fidelity_samples = np.sum(emukit_model.X[:, 2] == 1) - len(X2_init),
        L1 = l1(mu_plot_high, ground_truth_high_reshaped),
        L2 = l2(mu_plot_high, ground_truth_high_reshaped),
        MSE = mse(mu_plot_high, ground_truth_high_reshaped),
        MPDF_unseen = mpdf(mu_unseen_high, std_unseen_high, ground_truth_unseen_high),
        MPDF_all = mpdf(mu_plot_high, std_plot_high, ground_truth_high_reshaped),
        PSNR = psnr(mu_plot_high, ground_truth_high_reshaped),
        SSIM = ssim(mu_plot_high, ground_truth_high_reshaped)
    ))

    # Show results
    print("x_plot info:", x_plot_high.shape, np.min(x_plot_high), np.max(x_plot_high))
    print("mu_plot info:", mu_plot_high.shape,
          np.min(mu_plot_high), np.max(mu_plot_high))
    print("y_plot info:", ground_truth_high.shape, np.min(ground_truth_high), np.max(ground_truth_high))


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
    emukit_model = geographic_bayes_opt(dataloader, x_space, y_space, X_init, num_iter=num_iter)
    ground_truth = dataloader.load_data_local()
    ground_truth_reshaped = ground_truth.reshape(num_points ** 2, 1)
    mu_plot, var_plot = emukit_model.predict(x_plot)
    std_plot = np.sqrt(var_plot)

    idx_unseen = (x_plot[:, None] != emukit_model.X).any(-1).all(1)
    x_unseen = x_plot[idx_unseen]
    mu_unseen, var_unseen = emukit_model.predict(x_unseen)
    std_unseen = np.sqrt(var_unseen)
    ground_truth_unseen = ground_truth_reshaped[idx_unseen]


    LOGGER.log(dict(
        mean_plot=heatmap_comparison(mu_plot, ground_truth, num_points, emukit_model),
        variance_plot=plot_variance(var_plot, num_points, emukit_model),
        L1 = l1(mu_plot, ground_truth_reshaped),
        L2 = l2(mu_plot, ground_truth_reshaped),
        MSE = mse(mu_plot, ground_truth_reshaped),
        MPDF_unseen = mpdf(mu_unseen, std_unseen, ground_truth_unseen),
        MPDF_all = mpdf(mu_plot, std_plot, ground_truth_reshaped),
        PSNR = psnr(mu_plot, ground_truth_reshaped),
        SSIM = ssim(mu_plot, ground_truth_reshaped)
    ))

    # Show results
    print("x_plot info:", x_plot.shape, np.min(x_plot), np.max(x_plot))
    print("mu_plot info:", mu_plot.shape,
          np.min(mu_plot), np.max(mu_plot))
    print("y_plot info:", ground_truth.shape, np.min(ground_truth), np.max(ground_truth))

if __name__ == '__main__':
    print('Starting basic GP example')
    # basic_gp_example(VI_at, num_points)

    config = dict(
        num_fidelities=2,
        source = 'MODIS/061/MOD13Q1',
        source_low = 'MODIS/061/MOD13A2',
        scale=250,  # scale in meters
        scale_low=250, # scale low fidelity
        low_fidelity_cost = 1,
        high_fidelity_cost = 2,
        num_points = 101,  # per direction
        num_points_plot = 101,
        num_iter = 100,  # number of iterations
        # lat = 45.77,
        # lon= 4.855
        # points with water
        # lat = -82.8642,
        # lon = 42.33
    )
    LOGGER = CustomLogger(use_wandb=False, config=config)
    center_point = np.array([[LOGGER.config["lat"], LOGGER.config["lon"]]])
    # dataloader = DataLoad(LOGGER.config["source"], center_point, LOGGER.config["num_points"], LOGGER.config["scale"], veg_idx_band, data_load_type)
    # basic_gp(dataloader, LOGGER.config["num_points"], LOGGER.config["num_iter"])
    dataloader1 = DataLoad(LOGGER.config["source"], center_point, LOGGER.config["num_points"], LOGGER.config["scale"], veg_idx_band, data_load_type)
    dataloader2 = DataLoad(LOGGER.config["source_low"], center_point, LOGGER.config["num_points"], LOGGER.config["scale_low"], veg_idx_band, data_load_type)
    mf_gp(dataloader1, dataloader2, LOGGER.config["num_points"], LOGGER.config["num_iter"], LOGGER.config["num_fidelities"], LOGGER.config["low_fidelity_cost"], LOGGER.config["high_fidelity_cost"])
    input()