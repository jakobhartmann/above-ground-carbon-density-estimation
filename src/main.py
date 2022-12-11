from bayes_opt import *
from conversions import m_to_deg
from visualization import heatmap_comparison
import ee
import numpy as np
np.random.seed(20)
from data import DataLoad


# Trigger the authentication flow. Can comment out if auth token cached, eg after running it once
# ee.Authenticate()
# Initialize the library.
ee.Initialize()

# Parameters # TODO pass as args?
# Import the MODIS dataset
dataset = ee.ImageCollection('MODIS/061/MOD13Q1')
# Choose vegetation index band
veg_idx_band = 'NDVI'

# Data load option
#   'api', 'local' or 'optimal'(takes local if exist)
data_load_type = 'optimal'

# Define the center of our map.
lat, lon = 45.77, 4.855
center_point = np.array([[lat, lon]])

scale = 250  # scale in meters
num_points = 101  # per direction
num_points_plot = num_points
num_iter = 30 # number of iterations

dataloader = DataLoad(dataset, center_point, num_points, scale, veg_idx_band, data_load_type)

# Function that provides vegetation index, using parameters above
# NOTE: Function is not provided with values as paramters to simplify further refactoring for the data pipeline
def VI_at(coords):
    coords = center_point + m_to_deg(coords * (scale * num_points) / 2)

    geom_coords = ee.FeatureCollection(
        [ee.Geometry.Point(c[0], c[1]) for c in coords])
    samples = dataset.mean().reduceRegions(**{
        'collection': geom_coords,
        'scale': scale,
        'reducer': 'mean'}).getInfo()

    sample_results = []
    for sample in samples['features']:
        sample_results = sample_results + [sample['properties'][veg_idx_band]]

    # values = np.array([new_ds.mean().sample(ee.Geometry.Point(c[0], c[1]), scale).first().get(veg_idx_band).getInfo() for c in coords])[:, None]
    values = np.array(sample_results)[:, None]
    return values

# Do bayesian optimization over a given target function in 2 dimensions 
def basic_gp_example_old(target_function, num_points):
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)

    # Function we are trying to model
    target_function = VI_at

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
def basic_gp(dataloader, num_points):
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
    mu_plot, var_plot = emukit_model.predict(x_plot)

    # Show results
    print("x_plot info:", x_plot.shape, np.min(x_plot), np.max(x_plot))
    print("mu_plot info:", mu_plot.shape,
          np.min(mu_plot), np.max(mu_plot))
    print("y_plot info:", ground_truth.shape, np.min(ground_truth), np.max(ground_truth))
    heatmap_comparison(mu_plot, ground_truth, num_points, emukit_model)

if __name__ == '__main__':
    print('Starting basic GP example')
    # basic_gp_example(VI_at, num_points)
    basic_gp(dataloader, num_points)
    input()
