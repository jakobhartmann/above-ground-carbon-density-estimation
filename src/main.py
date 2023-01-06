import argparse
import ee
import numpy as np
np.random.seed(20)
from data import DataLoad, NormalDataLoad
import wandb

from bayes_opt import *
from visualization import *
from custom_logger import CustomLogger


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
def mf_gp(dataloader_high:'DataLoad', dataloader_low:'DataLoad', num_points):
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot_high = np.stack((x_plot[1], x_plot[0], np.zeros(x_plot[0].shape)), axis=-1).reshape(-1, 3)
    x_plot_low = np.stack((x_plot[1], x_plot[0], np.ones(x_plot[0].shape)), axis=-1).reshape(-1, 3)
    
    # Load Data Pipeline
    dataloader_high.load_data()
    dataloader_low.load_data()

    # find max and min for normalization
    data_max_val = np.max((dataloader_high.max_val, dataloader_low.max_val))
    data_min_val = np.min((dataloader_high.min_val, dataloader_low.min_val))

    #normalize data
    dataloader_high.normalize_data(data_max_val, data_min_val)
    dataloader_low.normalize_data(data_max_val, data_min_val)

    # Bayesian optimization
    X1_init = np.array([(0.2, 0.4, 0.0), (0.6, -0.4, 0.0), (0.9, 0.0, 0.0)])
    # X2_init = np.array([[0.2, 0.4, 1.0], [0.6, -0.4, 1.0], [0.9, 0.0, 1.0]])
    X2_init = np.array([(0.4, 0.2, 1.0), (-0.4, 0.6, 1.0), (0.0, 0.9, 1.0)])
    emukit_model = mf_bayes_opt(dataloader_high, dataloader_low, x_space, y_space, X1_init, X2_init, logger=LOGGER)

    # Get predictions
    mu_plot_high, var_plot_high = emukit_model.predict(x_plot_high)
    mu_plot_low, var_plot_low = emukit_model.predict(x_plot_low)
    std_plot_high = np.sqrt(var_plot_high)

    # Ground truth data
    ground_truth_high = dataloader_high.load_data_local()
    ground_truth_high_reshaped = ground_truth_high.reshape(num_points ** 2, 1)
    ground_truth_low = dataloader_low.load_data_local()

    # Separate unseen data for special metrics
    idx_unseen = (x_plot_high[:, None] != emukit_model.X).any(-1).all(1)
    x_unseen_high = x_plot_high[idx_unseen]
    mu_unseen_high, var_unseen_high = emukit_model.predict(x_unseen_high)
    std_unseen_high = np.sqrt(var_unseen_high)
    ground_truth_unseen_high = ground_truth_high_reshaped[idx_unseen]

    # High fidelity metrics
    LOGGER.log_metrics(ground_truth_high_reshaped, mu_plot_high, std_plot_high, mu_unseen_high, std_unseen_high, ground_truth_unseen_high)

    # Log summary of results
    LOGGER.log(dict(
        mean_plot_high=heatmap_comparison_mf(mu_plot_high, mu_plot_low, ground_truth_high, ground_truth_low, num_points, emukit_model, backend=LOGGER.config[MATPLOTLIB_BACKEND]),
        # mean_plot_low=heatmap_comparison_mf(mu_plot_low, ground_truth_low, num_points, emukit_model, mf_choose=1.0, backend=LOGGER.config[MATPLOTLIB_BACKEND]),
        variance_plot_high=plot_variance(var_plot_high, num_points, emukit_model, backend=LOGGER.config[MATPLOTLIB_BACKEND]),
        variance_plot_low=plot_variance(var_plot_low, num_points, emukit_model, backend=LOGGER.config[MATPLOTLIB_BACKEND]),
        num_high_fidelity_samples = np.sum(emukit_model.X[:, 2] == 0) - len(X1_init),
        num_low_fidelity_samples = np.sum(emukit_model.X[:, 2] == 1) - len(X2_init),
    ))
    # plt.close('all')

    # Show results
    print("x_plot info:", x_plot_high.shape, np.min(x_plot_high), np.max(x_plot_high))
    print("mu_plot info:", mu_plot_high.shape,
          np.min(mu_plot_high), np.max(mu_plot_high))
    print("y_plot info:", ground_truth_high.shape, np.min(ground_truth_high), np.max(ground_truth_high))


# Do bayesian optimization over a given target function in 2 dimensions
def basic_gp(dataloader, num_points):
    print('Starting basic GP example')
    # Setup the domain of our estimation
    x_space = np.linspace(-1, 1, num_points)
    y_space = np.linspace(-1, 1, num_points)
    x_plot = np.meshgrid(x_space, y_space)
    x_plot = np.stack((x_plot[1], x_plot[0]), axis=-1).reshape(-1, 2)

    # Load Data Pipeline
    dataloader.load_data()

    # Bayesian optimization setup
    X_init = np.array([[0.2, 0.4], [0.6, -0.4], [0.9, 0.0]])
    # X_init = np.array([[0.2, 0.4], [0.6, -0.4], [0.9, 0.0], [-0.8, 0.8], [0.0, 0.7], [0.7, 0.5], [-0.8, -0.8], [-0.3, -0.7], [-0.1, 0.0]])
    emukit_model = geographic_bayes_opt(dataloader, x_space, y_space, X_init, logger=LOGGER)

    # Get predictions
    mu_plot, var_plot = emukit_model.predict(x_plot)
    std_plot = np.sqrt(var_plot)

    # Get ground truth
    ground_truth = dataloader.load_data_local()
    ground_truth_reshaped = ground_truth.reshape(num_points ** 2, 1)

    # Separate unseen data for special metrics
    idx_unseen = (x_plot[:, None] != emukit_model.X).any(-1).all(1)
    x_unseen = x_plot[idx_unseen]
    mu_unseen, var_unseen = emukit_model.predict(x_unseen)
    std_unseen = np.sqrt(var_unseen)
    ground_truth_unseen = ground_truth_reshaped[idx_unseen]

    heatmap_plot = heatmap_comparison(mu_plot, ground_truth, num_points, emukit_model, backend=LOGGER.config[MATPLOTLIB_BACKEND])
    variance_plot = plot_variance(var_plot, num_points, emukit_model, backend=LOGGER.config[MATPLOTLIB_BACKEND])
    # x_labels, y_labels = (emukit_model.X[:, 1] + 1) * num_points / 2, (emukit_model.X[:, 0] + 1) * num_points / 2
    # plt.close('all')

    # Metrics
    LOGGER.log_metrics(ground_truth_reshaped, mu_plot, std_plot, mu_unseen, std_unseen, ground_truth_unseen)


    # Log summary of results
    LOGGER.log(dict(
        mean_plot = heatmap_plot,
        variance_plot = variance_plot,
        # TODO wandb.plots deprected. Change to wandb.plot
        # mean = wandb.plots.HeatMap(x_labels=x_labels, y_labels=y_labels, matrix_values=mu_plot.reshape(num_points, num_points)),
        # ground_truth = wandb.plots.HeatMap(x_labels=x_labels, y_labels=y_labels, matrix_values=ground_truth.reshape(num_points, num_points)),
        # variance = wandb.plots.HeatMap(x_labels=x_labels, y_labels=y_labels, matrix_values=var_plot.reshape(num_points, num_points)),
    ))

    # Show results
    print("x_plot info:", x_plot.shape, np.min(x_plot), np.max(x_plot))
    print("mu_plot info:", mu_plot.shape,
          np.min(mu_plot), np.max(mu_plot))
    print("y_plot info:", ground_truth.shape, np.min(ground_truth), np.max(ground_truth))


def main(use_wandb=True):
    # ee.Authenticate()
    # wandb.login()
    # basic_gp_example(VI_at, num_points)

    # Setup config
    config = dict(
        source = 'MODIS/061/MOD13Q1',
        source_low = 'MODIS/061/MOD13A2',
        scale=250,  # scale in meters
        scale_low=250, # scale low fidelity
        num_points = 101,  # per direction
        num_points_plot = 101,
        lat = 45.77,
        lon = 4.855,
        # points with water
        # lat = -82.8642,
        # lon = 42.33
        data_load_type = 'optimal', #   'api', 'local' or 'optimal'(takes local if exist)
        veg_idx_band = 'NDVI', # Choose vegetation index band. For our datasets, either 'NDVI' or 'EVI'.
    )
    config.update({
        NUM_FIDELITIES: 2,
        NUM_ITER: 200,
        KERNELS: MATERN32,
        KERNEL_COMBINATION: SUM,
        MATERN32_LENGTHSCALE: 130,
        MATERN32_VARIANCE: 1.0,
        RBF_LENGTHSCALE: 0.08,
        RBF_VARIANCE: 20.0,
        WHITE_VARIANCE: 20.0,
        PERIODIC_LENGTHSCALE: 0.08,
        PERIODIC_PERIOD: 1.0,
        PERIODIC_VARIANCE: 20.0,
        MODEL_NOISE_VARIANCE: 1e-13,
        OPTIMIZATION_RESTARTS: 0,
        OPTIMIZER_UPDATE_INTERVAL: 1,
        LOW_FIDELITY_COST: 1.0,
        HIGH_FIDELITY_COST: 2.0,
        MATPLOTLIB_BACKEND: 'Agg' if use_wandb else '', # set this to 'Agg' or another non-gui backend for wandb runs
    })
    global LOGGER
    LOGGER = CustomLogger(use_wandb=use_wandb, config=config)
    center_point = np.array([[LOGGER.config["lat"], LOGGER.config["lon"]]])
    dataloader_high_fidelity = NormalDataLoad(LOGGER.config["source"], center_point, LOGGER.config["num_points"], LOGGER.config["scale"], LOGGER.config["veg_idx_band"], LOGGER.config["data_load_type"])
    # basic_gp(dataloader_high_fidelity, LOGGER.config["num_points"])

    dataloader2 = NormalDataLoad(LOGGER.config["source_low"], center_point, LOGGER.config["num_points"], LOGGER.config["scale_low"], LOGGER.config["veg_idx_band"], LOGGER.config["data_load_type"])

    mf_gp(dataloader_high_fidelity, dataloader2, LOGGER.config["num_points"])
    LOGGER.stop_run()
    input()
    return

if __name__ == '__main__':
    # take in parameters from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    # Define the kernel parameter search
    sweep_config = {
        'name': 'Kernel param search: random, matern only',
        'method': 'random',
        'metric': {
            'name': 'L2',
            'goal': 'minimize',
        },
    }
    parameters_dict = {
        # KERNELS: {
        #     'values': [RBF, MATERN32, PERIODIC, WHITE, EXPONENTIAL, RBF+SEPARATOR+PERIODIC+SEPARATOR+WHITE, MATERN32+SEPARATOR+PERIODIC+SEPARATOR+WHITE],
        # },
        # KERNEL_COMBINATION: {
        #     'values': [PRODUCT, SUM, ''],
        # },
        RBF_LENGTHSCALE: {
            'distribution': 'log_uniform_values', # or 'uniform',
            'min': 10**(-5),
            'max': 10**(1),
            # 'values': [i for i in np.logspace(-5, 5, 20, base=10)],
        },
        RBF_VARIANCE: {
            'distribution': 'log_uniform_values',
            'min': 10**(-1),
            'max': 10**2,
            # 'values': np.linspace(0.0, 50.0, 10),
        },
        # MATERN32_LENGTHSCALE: {
        #     'distribution': 'log_uniform_values', # or 'uniform',
        #     'min': 10**(-3),
        #     'max': 10**(3),
        #     # 'values': [i for i in np.logspace(-5, 5, 20, base=10)],
        # },
        # MATERN32_VARIANCE: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-1),
        #     'max': 10**2,
        #     # 'values': np.linspace(0.0, 50.0, 10),
        # },
        # 'Matern_nu': {
        #     'distribution': 'categorical',
        #     'values': [0.5, 1.5, 2.5] #, np.inf],
        # },
        # WHITE_VARIANCE: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-2),
        #     'max': 10**2,
        # },
        # PERIODIC_LENGTHSCALE: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-3),
        #     'max': 10**(3),
        # },
        # PERIODIC_PERIOD: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-2),
        #     'max': 10**2,
        # },
        # PERIODIC_VARIANCE: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-2),
        #     'max': 10**2,
        # },
        # MODEL_NOISE_VARIANCE: {
        #     'distribution': 'log_uniform_values',
        #     'min': 10**(-10),
        #     'max': 10**1,
        # },
        # OPTIMIZATION_RESTARTS: {
        #     'distribution': 'categorical',
        #     'values': [1, 5, 10],
        # },
    }
    sweep_config['parameters'] = parameters_dict
    if sweep_config and args.sweep:
        sweep_id = wandb.sweep(sweep_config, project="sensor-placement", entity="camb-mphil")
        # sweep_id = wandb.sweep(sweep_config, project="test-sensor-placement", entity="sepand")
        print('sweep initialized')
        wandb.agent(sweep_id, main, count=20)
    else:
        main(use_wandb=False)

