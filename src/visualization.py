import matplotlib
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

output_folder = 'results/'


# Plot 2D heatmap of each of a and b functions, side by side
def heatmap_comparison(a, b, num_points, emukit_model, backend=''):
    if backend != '':
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)

    min_val = min([np.min(a), np.min(b)])
    max_val = max([np.max(a), np.max(b)])

    plot_2D_vis(a, num_points, emukit_model, axes, 0, min_val, max_val, 'Bayesian optimization')
    plot = plot_2D_vis(b, num_points, emukit_model, axes, 1, min_val, max_val, 'Ground truth')

    add_colorbar(fig, axes[1], plot, label="NDVI")

    return fig

# Plot 2D heatmap of each of a and b functions, side by side
def heatmap_comparison_mf(a1, a2, b1, b2, num_points, emukit_model):
    if backend != '':
        matplotlib.use(backend)
    import matplotlib.pyplot as plt

    min_val = min([np.min(a1), np.min(b1), np.min(a2), np.min(b2)])
    max_val = max([np.max(a1), np.max(b1), np.max(a2), np.max(b2)])

    fig1, axes1 = plt.subplots(1, 2)
    plot_2D_vis_mf(a1, num_points, emukit_model, axes1, 0, min_val, max_val, 'Bayesian optimization', 0)
    plot = plot_2D_vis_mf(b1, num_points, emukit_model, axes1, 1, min_val, max_val, 'Ground truth', 0)

    add_colorbar(fig1, axes1[1], plot, label="NDVI")

    fig2, axes2 = plt.subplots(1, 2)
    plot_2D_vis_mf(a2, num_points, emukit_model, axes2, 0, min_val, max_val, 'Bayesian optimization', 1)
    plot = plot_2D_vis_mf(b2, num_points, emukit_model, axes2, 1, min_val, max_val, 'Ground truth', 1)

    add_colorbar(fig2, axes2[1], plot, label="NDVI")

    return fig1, fig2


def plot_2D_vis(results, num_points, emukit_model, axes, axis_idx, min_val, max_val, title):
    # Estimated function
    plot = axes[axis_idx].imshow(results.reshape(num_points, num_points),
                          vmin=min_val, vmax=max_val)
    axes[axis_idx].scatter((emukit_model.X[:, 1] + 1) * num_points / 2, (emukit_model.X[:, 0] + 1) * num_points / 2,
                           c=emukit_model.Y[:, 0], label='observations', vmin=min_val, vmax=max_val, s=40,
                           edgecolor='black')
    axes[axis_idx].set_xlabel('$x$ (Latitude)')
    axes[axis_idx].set_ylabel('$y$ (Longitude)')
    axes[axis_idx].set_title(title)
    return plot

def plot_2D_vis_mf(results, num_points, emukit_model, axes, axis_idx, min_val, max_val, title, mf_choose):
    # Estimated function
    plot = axes[axis_idx].imshow(results.reshape(num_points, num_points),
                          vmin=min_val, vmax=max_val)
    # get points from wanted fidelity
    points_indices = emukit_model.X[:,2]==mf_choose

    plot_points_X = emukit_model.X[points_indices]
    plot_points_Y = emukit_model.Y[points_indices]

    axes[axis_idx].scatter((plot_points_X[:, 1] + 1) * num_points / 2, (plot_points_X[:, 0] + 1) * num_points / 2,
                           c=plot_points_Y[:, 0], label='observations', vmin=min_val, vmax=max_val, s=40,
                           edgecolor='black')
    axes[axis_idx].set_xlabel('$x$ (Latitude)')
    axes[axis_idx].set_ylabel('$y$ (Longitude)')
    axes[axis_idx].set_title(title)
    return plot


def plot_variance(variance, num_points, emukit_model, backend=''):
    if backend != '':
        matplotlib.use(backend)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot = ax.imshow(variance.reshape(num_points, num_points))
    ax.set_xlabel('$x$ (Latitude)')
    ax.set_ylabel('$y$ (Longitude)')
    ax.scatter((emukit_model.X[:, 1] + 1) * num_points / 2, (emukit_model.X[:, 0] + 1) * num_points / 2,
               c='red', label='observations', s=10)
    add_colorbar(fig, ax, plot, label="variance")

    return fig

def add_colorbar(fig, ax, plot, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation="vertical", **kwargs)

