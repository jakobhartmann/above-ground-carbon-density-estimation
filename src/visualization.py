import matplotlib.pyplot as plt
import numpy as np
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

output_folder = 'results/'


# Plot 2D heatmap of each of a and b functions, side by side
def heatmap_comparison(a, b, num_points, emukit_model):
    fig, axes = plt.subplots(1, 2)

    min_val = min([np.min(a), np.min(b)])
    max_val = max([np.max(a), np.max(b)])

    plot_2D_vis(a, num_points, emukit_model, axes, 0, min_val, max_val, 'Bayesian optimization')
    plot = plot_2D_vis(b, num_points, emukit_model, axes, 1, min_val, max_val, 'Ground truth')

    add_colorbar(fig, axes[1], plot, label="NDVI")

    return fig


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


def plot_variance(variance, num_points, emukit_model):
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

