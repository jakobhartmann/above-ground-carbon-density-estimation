import matplotlib.pyplot as plt
import numpy as np

# Plot 2D heatmap of each of a and b functions, side by side
def heatmap_comparison(a, b, num_points, emukit_model, output_filename='bayes_opt_vis.png'):
    fig, axes = plt.subplots(1, 2)

    plot_2D_vis(a, num_points, emukit_model, axes, 0)
    plot_2D_vis(b, num_points, emukit_model, axes, 1)

    plt.savefig(output_filename)
    plt.show()


def plot_2D_vis(results, num_points, emukit_model, axes, axis_idx):
    min_val = np.min(results)
    max_val = np.max(results)

    # Estimated function
    axes[axis_idx].imshow(results.reshape(num_points, num_points),
                          vmin=min_val, vmax=max_val)
    axes[axis_idx].scatter((emukit_model.X[:, 1] + 1) * num_points / 2, (emukit_model.X[:, 0] + 1) * num_points / 2,
                           c=emukit_model.Y[:, 0], label='observations', vmin=min_val, vmax=max_val, s=40, edgecolor='black')
    axes[axis_idx].set_xlabel('$x$ (Latitude)')
    axes[axis_idx].set_ylabel('$y$ (Longitude)')
