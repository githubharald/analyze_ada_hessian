import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_function(f, window):
    "plot 2d function by executing the PyTorch function for each grid point"
    num_vals = 50
    x_vals = np.linspace(window[0], window[1], num_vals)
    y_vals = np.linspace(window[2], window[3], num_vals)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.empty([num_vals, num_vals], dtype=np.float64)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            v = torch.tensor((x, y), dtype=torch.float64)
            Z[j, i] = f(v)

    plt.pcolormesh(X, Y, Z, cmap='rainbow', shading='auto')


def plot_path(path):
    "plot steps taken by optimizer"
    plt.plot(path[:, 0], path[:, 1], 'r-')
    plt.plot(path[:, 0], path[:, 1], 'r.')
    plt.plot(path[0, 0], path[0, 1], 'k^', label='start')
    plt.plot(path[-1, 0], path[-1, 1], 'k*', label='last')


def show_plot():
    "show plot with colorbar and legend"
    plt.colorbar()
    plt.legend()
    plt.show()
