"""Contains presentation logic."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def plot_counts(data: pd.DataFrame, title,  x_label: str, save_path: str):
    """Plot counts.

    Creates a bar chart with the provided
    data and saves it into the given path.
    """
    min_val = 0
    max_val = max(data[data.columns[1]].values)
    # calculate the base 10 step size based on the maximum value
    step = 10 ** (int(math.log(max_val, 10)))
    new_index = data.columns[0]
    ax = data.set_index(new_index).plot(kind="bar",
                                        figsize=(10, 8),
                                        fontsize=12,
                                        grid=True,
                                        yticks=np.arange(min_val, max_val + 1, step))

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(save_path, bbox_inches="tight")


def plot_points(data, title, save_path):
    """Plot points

    Creates a simple scatter plot
    with the data provided.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.1)
    plt.title(title, fontsize=16)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.savefig(save_path, bbox_inches="tight")


def plot_nearest_neighbors(data: np.ndarray,
                           title: str,
                           x_label: str,
                           y_label: str,
                           save_path: str):
    """Plot Nearest Neighbors distances.

    Creates a plot with the provided nearest
    neighbors distances.
    """
    plt.figure(figsize=(10, 8))
    plt.grid()
    axes = plt.gca()
    # TODO check how to parameterize them better
    axes.set_ylim([-0.005, 0.02])
    plt.yticks(np.arange(-0.005, 0.02, 0.001))
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.plot(data)
    plt.savefig(save_path, bbox_inches="tight")

