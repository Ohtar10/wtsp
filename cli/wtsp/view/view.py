"""Contains presentation logic."""
import itertools
import json
import re
from itertools import cycle
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import folium
from shapely.geometry import mapping


def plot_counts(data: pd.DataFrame, title,  x_label: str, save_path: str):
    """Plot counts.

    Creates a bar chart with the provided
    data and saves it into the given path.
    """
    min_val = 0
    max_val = max(data[data.columns[1]].values)
    # calculate the base 10 step size based on the maximum value
    step = __get_plot_step(max_val)
    new_index = data.columns[0]
    ax = data.set_index(new_index).plot(kind="bar",
                                        figsize=(10, 8),
                                        fontsize=12,
                                        grid=True,
                                        yticks=np.arange(min_val, max_val + step, step))

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(save_path, bbox_inches="tight")


def __get_plot_step(max_val: int):
    exp = int(math.log(max_val, 10))
    for e in range(exp, 0, -1):
        step = 10 ** e
        if (max_val - step) >= step:
            return step


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


def plot_cnn_history(cnn,
                     save_path,
                     acc='acc',
                     val_acc='val_acc',
                     loss='loss',
                     val_loss='val_loss'):
    """Plot a trained cnn history."""
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('Accuracy in training Vs validation', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(cnn.history[acc], 'r')
    plt.plot(cnn.history[val_acc], 'b')
    plt.legend(['training', 'validation'], fontsize=14)

    plt.subplot(1, 2, 2)
    plt.title('Loss in training Vs validation', fontsize=16)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(cnn.history[loss], 'r')
    plt.plot(cnn.history[val_loss], 'b')
    plt.legend(['training', 'validation'], fontsize=14)

    plt.savefig(save_path, bbox_inches="tight")


def plot_classification_report(cr,
                               class_labels,
                               save_path,
                               title="Classification Report",
                               cmap="RdYlBu"):
    """Plot a sklearn classification report."""
    cr = cr.replace("\n\n", "\n")
    cr = cr.replace(" / ", "/")
    lines = cr.split("\n")

    classes, plot_mat, support = [], [], []
    for line in lines[1:]:
        t = re.findall(r'(\s?\w+\s\w+\s?|\s?\d+\.?\d*\s?)', line)
        t = [i.strip() for i in t]
        if len(t) < 2:
            continue

        if t[0].isnumeric():
            cl_id = int(t[0])
            classes.append(class_labels[cl_id])
        else:
            classes.append(t[0])

        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        plot_mat.append(v)

    plot_mat = np.array(plot_mat)
    x_tick_labels = ["Precision", "Recall", "F1-score"]
    y_tick_labels = [""] + [f"{classes[idx]} ({sup})" for idx, sup in enumerate(support)] + [""]

    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=16)

    plt.imshow(plot_mat, interpolation="nearest", cmap=cmap, aspect="auto")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)

    plt.xticks(np.arange(3), x_tick_labels, rotation=45, fontsize=14)
    start = -1
    stop = len(classes) + 1
    y_ticks = np.arange(start, stop, 1)
    plt.yticks(y_ticks, y_tick_labels, fontsize=14)

    upper_threshold = plot_mat.min() + (plot_mat.max() - plot_mat.min()) / 10 * 8
    lower_threshold = plot_mat.min() + (plot_mat.max() - plot_mat.min()) / 10 * 2

    for i, j in itertools.product(range(plot_mat.shape[0]), range(plot_mat.shape[1])):
        text = f"{plot_mat[i, j]:.2f}"
        color = "white" if plot_mat[i, j] > upper_threshold or plot_mat[i, j] < lower_threshold else "black"
        ax = plt.text(j, i, text, fontsize=14,
                      horizontalalignment="center",
                      color=color)

    plt.ylabel("Classes", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)

    plt.savefig(save_path, bbox_inches="tight")


def plot_clusters_on_map(clusters,
                         save_path,
                         print_classes=False,
                         center=(34, -118),
                         zoom=10,
                         tiles="OpenStreetMap",
                         score_threshold=0.1):
    unique_clusters = np.unique(clusters.cluster.values)
    n_clusters = unique_clusters.shape[0]
    colormap = cycle(plt.cm.rainbow(np.linspace(0, 1, n_clusters)))
    colors = [matplotlib.colors.to_hex(next(colormap)) for _ in range(len(unique_clusters))]
    random.shuffle(colors)
    colors = {k: c for k, c in zip(unique_clusters, colors)}

    m = folium.Map(location=center,
                   zoom_start=zoom,
                   tiles=tiles)

    def style_function(row):
        color = colors[row.cluster]
        return lambda x: {'fillColor': color,
                          'fillOpacity': 0.6,
                          'weight': 1}

    def highlight_function(row):
        color = colors[row.cluster]
        return lambda x: {'fillColor': color,
                          'fillOpacity': 0.8,
                          'weight': 1}

    def create_tooltip(row):
        if print_classes:
            cluster_id = row.cluster
            size = row['size']
            classes = row.predictions
            classes = [f"{cl} - {score * 100:.2f}%" for cl, score in classes if score >= score_threshold]
            classes = '<br>'.join(classes)
            return folium.map.Tooltip(text=f"Cluster id: {cluster_id}<br>Size: {size}<br>{classes}",
                                      style="color: DodgerBlue")
        else:
            return folium.map.Tooltip(text=f"Cluster id: {row.cluster}<br>Size: {row['size']}",
                                      style="color: DodgerBlue")

    def create_polygon(row):
        geojson = json.dumps(mapping(row['polygon']))
        tooltip = create_tooltip(row)

        folium.GeoJson(geojson,
                       style_function=style_function(row),
                       highlight_function=highlight_function(row),
                       tooltip=tooltip).add_to(m)

    clusters.apply(create_polygon, axis=1)

    m.save(save_path)
