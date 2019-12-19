"""Contains presentation logic."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_counts(data: pd.DataFrame, title,  x_label: str, save_path: str):
    """Plot counts.

    Creates a bar chart with the provided
    data and saves it into the given path.
    """
    min_val = 0
    max_val = max(data[data.columns[1]].values)
    new_index = data.columns[0]
    ax = data.set_index(new_index).plot(kind="bar",
                     figsize=(10, 8),
                     fontsize=12,
                     grid=True,
                     yticks=np.arange(min_val, max_val + 1, 1000.0))

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.legend(prop={"size": 14})
    plt.savefig(save_path, bbox_inches="tight")
