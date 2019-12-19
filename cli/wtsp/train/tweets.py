"""Tweets training module."""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from wtsp.core.sklearn.transformers import DataFrameFilter, GeoPandasTransformer, GeoPointTransformer
from wtsp.utils import parse_kwargs
from typing import Dict

from wtsp.view.view import plot_nearest_neighbors, plot_points

DEFAULT_COLUMNS = ["id",
                   "tweet",
                   "country",
                   "country_code",
                   "place_name",
                   "place_full_name",
                   "location_geometry",
                   "place_geometry",
                   "created_timestamp"]


class TweetsTrainer:
    """Tweets Trainer.

    Orchestrator class to train tweets
    related models.
    """
    def __init__(self, model: str, filters: str, params: str, output_dir: str):
        self.model = model
        self.filters = parse_kwargs(filters) if filters else None
        self.params = parse_kwargs(params) if params else None
        self.output_dir = output_dir

    def train(self, input_data: str) -> str:
        if self.model == "nearest-neighbors":
            logging.debug(f"About to train the nearest-neighbors on {input_data}")
            trainer = GeoTweetsNearestNeighbors(self.filters,
                                                self.params,
                                                self.output_dir)
            return trainer.train(input_data)


class GeoTweetsNearestNeighbors:
    """Geo tweets Nearest Neighbors.

    It will train a nearest neighbors model
    over geo tagged tweets and will create a
    plot with the sorted distances in the
    output directory and a scatter plot.

    It will automatically filter out records
    with no valid location geometries.
    """
    def __init__(self, filters: Dict[str, object],
                 params: Dict[str, object],
                 output_dir: str):
        self.filters = filters
        self.params = params
        self.output_dir = output_dir

    def train(self, input_data: str) -> str:
        # Transform the data
        logging.debug(f"Transforming input data to get the geo-points")
        points = self.__transform_data(input_data)
        # Train the nearest neighbors
        logging.debug(f"Training the nearest neighbors model...")
        distances = self.__train_nearest_neighbors(points)

        # Plot the results
        logging.debug(f"Plotting the results...")
        filter_key = next(iter(self.filters))
        filter_value = self.filters[filter_key]
        output_dir = f"{self.output_dir}/tweets/{filter_key}={filter_value}"
        os.makedirs(output_dir, exist_ok=True)

        # nn plot
        n_neighbors = self.params["n_neighbors"]
        title = f"Nearest Neighbors for Geo-tagged tweets in {filter_value}"
        y_label = f"{n_neighbors}th Nearest Neighbor Distance"
        x_label = f"Points sorted according to the distance of the {n_neighbors}th Nearest Neighbor"
        save_path = f"{output_dir}/nearest_neighbors.png"
        plot_nearest_neighbors(distances, title, x_label, y_label, save_path)

        # scatter plot
        save_path = f"{output_dir}/scatter_plot.png"
        plot_points(points, f"{filter_value} - Geo-tagged tweets scatter plot", save_path)
        return f"Result generated successfully at: {output_dir}"

    def __transform_data(self, input_data: str) -> pd.DataFrame:
        data = pd.read_parquet(input_data)
        geometry_field = self.params["location_column"]
        pipeline = Pipeline(
            [
                ("filter", DataFrameFilter(self.filters)),
                ("to_geopandas", GeoPandasTransformer(geometry_field, DEFAULT_COLUMNS)),
                ("to_geopoint", GeoPointTransformer(geometry_field, only_points=True))
            ]
        )

        return pipeline.transform(data)

    def __train_nearest_neighbors(self, points):
        n_neighbors = self.params["n_neighbors"]
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs = neigh.fit(points)
        distances, indices = nbrs.kneighbors(points)
        distances = np.sort(distances, axis=0)
        return distances[:, 1]
