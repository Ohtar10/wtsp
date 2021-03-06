"""Tweets training module."""

import logging
import os
from typing import Dict

import modin.pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from wtsp.core.base import DEFAULT_TWEETS_COLUMNS, DataLoader, Filterable, Parametrizable
from wtsp.core.sklearn.transformers.clustering import GeoPandasTransformer, GeoPointTransformer
from wtsp.core.sklearn.transformers.generic import DataFrameFilter
from wtsp.exceptions import ModelTrainingException, InvalidArgumentException
from wtsp.view.view import plot_nearest_neighbors, plot_points


class TweetsTrainer(Filterable, Parametrizable):
    """Tweets Trainer.

    Orchestrator class to train tweets
    related models.
    """
    def __init__(self, model: str, filters: str, params: str, output_dir: str):
        Filterable.__init__(self, filters)
        Parametrizable.__init__(self, params)
        self.model = model
        self.output_dir = output_dir

    def train(self, input_data: str) -> str:
        if self.model == "nearest-neighbors":
            logging.info(f"Training nearest-neighbors on {input_data}")
            trainer = GeoTweetsNearestNeighbors(self.filters,
                                                self.params,
                                                self.output_dir)
            return trainer.train(input_data)
        raise InvalidArgumentException(f"The model requested is not recognized: {self.model}")


class GeoTweetsNearestNeighbors(DataLoader):
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
        super().__init__()
        self.filters = filters
        self.params = params
        self.output_dir = output_dir

    def train(self, input_data: str) -> str:
        # Transform the data
        logging.info(f"Transforming input data to get the geo-points")
        points = self.__transform_data(input_data)
        # Train the nearest neighbors
        logging.info(f"Training the nearest neighbors model...")
        try:
            distances = self.__train_nearest_neighbors(points)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise ModelTrainingException("There is a problem processing the data, see the error message", e)

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
        data = self.load_data(input_data, DEFAULT_TWEETS_COLUMNS)
        geometry_field = self.params["location_column"]
        pipeline = Pipeline(
            [
                ("filter", DataFrameFilter(self.filters)),
                ("to_geopandas", GeoPandasTransformer(geometry_field, DEFAULT_TWEETS_COLUMNS)),
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
