"""Tweets training module."""

import os.path
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
from wtsp.core.base import Process
from wtsp.exceptions import InvalidArgumentException
from wtsp.utils import parse_geometry
from sklearn.neighbors import NearestNeighbors


class GeoTweetsNearestNeighbors(Process):

    """GeoTweetsNearestNeighbors.

    This class trains a Nearest Neighbors model over
    the geo-tagged tweets found in the input path.
    """

    SUCCESS_MESSAGE = "Geo tweets nearest neighbors executed successfully. Use the report command to see the results."

    def __init__(self, work_dir, debug=False, **kwargs):
        """Create GeoTweetNearestNeighbor object."""
        super().__init__(work_dir, debug)
        params = ["n_neighbors", "country", "place"]
        if not all([arg in params for arg in kwargs]):
            raise InvalidArgumentException("For nearest neighbors n_neighbors, country and place are required.")

        for k in kwargs.keys():
            if k in params:
                self.__setattr__(k, kwargs[k])

    def run(self, input_file):
        self.__validate_input(input_file)
        df = self.__load_data(input_file)
        points = self.__extract_points(df)
        distances = self.__train_model(points)
        self.__save_results(df, distances)
        return GeoTweetsNearestNeighbors.SUCCESS_MESSAGE

    def __validate_input(self, input_file):
        if not input_file:
            raise InvalidArgumentException("Input file is mandatory")
        if not os.path.exists(input_file):
            raise InvalidArgumentException("Input file does not exist")
        if not self.n_neighbors:
            raise InvalidArgumentException("n_neighbors is mandatory")
        if not self.country:
            raise InvalidArgumentException("country is mandatory")
        if not self.place:
            raise InvalidArgumentException("place is mandatory")
        try:
            int(self.n_neighbors)
        except(ValueError, TypeError):
            raise InvalidArgumentException("n_neighbors should be an integer")

    def __load_data(self, input_file):
        df = pd.read_parquet(input_file)
        df = df[(df.country == self.country) & (df.place_name == self.place)]
        df = df[(df.location_geometry.notnull()) | (df.place_geometry.notnull())]
        df.location_geometry = df.location_geometry.apply(parse_geometry)
        df.place_geometry = df.place_geometry.apply(parse_geometry)
        return gpd.GeoDataFrame(df, geometry="location_geometry")

    def __extract_points(self, df):
        points = df[df.location_geometry.notnull()].location_geometry.apply(lambda p: [p.x, p.y])
        return np.array(points.values.tolist())

    def __train_model(self, points):
        neigh = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs = neigh.fit(points)
        distances, indices = nbrs.kneighbors(points)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        return distances

    def __save_results(self, df :pd.DataFrame, distances):
        output = self.__get_workdir()
        os.makedirs(output, exist_ok=True)
        #df.to_parquet(f"{output}/tweets.parquet", engine="pyarrow", compression="snappy")
        df.to_csv(f"{output}/tweets.csv")
        with open(f"{output}/nnm_distances.sav", "wb") as file:
            pickle.dump(distances, file)

    def __get_workdir(self):
        return f"{self.work_dir}/twitter/nnm"
