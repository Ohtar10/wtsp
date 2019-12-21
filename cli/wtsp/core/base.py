"""Base module.

Contains the base classes and functionalities for the
rest of the project.
"""
from pathlib import Path
import os
import pandas as pd

from wtsp.exceptions import InvalidArgumentException, DataLoadException

DEFAULT_TWEETS_COLUMNS = ["id",
                          "tweet",
                          "country",
                          "country_code",
                          "place_name",
                          "place_full_name",
                          "location_geometry",
                          "place_geometry",
                          "created_timestamp"]


DEFAULT_WORK_DIR = f"{str(Path.home())}/wtsp"


class DataLoader:

    def __init__(self, engine="pyarrow"):
        self.engine = engine

    def load_data(self, input_data):
        if not os.path.exists(input_data):
            raise InvalidArgumentException("The provided input data path is not valid")
        try:
            return pd.read_parquet(input_data, engine=self.engine)
        except Exception as e:
            raise DataLoadException("The provided input data is not a valid parquet file", e)
