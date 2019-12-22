"""Base module.

Contains the base classes and functionalities for the
rest of the project.
"""
from pathlib import Path
import os
from typing import Optional, Dict

import pandas as pd

from wtsp.exceptions import InvalidArgumentException, DataLoadException
from wtsp.utils import parse_kwargs

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
    """Data loader.

    Contains general functionality for
    data loading in the project.
    """
    def __init__(self, engine="pyarrow"):
        self.engine = engine

    def load_data(self, input_data):
        if not input_data or not os.path.exists(input_data):
            raise InvalidArgumentException("The provided input data path is not valid")
        try:
            return pd.read_parquet(input_data, engine=self.engine)
        except Exception as e:
            raise DataLoadException("The provided input data is not a valid parquet file", e)


class Trainer:
    """Trainer.

    Parent class of all trainers
    """
    def train(self, input_data):
        # This is expected to be overwritten
        return "Not Implemented"


class Filterable:
    """Filterable.

    Contains general functionality for
    filterable structures.
    """
    def __init__(self, filters: Optional[str], can_be_none: bool = False):
        try:
            if can_be_none:
                self.filters: Optional[Dict[str, object]] = parse_kwargs(filters) if filters else None
            else:
                self.filters = parse_kwargs(filters)
        except (ValueError, AttributeError) as e:
            raise InvalidArgumentException("Filter value is invalid. use: key=value", e)


class Parametrizable:
    """Parametrizable.

    Contains general functionality for
    parametrizable structures.
    """
    def __init__(self, params: str, can_be_none: bool = False):
        try:
            if can_be_none:
                self.params = parse_kwargs(params) if params else None
            else:
                self.params = parse_kwargs(params)
        except (ValueError, AttributeError) as e:
            raise InvalidArgumentException("Params value is invalid. use: key=value", e)