"""Scikit learn transformers of data."""

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict
import pandas as pd


class CountTransformer(BaseEstimator, TransformerMixin):
    """Count Transformer.

    Transforms the provided data in such way
    that can be used to count values
    by specific field.
    """

    def __init__(self, groupby: str, count_col: str, min_count: int = 5000):
        """Creates a count transformer."""
        self.min_count = min_count
        self.groupby = groupby
        self.count_col = count_col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """Transform for count.

        Transforms the provided data set to count
        by the provided column.
        """
        data = X.copy()
        data = data.groupby(self.groupby).agg({self.count_col: ["count"]})
        data = flat_columns(data)
        data = data[data[f"{self.count_col}count"] >= self.min_count]
        data.reset_index(inplace=True)
        data.sort_values(f"{self.count_col}count", ascending=False, inplace=True)
        return data


class DataFrameFilter(BaseEstimator, TransformerMixin):
    """Data Frame Filter.

    This transformer will simply filter the
    input data frame according to the provided
    filed values.
    """
    def __init__(self, field_values: Dict[str, str]):
        self.field_values = field_values

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X: pd.DataFrame, y=None):
        data = X.copy()
        for field, value in self.field_values.items():
            data = data[data[field].isin([value])]
        return data


class MultiValueColumnExpander(BaseEstimator, TransformerMixin):
    """Multi value column expander.

    Use this when you have a data frame with a multi value
    column represented as a string separated by a specific
    character. This transformer will take the data frame
    and explode the records by each individual value in
    the multi value column.
    """
    def __init__(self, expand_column, value_split_char=";"):
        self.expand_column = expand_column
        self.value_split_char = value_split_char

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X: pd.DataFrame, y=None):
        data = X.copy()
        split_fn = lambda x: x.split(self.value_split_char)
        data[self.expand_column] = data[self.expand_column].apply(split_fn)
        return data.explode(self.expand_column)


def flat_columns(df: pd.DataFrame, colnames :Dict[str, str] = {}):
    """Flat columns.

    Given a multi-indexed pandas DataFrame,
    it will flat the column names given the
    colnames.
    """
    df = df.copy()
    df.columns = [''.join(col).strip() for col in df.columns]
    if colnames:
        return df.rename(columns=colnames)
    return df
