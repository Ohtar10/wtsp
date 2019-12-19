"""Scikit learn transformers of data."""
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import shutil
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from shapely import wkt
from sklearn.base import BaseEstimator, TransformerMixin
from wtsp.utils import ensure_nltk_resource_is_available


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
    def __init__(self, field_values: Dict[str, object]):
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


class GeoPandasTransformer(BaseEstimator, TransformerMixin):
    """Geo Pandas Transformer.

    Given a normal pandas data frame with a geometry column
    it will convert it into a geo pandas data frame.
    """
    def __init__(self, geometry_field, columns):
        self.geometry_field = geometry_field
        self.columns = columns

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X, y=None):
        data = X[self.columns]
        data = data[data[self.geometry_field].notnull()]
        data[self.geometry_field] = data[self.geometry_field].apply(parse_geometry)
        return gpd.GeoDataFrame(data, geometry=self.geometry_field)


class GeoPointTransformer(BaseEstimator, TransformerMixin):
    """Geo Point transformer.

    Given a geo pandas data frame and
    a location column, it will get the points
    as x, y coordinates and either add them as an additional
    column in the data frame or return only the points as
    a numpy array.
    """
    def __init__(self, location_column, only_points=False):
        self.location_column = location_column
        self.only_points = only_points

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X, y=None):
        data = X[X[self.location_column].notnull()]
        points = data[self.location_column].apply(lambda p: [p.x, p.y])
        if self.only_points:
            points = np.array(points.values.tolist())
            return points
        else:
            data["location_coordinates"] = points
            return data


class DocumentTagger(BaseEstimator, TransformerMixin):
    """Document Tagger.

    It will add a new column to the given
    data frame containing the tagged document
    of the corpus column.
    """
    def __init__(self, tags_column, corpus_column):
        self.tags_column = tags_column
        self.corpus_column = corpus_column

    def fit(self, X, y=None):
        return self  # Do nothing

    def transform(self, X, y=None):
        ensure_nltk_resource_is_available("punkt")
        data = X.copy()
        tag_document_fn = create_tagged_document_fn(self.tags_column, self.corpus_column)
        return data.apply(tag_document_fn, axis=1)


class Doc2VecWrapper(BaseEstimator, TransformerMixin):
    """Doc2Vec model wrapper.

    Gensimś Doc2Vec Model wrapper which will
    take the pandas data frame as input and train
    a document embedding model on it.
    """
    def __init__(self, document_column,
                 tag_doc_column="tagged_document",
                 lr=0.01,
                 epochs=10,
                 vec_size=100,
                 alpha=0.1,
                 min_alpha=0.0001,
                 min_count=1,
                 dm=0):
        self.document_column = document_column
        self.tag_doc_column = tag_doc_column
        self.lr = lr
        self.epochs = epochs
        self.vec_size = vec_size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.dm = dm
        self.d2v_model = None

    def fit(self, X, y=None):
        tagged_documents = X[self.tag_doc_column]

        d2v_model = Doc2Vec(vector_size=self.vec_size,
                            alpha=self.alpha,
                            min_alpha=self.min_alpha,
                            min_count=self.min_count,
                            dm=self.dm)

        d2v_model.build_vocab(tagged_documents)

        for epoch in range(self.epochs):
            d2v_model.train(tagged_documents,
                            total_examples=d2v_model.corpus_count,
                            epochs=d2v_model.epochs)
            d2v_model.alpha -= self.lr
            d2v_model.min_alpha = d2v_model.alpha

        self.d2v_model = d2v_model

        return self

    def transform(self, X, y=None):
        data = X.copy()
        tagged_docs = data[self.document_column].apply(word_tokenize)
        embeddings = tagged_docs.apply(self.d2v_model.infer_vector)
        data["d2v_embedding"] = embeddings
        return data

    def save_model(self, save_path):
        # always overwrite
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(f"{save_path}.trainables.syn1neg.npy"):
            os.remove(f"{save_path}.trainables.syn1neg.npy")
        if os.path.exists(f"{save_path}.wv.vectors.npy"):
            os.remove(f"{save_path}.wv.vectors.npy")
        self.d2v_model.save(save_path)


def flat_columns(df: pd.DataFrame, colnames: Dict[str, str] = None):
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


def parse_geometry(geom):
    """Parse Geometry.

    If a valid WKT geometry is provided
    it will convert it ino a shapely geometry.
    Otherwise None is returned.
    """
    if geom:
        return wkt.loads(geom)
    else:
        return None


def create_tagged_document_fn(tags_column, corpus_column):
    """Create tagged documents.

    Designed to be applied to a pandas
    data frame. Given a row, it will create
    a gensim TaggedDocument with the text
    and tags.
    """
    def create_tagged_document(row):
        categories = row[tags_column]
        document = row[corpus_column]
        tagged_document = TaggedDocument(words=word_tokenize(document), tags=categories.split(";"))
        index = [tags_column, corpus_column, "tagged_document"]
        return pd.Series([categories, document, tagged_document], index=index)

    return create_tagged_document
