import logging
import pickle
from operator import itemgetter

import geopandas as gpd
import numpy as np
from gensim.models import Doc2Vec
from keras.engine.saving import model_from_yaml
from modin import pandas as pd
from scipy.spatial.qhull import ConvexHull, QhullError
from shapely import wkt
from shapely.geometry import Polygon, Point
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import OPTICS

from wtsp.core.sklearn.transformers.documents import concatenate_text, DocumentTokenizer
from wtsp.core.sklearn.transformers.generic import flat_columns
from wtsp.utils import ensure_nltk_resource_is_available


class GeoPandasTransformer(BaseEstimator, TransformerMixin):
    """Geo Pandas Transformer.

    Given a normal pandas data frame with a geometry column
    it will convert it into a geo pandas data frame.
    """

    def __init__(self, geometry_field, columns, as_geopandas=False):
        self.geometry_field = geometry_field
        self.columns = columns
        self.as_geopandas = as_geopandas

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X: pd.DataFrame, y=None):
        data = X[self.columns]
        data = data[data[self.geometry_field].notnull()]
        data[self.geometry_field] = data[self.geometry_field].apply(parse_geometry)
        if self.as_geopandas:
            return gpd.GeoDataFrame(data, geometry=self.geometry_field)
        return data


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


class ClusterAggregator(BaseEstimator, TransformerMixin):
    """Cluster Aggergator.

    This transformer will aggregate all the
    common clusters in terms of its corpus
    and points that composes it.
    """

    def __init__(self, columns,
                 location_column,
                 agg_colnames,
                 clusters=None,
                 n_neighbors=10,
                 eps=0.004,
                 filter_noise=True):
        self.columns = columns
        self.location_column = location_column
        self.agg_colnames = agg_colnames
        self.filter_noise = filter_noise
        self.clusters = clusters
        self.n_neighbors = n_neighbors
        self.eps = eps

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X, y=None):
        data = X

        if not self.clusters:
            self.clusters = self.__fit_clusters(data)

        data['cluster'] = self.clusters.labels_
        data = data[["cluster", "tweet"] + self.columns]

        if self.filter_noise:
            data = data[data.cluster != -1]

        data = data.groupby("cluster").filter(is_valid_polygon(self.location_column))
        data = data.groupby("cluster").agg({
            self.location_column: ['count', calculate_polygon],
            'tweet': [concatenate_text()]
        })
        data = flat_columns(data, self.agg_colnames)
        data = data[data["polygon"].notna()]
        return data.reset_index()

    def __fit_clusters(self, X):
        points_getter = GeoPointTransformer(location_column=self.location_column,
                                            only_points=True)
        points = points_getter.transform(X)
        return OPTICS(cluster_method="dbscan",
                      eps=self.eps,
                      min_samples=self.n_neighbors,
                      metric="minkowski",
                      n_jobs=-2).fit(points)


class ClusterProductPredictor(BaseEstimator, TransformerMixin):
    """Cluster Product Predictor.

    This transformer takes a aggregates cluster
    data frame as input and with the given models,
    it will predict the product class for each.
    """

    def __init__(self, corpus_column,
                 d2v_model_path,
                 category_encoder_path,
                 prod_predictor_path,
                 prod_predictor_name,
                 remove_stop_words=False):
        self.corpus_column = corpus_column
        self.d2v_model_path = d2v_model_path
        self.category_encoder_path = category_encoder_path
        self.prod_predictor_path = prod_predictor_path
        self.prod_predictor_name = prod_predictor_name
        self.classes = None
        self.d2v_model = None
        self.category_encoder = None
        self.prod_predictor_model = None
        self.remove_stop_words = remove_stop_words

    def fit(self, X, y=None):
        return self  # do nothing

    def transform(self, X, y=None):
        self.__load_models()
        data = X
        data["d2v_embeddings"] = data[self.corpus_column].apply(self.__featurize_text)
        data['predictions'] = data["d2v_embeddings"].apply(self.__classify_embedding)
        return data

    def __load_models(self):
        d2v_model = Doc2Vec.load(self.d2v_model_path)

        with open(self.category_encoder_path, "rb") as file:
            category_encoder = pickle.load(file)

        ann_def_path = f"{self.prod_predictor_path}/{self.prod_predictor_name}-def.yaml"
        ann_weights_path = f"{self.prod_predictor_path}/{self.prod_predictor_name}-weights.h5"

        with open(ann_def_path, 'r') as file:
            prod_predictor_model = model_from_yaml(file.read())

        prod_predictor_model.load_weights(ann_weights_path)

        self.d2v_model = d2v_model
        self.category_encoder = category_encoder
        self.prod_predictor_model = prod_predictor_model
        self.classes = category_encoder.classes_.tolist()

    def __featurize_text(self, text):
        ensure_nltk_resource_is_available("punkt")
        ensure_nltk_resource_is_available("stopwords")
        tokenizer = DocumentTokenizer(t_type='regex', regex=r'\w+')
        tokens = tokenizer.tokenize(text)
        return self.d2v_model.infer_vector(tokens)

    def __classify_embedding(self, embedding, with_classes=True, sort=True):
        entry_arr = np.array([embedding]).reshape(1, -1, 1)
        pred = self.prod_predictor_model.predict(entry_arr)[0]

        if with_classes:
            pred = pred.T.tolist()
            pred = [(cat, score) for cat, score in list(zip(self.classes, pred))]
        if sort:
            pred.sort(key=itemgetter(1), reverse=True)
        return pred


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


def calculate_polygon(points):
    """Calculate polygon
    This function will calculate the polygon
    enclosing all the points provided as argument.
    It will use the convex-hull geometric function."""
    points = np.array([list(p.coords[0]) for p in points])
    try:
        hull = ConvexHull(points)
        x = points[hull.vertices, 0]
        y = points[hull.vertices, 1]
        boundaries = list(zip(x, y))
        return Polygon(boundaries)
    except QhullError:
        logging.warning(
            f"There was an error calculating the polygon for cluster with points: {points}. "
            f"This wil be skipped from the final results",
            exc_info=True)

    return None


def is_valid_polygon(location_column):
    """Check if provided polygon is valid.

    Verifies that the input data is a valid polygon.
    """

    def is_valid_polygon_fn(geometry):
        points = geometry[location_column].values
        if isinstance(points, Point):
            return False
        if points.shape[0] < 3:
            return False
        coords = np.array([p.coords[0] for p in points])
        upoints = np.unique(coords)
        if upoints.shape[0] < 3:
            return False
        return True

    return is_valid_polygon_fn
