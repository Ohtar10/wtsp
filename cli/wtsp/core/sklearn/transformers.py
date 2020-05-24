"""Scikit learn transformers of data."""
import logging
import math
import os
import pickle
import warnings
import multiprocessing
from operator import itemgetter
from typing import Dict

from scipy.spatial.qhull import ConvexHull
from shapely.geometry import Polygon, Point
from sklearn.cluster import OPTICS

# To suppress tensorflow warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #from tensorflow import logging as tf_logging
    #tf_logging.set_verbosity(tf_logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from keras import regularizers
    from keras.callbacks import EarlyStopping
    from keras.engine.saving import model_from_yaml
    from keras.layers import Input, Dense, Conv1D, Flatten, MaxPool1D, Dropout, SpatialDropout1D
    from keras.models import Model


import geopandas as gpd
import numpy as np
import modin.pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from shapely import wkt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

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
        data = X
        logging.info("Aggregating and counting records...")
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
        data = X
        logging.info("Filtering records...")
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
        data = X
        logging.info(f"Expanding values in {self.expand_column}")
        split_fn = lambda x: x.split(self.value_split_char)
        data[self.expand_column] = data[self.expand_column].apply(split_fn)
        return data.explode(self.expand_column)


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
        ensure_nltk_resource_is_available("stopwords")
        data = X
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        tagged_docs = data.agg(lambda x: TaggedDocument(
            words=[word.lower() for word in tokenizer.tokenize(x[self.corpus_column]) if word.lower() not in stop_words],
            tags=[x[self.tags_column]]), axis=1)
        return tagged_docs[0]


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
        # Always use 90% of the capacity
        workers = math.floor(multiprocessing.cpu_count() * .9)
        d2v_model = Doc2Vec(vector_size=self.vec_size,
                            alpha=self.alpha,
                            min_alpha=self.min_alpha,
                            min_count=self.min_count,
                            dm=self.dm,
                            workers=workers)

        logging.info(f"Calculating word frequency out of {X.shape[0]} documents...")
        word_freq = X.apply(lambda x: x.words).explode().value_counts()
        total_words = word_freq.sum()
        word_freq = word_freq[word_freq >= self.min_count].to_dict()
        logging.info(f"Building vocabulary out of {len(word_freq)} words...")
        d2v_model.build_vocab_from_freq(word_freq, corpus_count=X.shape[0])

        tagged_documents = Doc2VecWrapper.__create_tagged_document_generator(X)
        logging.info("Calculating embeddings...")
        for epoch in range(self.epochs):
            logging.info(f"Meta epoch {epoch}/{self.epochs}...")
            d2v_model.train(tagged_documents,
                            total_examples=d2v_model.corpus_count,
                            epochs=d2v_model.epochs,
                            total_words=total_words)
            d2v_model.alpha -= self.lr
            d2v_model.min_alpha = d2v_model.alpha

        d2v_model.delete_temporary_training_data()
        self.d2v_model = d2v_model

        return self

    @staticmethod
    def __create_tagged_document_generator(documents: pd.DataFrame):
        def generator():
            n = 0
            while n < documents.shape[0]:
                yield documents[n]
                n += 1
        return generator()

    def transform(self, X, y=None):
        ensure_nltk_resource_is_available("punkt")
        ensure_nltk_resource_is_available("stopwords")
        data = X
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        tagged_docs = data[self.document_column].apply(
            lambda x: [w.lower() for w in tokenizer.tokenize(x) if w.lower() not in stop_words])
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

    @staticmethod
    def load(model_path: str, document_column: str):
        d2v_model = Doc2Vec.load(model_path)
        transformer = Doc2VecWrapper(document_column=document_column)
        transformer.d2v_model = d2v_model
        return transformer


class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Category encoder.

    This transformer will train a sklearn MultilabelBinarizer
    on the selected column of the provided data frame and
    on transform it will add the encoded values as an
    additional column. The encoder can be saved for
    later usage.
    """
    def __init__(self, label_column="category"):
        self.label_column = label_column
        self.label_encoder = None

    def fit(self, X, y=None):
        data = X
        categories = data[self.label_column].apply(lambda cat: cat.split(";")).values.tolist()
        category_encoder = MultiLabelBinarizer()
        category_encoder.fit(categories)
        self.label_encoder = category_encoder
        return self

    def transform(self, X, y=None):
        data = X
        categories = data[self.label_column].apply(lambda cat: cat.split(";")).values.tolist()
        encoded_labels = self.label_encoder.transform(categories)
        encoded_labels = [arr for arr in encoded_labels]
        data["encoded_label"] = encoded_labels
        return data

    def save_model(self, save_path):
        with open(save_path, 'wb') as model_file:
            pickle.dump(self.label_encoder, model_file)


class ProductsCNN(BaseEstimator, TransformerMixin):
    """Products Classifier CNN.

    This transformer contains the logic and architecture
    definition of the Convolutional Neural Network that classifies
    products in based on the document embeddings that represents them.
    """
    def __init__(self, features_column, label_column, classes,
                 vec_size=100,
                 epochs=100,
                 batch_size=1000,
                 validation_split=0.2):

        self.features_column = features_column
        self.label_column = label_column
        self.vec_size = vec_size
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.history = None
        self.ann_model = None

    def __build_ann_architecture(self):
        init_tensorflow()
        # Define the inputs
        embedding_input = Input(shape=(self.vec_size, 1), dtype='float32', name='comment_text')

        # Define convolutional layers
        conv = Conv1D(64, 3, activation='tanh', input_shape=(self.vec_size,), kernel_regularizer=regularizers.l2())(
            embedding_input)
        conv = MaxPool1D(2, strides=None, padding='valid')(conv)
        conv = Conv1D(128, 3, activation='tanh')(conv)
        conv = SpatialDropout1D(0.2)(conv)
        conv = MaxPool1D(2, strides=None, padding='valid')(conv)
        conv = Conv1D(128, 3, activation='tanh')(conv)
        conv = MaxPool1D(2, strides=None, padding='valid')(conv)
        conv = SpatialDropout1D(0.1)(conv)
        conv = Conv1D(64, 3, activation='tanh')(conv)
        conv = MaxPool1D(2, strides=None, padding='valid')(conv)
        conv_output = Flatten()(conv)

        # Define dense layers
        # minimize the dense layers - maybe add one of 64
        x = Dense(128, activation='relu')(conv_output)
        x = Dropout(0.5)(x)

        # And finally make the predictions using the previous layer as input
        main_output = Dense(self.classes, activation='softmax', name='prediction')(x)

        ann_model = Model(inputs=embedding_input, outputs=main_output)
        ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.ann_model = ann_model

    def fit(self, X, y=None):
        self.__build_ann_architecture()
        X_train = X[self.features_column].values
        X_train = np.array([e for e in X_train])
        y_true = np.array([l for l in y])
        X_rs = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-7, restore_best_weights=True)
        history = self.ann_model.fit(X_rs, y_true,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     validation_split=self.validation_split,
                                     callbacks=[early_stopping])
        self.history = history
        return self

    def transform(self, X, y=None):
        data = X
        features = data[self.features_column].values
        features = np.array([e for e in features])
        features = features.reshape(features.shape[0], features.shape[1], 1)
        predictions = self.ann_model.predict(features)
        predictions = [p for p in predictions]
        data["predictions"] = predictions
        return data

    def save_model(self, save_path, name=None):
        if not name:
            name = "product_cnn"
        definition_path = f"{save_path}/{name}-def.yaml"
        weights_path = f"{save_path}/{name}-weights.h5"

        ann_model_definition = self.ann_model.to_yaml()

        with open(definition_path, 'w') as file:
            file.write(ann_model_definition)

        self.ann_model.save_weights(weights_path)


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
                 prod_predictor_name):
        self.corpus_column = corpus_column
        self.d2v_model_path = d2v_model_path
        self.category_encoder_path = category_encoder_path
        self.prod_predictor_path = prod_predictor_path
        self.prod_predictor_name = prod_predictor_name
        self.classes = None
        self.d2v_model = None
        self.category_encoder = None
        self.prod_predictor_model = None

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
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        tokens = [token.lower() for token in tokenizer.tokenize(text) if token.lower() not in stop_words]
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


def flat_columns(df: pd.DataFrame, colnames: Dict[str, str] = None):
    """Flat columns.

    Given a multi-indexed pandas DataFrame,
    it will flat the column names given the
    colnames.
    """
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
        ensure_nltk_resource_is_available("punkt")
        ensure_nltk_resource_is_available("stopwords")
        categories = row[tags_column]
        document = row[corpus_column]
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in tokenizer.tokenize(document) if word.lower() not in stop_words]
        tagged_document = TaggedDocument(words=words, tags=categories.split(";"))
        index = [tags_column, corpus_column, "tagged_document"]
        return pd.Series([categories, document, tagged_document], index=index)

    return create_tagged_document


def concatenate_text(sep='\n'):
    """Concatenate Text
    This function will simply join every
    text it receives as a single text and
    separate it by the provide separator.
    Default \n """

    def concat(text):
        return sep.join(text)

    return concat


def calculate_polygon(points):
    """Calculate polygon
    This function will calculate the polygon
    enclosing all the points provided as argument.
    It will use the convex-hull geometric function."""
    points = np.array([list(p.coords[0]) for p in points])
    hull = ConvexHull(points)
    x = points[hull.vertices, 0]
    y = points[hull.vertices, 1]
    boundaries = list(zip(x, y))

    return Polygon(boundaries)


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


def init_tensorflow():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            from tensorflow.compat.v1 import ConfigProto, InteractiveSession
            config = ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            # Necessary to default a session in tensorflow for keras to grab
            session = InteractiveSession(config=config)
