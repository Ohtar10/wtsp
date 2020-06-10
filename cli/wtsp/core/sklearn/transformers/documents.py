import logging
import math
import multiprocessing
import os
import pickle
import warnings

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from modin import pandas as pd
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

from wtsp.core import get_df_engine
from wtsp.core.sklearn.transformers import init_tensorflow
from wtsp.utils import ensure_nltk_resource_is_available

# To suppress tensorflow warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # from tensorflow import logging as tf_logging
    # tf_logging.set_verbosity(tf_logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras import Input, Model
    from keras.callbacks import EarlyStopping
    from keras.layers import Conv1D, SpatialDropout1D, MaxPool1D, Flatten, Dense, Dropout
    from keras.optimizers import Adam


class DocumentTagger(BaseEstimator, TransformerMixin):
    """Document Tagger.

    It will add a new column to the given
    data frame containing the tagged document
    of the corpus column.
    """

    def __init__(self, tags_column, corpus_column, remove_stop_words=False):
        self.tags_column = tags_column
        self.corpus_column = corpus_column
        self.remove_stop_words = remove_stop_words

    def fit(self, X, y=None):
        return self  # Do nothing

    def transform(self, X, y=None):
        ensure_nltk_resource_is_available("punkt")
        ensure_nltk_resource_is_available("stopwords")
        tokenizer = DocumentTokenizer(t_type='regex', regex=r'\w+')
        return X.apply(lambda row: TaggedDocument(words=tokenizer.tokenize(row[self.corpus_column]),
                                                  tags=row[self.tags_column].split(';')),
                       axis=1)


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
                 dm=0,
                 remove_stop_words=False):
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
        self.remove_stop_words = remove_stop_words

    def fit(self, X, y=None):
        # Always use 90% of the capacity
        workers = math.floor(multiprocessing.cpu_count() * .9)
        d2v_model = Doc2Vec(vector_size=self.vec_size,
                            alpha=self.alpha,
                            min_alpha=self.min_alpha,
                            min_count=self.min_count,
                            dm=self.dm,
                            workers=workers)
        logging.info(f"Building vocabulary out of {X.shape[0]} documents...")

        if get_df_engine() == "pandas":
            tagged_documents = X
        else:
            tagged_documents = X[0]

        total_words = tagged_documents.apply(lambda row: len(row.words)).sum()
        d2v_model.build_vocab(tagged_documents)
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

    def transform(self, X, y=None):
        tagger = DocumentTagger(self.tag_doc_column, self.document_column)
        embeddings = tagger.transform(X)

        if get_df_engine() == "pandas":
            return embeddings.to_frame().apply(lambda row: self.d2v_model.infer_vector(row[0].words), axis=1,
                                               result_type='expand')
        else:
            embeddings = embeddings.apply(lambda row: self.d2v_model.infer_vector(row[0].words), axis=1)
            return embeddings._to_pandas().apply(lambda row: row[0], axis=1, result_type='expand')

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
        categories = data[self.label_column].apply(lambda cat: cat.split(";"))
        category_encoder = MultiLabelBinarizer()
        category_encoder.fit(categories)
        self.label_encoder = category_encoder
        return self

    def transform(self, X, y=None):
        data = X
        categories = data[self.label_column].apply(lambda cat: cat.split(";"))
        encoded_labels = self.label_encoder.transform(categories)
        return encoded_labels

    def save_model(self, save_path):
        with open(save_path, 'wb') as model_file:
            pickle.dump(self.label_encoder, model_file)

    def load_model(self, load_path):
        with open(load_path, 'rb') as model_file:
            self.label_encoder = pickle.load(model_file)


class ProductsCNN(BaseEstimator, TransformerMixin):
    """Products Classifier CNN.

    This transformer contains the logic and architecture
    definition of the Convolutional Neural Network that classifies
    products in based on the document embeddings that represents them.
    """

    def __init__(self, features_column,
                 label_column,
                 classes,
                 learning_rate=0.001,
                 vec_size=100,
                 epochs=100,
                 batch_size=1000,
                 validation_split=0.2):
        self.features_column = features_column
        self.label_column = label_column
        self.vec_size = vec_size
        self.classes = classes
        self.learning_rate = learning_rate
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
        conv = Conv1D(384, 5, activation='relu')(embedding_input)
        conv = SpatialDropout1D(0.1)(conv)
        conv = MaxPool1D(2, strides=2, padding='valid')(conv)

        conv = Conv1D(192, 2, activation='relu')(conv)
        conv = SpatialDropout1D(0.1)(conv)
        conv = MaxPool1D(2, strides=2, padding='valid')(conv)

        conv = Conv1D(96, 2, activation='relu')(conv)
        conv = SpatialDropout1D(0.1)(conv)
        conv = MaxPool1D(2, strides=2, padding='valid')(conv)

        conv = Conv1D(48, 2, activation='relu')(conv)
        conv = SpatialDropout1D(0.1)(conv)
        conv = MaxPool1D(2, strides=2, padding='valid')(conv)

        conv = Conv1D(32, 2, activation='relu')(conv)
        conv = SpatialDropout1D(0.1)(conv)
        conv = MaxPool1D(2, strides=2, padding='valid')(conv)

        conv_output = Flatten()(conv)

        # Define dense layers
        # minimize the dense layers - maybe add one of 64
        x = Dense(416, activation='tanh')(conv_output)
        x = Dropout(0.2)(x)
        x = Dense(208, activation='tanh')(x)
        x = Dropout(0.2)(x)

        # And finally make the predictions using the previous layer as input
        main_output = Dense(self.classes, activation='softmax', name='prediction')(x)

        ann_model = Model(inputs=embedding_input, outputs=main_output)
        optimizer = Adam(learning_rate=self.learning_rate)
        ann_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        self.ann_model = ann_model

    def fit(self, X, y=None):
        self.__build_ann_architecture()
        X_rs = X.reshape(X.shape[0], X.shape[1], 1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-7, restore_best_weights=True)
        history = self.ann_model.fit(X_rs, y,
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


class DocumentTokenizer:
    def __init__(self, t_type: str = 'word_tokenizer',
                 regex: str = r'\w+',
                 lower: bool = True,
                 remove_stop_words: bool = False):
        self.t_type = t_type
        self.regex = regex
        self.lower = lower
        self.remove_stop_words = remove_stop_words
        self.tokenizer = self.__init_tokenizer()

    def __init_tokenizer(self):
        if self.t_type == 'word_tokenizer':
            return word_tokenize
        elif self.t_type == 'regex':
            return RegexpTokenizer(self.regex)

    def tokenize(self, string: str):
        tokens = []
        if self.t_type == 'word_tokenizer':
            tokens = self.tokenizer(string)
        elif self.t_type == 'regex':
            tokens = self.tokenizer.tokenize(string)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        if self.remove_stop_words:
            stop_words = set(stopwords.words('english'))
            return [word for word in tokens if word not in stop_words]
        else:
            return tokens


def create_tagged_document_fn(tags_column, corpus_column, remove_stop_words=False):
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
        tokenizer = DocumentTokenizer()
        if remove_stop_words:
            stop_words = set(stopwords.words('english'))
        else:
            stop_words = set()
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


