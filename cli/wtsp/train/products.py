"""Contains logic to train products related models"""
import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from wtsp.core.base import Parametrizable, DataLoader, Trainer, DEFAULT_PRODUCT_DOCS_COLUMNS
from wtsp.core.sklearn.transformers import DocumentTagger, Doc2VecWrapper, CategoryEncoder, ProductsCNN
from wtsp.exceptions import InvalidArgumentException, ModelTrainingException
from wtsp.utils import parse_kwargs
from wtsp.view.view import plot_cnn_history, plot_classification_report


class ProductsTrainer(Parametrizable):
    """Products Trainer.

    Main orchestrator of product related models.
    """
    def __init__(self, work_dir: str, model: str, params: str):
        super().__init__(params)
        if not work_dir:
            raise InvalidArgumentException("The working directory is required.")

        self.work_dir = work_dir
        self.model = model
        self.params: Dict[str, object] = parse_kwargs(params)

    def train(self, input_data) -> str:
        trainer: Optional[Trainer] = None
        if self.model == "embeddings":
            trainer = DocumentEmbeddingsTrainer(self.work_dir, **self.params)
        elif self.model == "classifier":
            trainer = ProductsClassifierTrainer(self.work_dir, **self.params)

        if not trainer:
            raise InvalidArgumentException(f"There is no '{self.model}' model to train")

        result = trainer.train(input_data)
        return result


class DocumentEmbeddingsTrainer(Trainer, DataLoader):
    """Document Embeddings trainer.

    Orchestrates the document embedding training.
    """
    def __init__(self, work_dir,
                 label_col="categories",
                 doc_col="document",
                 lr=0.01,
                 epochs=10,
                 vec_size=100,
                 alpha=0.1,
                 min_alpha=0.0001,
                 min_count=1,
                 dm=0,
                 **kwargs):
        super().__init__()
        self.work_dir = work_dir
        self.label_col = label_col
        self.doc_col = doc_col
        self.lr = lr
        self.epochs = epochs
        self.vec_size = vec_size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.dm = dm

        # overwrite those provided
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def train(self, input_data: str) -> str:
        data = self.load_data(input_data, DEFAULT_PRODUCT_DOCS_COLUMNS)

        document_tagger = DocumentTagger(self.label_col,
                                         self.doc_col)
        d2v_wrapper = Doc2VecWrapper(document_column=self.doc_col,
                                     lr=self.lr,
                                     epochs=self.epochs,
                                     vec_size=self.vec_size,
                                     alpha=self.alpha,
                                     min_alpha=self.min_alpha,
                                     min_count=self.min_count,
                                     dm=self.dm)

        try:
            logging.info("Creating tagged documents...")
            tagged_docs = document_tagger.transform(data)
            del data  # free up memory from the original dataset
            logging.info("Training document embeddings...")
            d2v_wrapper.fit(tagged_docs)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise ModelTrainingException("There is a problem processing the data, see the error message", e)

        # save the model
        output_dir = f"{self.work_dir}/products/models/embeddings"
        os.makedirs(output_dir, exist_ok=True)
        d2v_wrapper.save_model(f"{output_dir}/d2v_model.model")
        return f"Product document embeddings trained successfully. Result is stored at: {output_dir}"


class ProductsClassifierTrainer(Trainer, DataLoader):
    """Product Classifier Trainer.

    Orchestrates the training of the product
    classifier.
    """
    def __init__(self, work_dir: str,
                 classes: int,
                 document_column: str = "document",
                 label_col: str = "category",
                 test_size=0.3,
                 vec_size=100,
                 epochs=100,
                 lr=0.001,
                 batch_size=1000,
                 validation_split=0.2,
                 from_embeddings=False,
                 **kwargs):
        super().__init__()
        self.work_dir = work_dir
        self.classes = classes
        self.document_column = document_column
        self.label_column = label_col
        self.test_size=test_size
        self.vec_size = vec_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.lr = lr
        self.from_embeddings = from_embeddings
        self.category_encoder = None
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __transform_embeddings(self, data):
        d2v_model_path = f"{self.work_dir}/products/models/embeddings/d2v_model.model"
        if not os.path.exists(d2v_model_path):
            raise ModelTrainingException("No product embeddings found in the working directory. "
                                         "This is needed to train the classifier")
        embeddings_transformer = Doc2VecWrapper.load(d2v_model_path,
                                                     document_column=self.document_column)
        embeddings_transformer.tag_doc_column = self.label_column

        try:
            logging.info("Transforming documents into embeddings...")
            return embeddings_transformer.transform(data)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise ModelTrainingException("There is a problem processing the data, see the error message", e)

    def __dataset_from_embeddings(self, input_data: str):
        category_encoder_path = f"{input_data}/category_encoder.save"
        self.category_encoder = CategoryEncoder()
        self.category_encoder.load_model(category_encoder_path)

        embeddings_path = f"{input_data}/document_embeddings.npz"
        embeddings = np.load(embeddings_path)

        if 'x_train' in embeddings:
            return embeddings['x_train'], embeddings['x_test'], embeddings['y_train'], embeddings['y_test']
        else:
            return train_test_split(embeddings['x'], embeddings['y'])

    def __dataset_from_scratch(self, input_data: str):
        data = self.load_data(input_data, DEFAULT_PRODUCT_DOCS_COLUMNS)
        document_embeddings = self.__transform_embeddings(data)
        category_encoder = CategoryEncoder(label_column=self.label_column)
        try:
            logging.info("Encoding the categories...")
            encoded_categories = category_encoder.fit_transform(data)
            self.category_encoder = category_encoder
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise ModelTrainingException("There is a problem processing the data, see the error message", e)

        # train test split to validate at the end
        y = encoded_categories
        X_train, X_test, y_train, y_test = train_test_split(document_embeddings.values, y, test_size=self.test_size)
        # Free up some space
        del document_embeddings
        del data
        del encoded_categories
        return X_train, X_test, y_train, y_test

    def train(self, input_data: str) -> str:

        if self.from_embeddings:
            X_train, X_test, y_train, y_test = self.__dataset_from_embeddings(input_data)
        else:
            X_train, X_test, y_train, y_test = self.__dataset_from_scratch(input_data)

        logging.info("Training the Neural Network...")
        prod_classifier_cnn = ProductsCNN(features_column="d2v_embedding",
                                          label_column="encoded_label",
                                          classes=self.classes,
                                          learning_rate=self.lr,
                                          vec_size=self.vec_size,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_split=self.validation_split)
        try:
            prod_classifier_cnn.fit(X_train, y_train)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise ModelTrainingException("There is a problem processing the data, see the error message", e)

        # score against the testing set
        logging.info("Scoring against the testing set...")
        X_test_rs = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred = np.where(prod_classifier_cnn.ann_model.predict(X_test_rs) > 0.5, 1., 0.)
        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # persist the results
        logging.info("Saving results...")
        output_dir = f"{self.work_dir}/products/models/classifier"
        os.makedirs(output_dir, exist_ok=True)

        if not self.from_embeddings:
            self.category_encoder.save_model(f"{output_dir}/category_encoder.model")

        prod_classifier_cnn.save_model(f"{output_dir}", "prod_classifier")

        # plot the history
        plot_cnn_history(prod_classifier_cnn.ann_model.history,
                         f"{output_dir}/training_history.png")

        # plot classification report
        title = f"Products classification report (Acc: {acc:.2f})"
        file = f"{output_dir}/classification_report.png"
        plot_classification_report(cr,
                                   self.category_encoder.label_encoder.classes_,
                                   file,
                                   title)
        return f"Product classifier trained successfully. Result is stored at: {output_dir}"
