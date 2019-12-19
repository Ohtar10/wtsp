"""Contains logic to train products related models"""
import logging

import pandas as pd
import os

from wtsp.core.sklearn.transformers import DocumentTagger, Doc2VecWrapper, CategoryEncoder, ProductsCNN
from wtsp.exceptions import InvalidArgumentException, ModelTrainingException
from wtsp.utils import parse_kwargs


class ProductsTrainer:
    """Products Trainer.

    Main orchestrator of product related models.
    """
    def __init__(self, work_dir: str, model: str, params: str):
        if not work_dir:
            raise InvalidArgumentException("The working directory is required.")
        if not params:
            raise InvalidArgumentException("Model parameters are required.")

        self.work_dir = work_dir
        self.model = model
        self.params = parse_kwargs(params)

    def train(self, input_data) -> str:
        trainer = None
        if self.model == "embeddings":
            trainer = DocumentEmbeddingsTrainer(self.work_dir, **self.params)
        elif self.model == "classifier":
            trainer = ProductsClassifierTrainer(self.work_dir, **self.params)

        if not trainer:
            raise InvalidArgumentException(f"There is no '{self.model}' model to train")

        result = trainer.train(input_data)
        return result


class DocumentEmbeddingsTrainer:
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
        data = pd.read_parquet(input_data, engine="pyarrow")

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

        tagged_docs = document_tagger.transform(data)
        d2v_wrapper.fit(tagged_docs)

        # save the model
        output_dir = f"{self.work_dir}/products/models/embeddings"
        os.makedirs(output_dir, exist_ok=True)
        d2v_wrapper.save_model(f"{output_dir}/d2v_model.model")
        return f"Product document embeddings trained successfully. Result is stored at: {output_dir}"


class ProductsClassifierTrainer:
    """Product Classifier Trainer.

    Orchestrates the training of the product
    classifier.
    """
    def __init__(self, work_dir: str,
                 classes: int,
                 document_column: str = "document",
                 label_column: str = "categories",
                 vec_size=100,
                 epochs=100,
                 batch_size=1000,
                 validation_split=0.2,
                 **kwargs):
        self.work_dir = work_dir
        self.classes = classes
        self.document_column = document_column
        self.label_column = label_column
        self.vec_size = vec_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def train(self, input_data: str) -> str:
        data = pd.read_parquet(input_data, engine="pyarrow")
        d2v_model_path = f"{self.work_dir}/products/models/embeddings/d2v_model.model"
        if not os.path.exists(d2v_model_path):
            raise ModelTrainingException("No product embeddings found in the working directory. "
                                         "This is needed to train the classifier")
        embeddings_transformer = Doc2VecWrapper.load(d2v_model_path,
                                                     document_column=self.document_column)
        category_encoder = CategoryEncoder(label_column=self.label_column)
        prod_classifier_cnn = ProductsCNN(features_column="d2v_embedding",
                                          label_column="encoded_label",
                                          classes=self.classes,
                                          vec_size=self.vec_size,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          validation_split=self.validation_split)

        # Since we only need to execute transform in some and fit in others
        # we invoke them directly instead of chaining them in a pipeline
        logging.debug("Transforming documents into embeddings...")
        document_embeddings = embeddings_transformer.transform(data)

        logging.debug("Encoding the categories...")
        encoded_embeddings = category_encoder.fit_transform(document_embeddings)

        logging.debug("Training the Neural Network...")
        y = encoded_embeddings["encoded_label"].values
        prod_classifier_cnn.fit(encoded_embeddings, y)

        # persist the results
        output_dir = f"{self.work_dir}/products/models/classifier"
        os.makedirs(output_dir, exist_ok=True)
        category_encoder.save_model(f"{output_dir}/category_encoder.model")
        prod_classifier_cnn.save_model(f"{output_dir}", "prod_classifier")
        return f"Product classifier trained successfully. Result is stored at: {output_dir}"
