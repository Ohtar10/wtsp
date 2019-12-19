"""Contains logic to train products related models"""
import pandas as pd
import os

from wtsp.core.sklearn.transformers import DocumentTagger, Doc2VecWrapper
from wtsp.exceptions import InvalidArgumentException
from wtsp.utils import parse_kwargs


class ProductsTrainer:

    def __init__(self, work_dir: str, model: str, params: str):
        if not work_dir:
            raise InvalidArgumentException("The working directory is required.")
        if not params:
            raise InvalidArgumentException("Model parameters are required.")

        self.work_dir = work_dir
        self.model = model
        self.params = parse_kwargs(params)

    def train(self, input_data) -> str:
        result = ""
        if self.model == "embeddings":
            trainer = DocumentEmbeddingsTrainer(self.work_dir, **self.params)
            result = trainer.train(input_data)

        return result


class DocumentEmbeddingsTrainer:

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
