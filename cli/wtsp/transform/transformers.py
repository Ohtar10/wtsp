import logging
import os

import modin.pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from wtsp.core.base import DEFAULT_TWEETS_COLUMNS, DataLoader, Filterable, Parametrizable, DEFAULT_PRODUCT_DOCS_COLUMNS
from wtsp.core.sklearn.transformers import DataFrameFilter, GeoPandasTransformer, ClusterAggregator, \
    ClusterProductPredictor
from wtsp.exceptions import WTSPBaseException
from wtsp.view.view import plot_clusters_on_map
from wtsp.core import get_df_engine


class WhereToSellProductsTransformer(DataLoader, Filterable, Parametrizable):
    """Where to sell products transformer.

    Main transformers which takes the tweet data provided
    and will predict the polygons and their relationship
    with products.
    """
    def __init__(self, work_dir: str, filters: str, params: str):
        DataLoader.__init__(self)
        Filterable.__init__(self, filters)
        Parametrizable.__init__(self, params)
        self.work_dir = work_dir

    def transform(self, input_data: str):
        data = self.load_data(input_data, DEFAULT_TWEETS_COLUMNS)

        location_column = self.params["location_column"]

        filter_transformer = DataFrameFilter(self.filters)
        geo_transformer = GeoPandasTransformer(location_column,
                                               DEFAULT_TWEETS_COLUMNS)
        colnames = {f"{location_column}calculate_polygon": "polygon",
                    f"{location_column}count": "size",
                    "tweetconcat": "corpus"}
        eps = self.params["eps"]
        n_neighbors = self.params["n_neighbors"]
        cluster_aggregator = ClusterAggregator(columns=[location_column],
                                               location_column=location_column,
                                               agg_colnames=colnames,
                                               eps=eps,
                                               n_neighbors=n_neighbors)

        logging.info("Loading ML models")
        models_path = f"{self.work_dir}/products/models/"
        label_encoder_path= f"{models_path}/classifier/category_encoder.model"
        d2v_model_path = f"{models_path}/embeddings/d2v_model.model"
        prod_classifier_path = f"{models_path}/classifier/"
        prod_classifier_name = "prod_classifier"
        cluster_predictor = ClusterProductPredictor("corpus",
                                                    d2v_model_path,
                                                    label_encoder_path,
                                                    prod_classifier_path,
                                                    prod_classifier_name)

        pipeline = Pipeline(
            [
                ("filter", filter_transformer),
                ("geo_transformer", geo_transformer),
                ("cluster_aggergator", cluster_aggregator),
                ("cluster_predictor", cluster_predictor)
            ]
        )

        try:
            logging.info("Transforming and predicting the data.")
            classified_clusters: pd.DataFrame = pipeline.transform(data)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise WTSPBaseException("There is a problem processing the data, see the error message", e)

        # visualization
        logging.info("Generating visualizations...")
        filter_key = next(iter(self.filters))
        filter_value = self.filters[filter_key]
        result_dir = f"{self.work_dir}/where_to_sell_in/{filter_key}={filter_value}"
        os.makedirs(result_dir, exist_ok=True)

        classified_clusters.to_csv(f"{result_dir}/classified_clusters.csv", index=False)
        center = [float(s.strip()) for s in self.params["center"].split(";")]
        plot_clusters_on_map(classified_clusters,
                             f"{result_dir}/classified_clusters.html",
                             center=center,
                             print_classes=True,
                             score_threshold=self.params["min_score"])
        return f"Transformation finished successfully. Results saved in {result_dir}"


class EmbeddingsTransformer(DataLoader):
    """Embeddings Transformer.

    This class will take the input data and the document 2 vec
    model, that should be available in the working directory,
    to transform the raw documents into embeddings and will
    save the results as compressed numpy objects.
    """
    def __init__(self, work_dir: str,
                 label_column,
                 document_column,
                 tt_split: bool,
                 test_size: float):
        DataLoader.__init__(self)
        self.work_dir = work_dir
        self.label_column = label_column
        self.document_column = document_column
        self.tt_split = tt_split
        self.test_size = test_size

    def transform(self, input_data: str):
        data = self.load_data(input_data, DEFAULT_PRODUCT_DOCS_COLUMNS)

        logging.info("Loading ML models")
        models_path = f"{self.work_dir}/products/models/"
        d2v_model_path = f"{models_path}/embeddings/d2v_model.model"
        d2v_model = Doc2Vec.load(d2v_model_path)

        logging.info("Encoding categories")
        categories = data[self.label_column].apply(lambda x: x.split(";")).values.tolist()
        categories_encoder = MultiLabelBinarizer()
        categories_encoder.fit(categories)

        logging.info("Encoding documents")
        tokenizer = RegexpTokenizer(r'\w+')
        y = categories_encoder.transform(categories)
        X = data.apply(
            lambda row: d2v_model.infer_vector(
                [word.lower() for word in tokenizer.tokenize(row[self.document_column])]), axis=1)

        if get_df_engine() == 'modin':
            X = X._to_pandas()
        else:
            X = X.to_frame()

        X = X.apply(lambda x: x[0], axis=1, result_type='expand')

        logging.info("Saving results...")
        result_dir = f"{self.work_dir}/embeddings/"
        result_file = f"{result_dir}document_embeddings.npz"
        os.makedirs(result_dir, exist_ok=True)
        if self.tt_split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            np.savez_compressed(f"{result_file}", x_train=X_train, x_test=X_test, y_train=y_train,
                                y_test=y_test)
        else:
            np.savez_compressed(f"{result_file}", x=X, y=y)

        return f"Transformation finished successfully. Results saved in {result_file}"
