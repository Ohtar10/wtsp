import logging
import os

import modin.pandas as pd
from sklearn.pipeline import Pipeline

from wtsp.core.base import DEFAULT_TWEETS_COLUMNS, DataLoader, Filterable, Parametrizable
from wtsp.core.sklearn.transformers import DataFrameFilter, GeoPandasTransformer, ClusterAggregator, \
    ClusterProductPredictor
from wtsp.exceptions import WTSPBaseException
from wtsp.view.view import plot_clusters_on_map


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

        logging.debug("Loading ML models")
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
            logging.debug("Transforming and predicting the data.")
            classified_clusters: pd.DataFrame = pipeline.transform(data)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise WTSPBaseException("There is a problem processing the data, see the error message", e)

        # visualization
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
