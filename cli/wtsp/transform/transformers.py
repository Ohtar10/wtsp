import os

import pandas as pd
from sklearn.pipeline import Pipeline

from wtsp.core.base import DEFAULT_TWEETS_COLUMNS
from wtsp.core.sklearn.transformers import DataFrameFilter, GeoPandasTransformer, ClusterAggregator, \
    ClusterProductPredictor
from wtsp.utils import parse_kwargs
from wtsp.view.view import plot_clusters_on_map


class WhereToSellProductsTransformer:
    def __init__(self, work_dir: str, filters: str, params: str):
        self.work_dir = work_dir
        self.filters = parse_kwargs(filters)
        self.params = parse_kwargs(params)

    def transform(self, input_data: str):
        data = pd.read_parquet(input_data, engine="pyarrow")

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
        classified_clusters: pd.DataFrame = pipeline.transform(data)

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
