import os
import logging
from sklearn.pipeline import Pipeline

from wtsp.core.base import DataLoader, Filterable, DEFAULT_TWEETS_COLUMNS, DEFAULT_PRODUCT_DOCS_COLUMNS
from wtsp.core.sklearn.transformers.generic import CountTransformer, DataFrameFilter, MultiValueColumnExpander
from wtsp.exceptions import DescribeException
from wtsp.view import view


class Describer(DataLoader, Filterable):
    """Describer.

    This is the common place where data set describe
    operations work.
    """
    def __init__(self, output_dir: str,
                 groupby: str,
                 count_col: str,
                 domain: str,
                 filters: str = None,
                 min_count: int = 5000,
                 explode: bool = False):
        DataLoader.__init__(self)
        Filterable.__init__(self, filters, can_be_none=True)
        self.output_dir = output_dir
        self.groupby = groupby
        self.count_col = count_col
        self.domain = domain
        self.min_count = min_count
        self.explode = explode

    def describe(self, input_data):
        """Describe.

        It will count the values in the data set
        grouped by the specified values at class
        creation.
        """
        if self.domain == "tweets":
            columns = DEFAULT_TWEETS_COLUMNS
        else:
            columns = DEFAULT_PRODUCT_DOCS_COLUMNS

        data = self.load_data(input_data, columns)

        count_transformer = CountTransformer(self.groupby,
                                             self.count_col,
                                             self.min_count)

        steps = []
        if self.filters:
            filter_transformer = DataFrameFilter(self.filters)
            steps.append(("data_filter", filter_transformer))

        if self.domain == "documents" and self.explode:
            multi_val_transformer = MultiValueColumnExpander(self.groupby)
            steps.append(("column_expander", multi_val_transformer))

        steps.append(("count_transformer", count_transformer))

        pipeline = Pipeline(steps=steps)

        logging.info(f"Describing elements in {input_data}")

        try:
            counts = pipeline.transform(data)
        except Exception as e:
            logging.error("There is a problem processing the data, see the error message", e)
            raise DescribeException("There is a problem processing the data, see the error message", e)

        logging.debug("Ensuring output folders exist")
        output_dir = f"{self.output_dir}/{self.domain}"
        if self.filters:
            filter_field = next(iter(self.filters))
            filter_value = self.filters[filter_field]
            output_dir = f"{output_dir}/{filter_field}={filter_value}"
            title = f"{self.domain.capitalize()} count by {self.groupby} in {filter_value}"
        else:
            title = f"{self.domain.capitalize()} count by {self.groupby}"

        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Saving results in destination folder: {output_dir}")
        counts.to_csv(f"{output_dir}/counts.csv")
        x_label = "Categories" if self.domain == "documents" else "Cities"
        view.plot_counts(counts, title, x_label=x_label, save_path=f"{output_dir}/bar_chart.png")

        return f"Result generated successfully at: {output_dir}"
