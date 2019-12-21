import pandas as pd
import os
import logging
from sklearn.pipeline import Pipeline

from wtsp.core.base import DataLoader
from wtsp.core.sklearn.transformers import CountTransformer, DataFrameFilter, MultiValueColumnExpander
from wtsp.exceptions import InvalidArgumentException, DescribeException
from wtsp.view import view
from wtsp.utils import parse_kwargs


class Describer(DataLoader):
    """Describer.

    This is the common place where data set describe
    operations work.
    """
    def __init__(self, output_dir: str,
                 groupby: str,
                 count_col: str,
                 domain: str,
                 filters: str = None,
                 min_count: int = 5000):

        super().__init__()
        try:
            self.filters = parse_kwargs(filters) if filters else None
        except (ValueError, AttributeError) as e:
            raise InvalidArgumentException("Filter value is invalid. use: key=value", e)

        self.output_dir = output_dir
        self.groupby = groupby
        self.count_col = count_col
        self.domain = domain
        self.min_count = min_count

    def describe(self, input_data):
        """Describe.

        It will count the values in the data set
        grouped by the specified values at class
        creation.
        """
        data = self.load_data(input_data)

        count_transformer = CountTransformer(self.groupby,
                                             self.count_col,
                                             self.min_count)

        steps = []
        if self.filters:
            filter_transformer = DataFrameFilter(self.filters)
            steps.append(("data_filter", filter_transformer))

        if self.domain == "documents":
            multi_val_transformer = MultiValueColumnExpander(self.groupby)
            steps.append(("column_expander", multi_val_transformer))

        steps.append(("count_transformer", count_transformer))

        pipeline = Pipeline(steps=steps)

        logging.debug("Executing the describe pipeline")

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

        logging.debug("Saving results in destination folder")
        counts.to_csv(f"{output_dir}/counts.csv")
        view.plot_counts(counts, title, x_label="Cities", save_path=f"{output_dir}/bar_chart.png")

        return f"Result generated successfully at: {output_dir}"
