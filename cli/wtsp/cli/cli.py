"""CLI implementation."""

import logging

import click

from wtsp.__version__ import __version__
from wtsp.core.base import DEFAULT_WORK_DIR
from wtsp.describe.describe import Describer
from wtsp.exceptions import ModelTrainingException
from wtsp.train.products import ProductsTrainer
from wtsp.train.tweets import TweetsTrainer


def docstring_parameter(*sub):
    """Decorate the main click command to format the docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output.')
@click.option('-wd', '--work-dir', default=DEFAULT_WORK_DIR,
              help='Which folder to use as working directory. Default to ~/.wtsp')
@click.pass_context
@docstring_parameter(__version__)
def wtsp(ctx, debug, work_dir):
    """Where To Sell Products (wtsp) {0}."""
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s ",
                        level=logging.INFO if not debug else logging.DEBUG)
    ctx.ensure_object(dict)
    ctx.obj['WORK_DIR'] = work_dir


@wtsp.group()
@click.pass_context
def describe(ctx):
    """Describe module.

    Use this module to generate descriptive data 
    that might help you to take decisions.
    """
    pass


@describe.command("tweets")
@click.pass_context
@click.option("-f", "--filters", required=True, help="Filters to use over the data set columns to narrow down the load.")
@click.option("-o", "--output-dir", required=True, help="Path where the describe results will be printed out.")
@click.option("-g", "--groupby", default="place_name", help="The group by column to use.")
@click.option("-c", "--count", default="tweet", help="The value to count by group.")
@click.option("-mc", "--min-count", default=5000, help="Only present counts above this minimum count")
@click.argument("input-data", required=True)
def describe_tweets(ctx, filters, output_dir, groupby, count, min_count, input_data):
    """Describe tweets.

    Use this command to print counts of tweets per place_name.
    """
    describer = Describer(output_dir, groupby, count, "tweets", filters, min_count)
    result = describer.describe(input_data)
    print(result)


@describe.command("products")
@click.pass_context
@click.option("-o", "--output-dir", required=True, help="Path where the describe results will be printed out.")
@click.option("-g", "--groupby", default="categories", help="The group by column to use.")
@click.option("-c", "--count", default="document", help="The value to count by group.")
@click.option("-mc", "--min-count", default=5000, help="Only present counts above this minimum count")
@click.argument("input-data", required=True)
def describe_products(ctx, output_dir, groupby, count, min_count, input_data):
    """Describe products.

    Use this command to print counts of products per category.
    """
    describer = Describer(output_dir, groupby, count, "documents", min_count=min_count)
    result = describer.describe(input_data)
    print(result)


@wtsp.group()
@click.pass_context
def train(ctx):
    """Train module.

    Use this module to train the different models used for the project.
    """
    pass


@train.command("tweets")
@click.pass_context
@click.option('-m', '--model', default="nearest-neighbors",
              help='Executes a model training in the tweets domain')
@click.option("-f", "--filters", required=True,
              help="Filters to use over the data set columns to narrow down the load.")
@click.option('-p', "--params", required=True, help="Model parameters")
@click.option("-o", "--output-dir", required=True, help="Path where the describe results will be printed out.")
@click.argument('input_data')
def train_tweets(ctx, model, filters, params, output_dir, input_data):
    r"""Train ML models within the tweets domain.

    Provide the model to train via the --model option (default to 'nearest-neighbors')

    KWARGS: Depending on the model to train, the arguments my vary.
    Provide them as a comma separated key=value argument string, e.g.,
    key1=value1,key2=value2. Arguments with (*) are mandatory

    For model 'nearest-neighbors':

        n_neighbors*         The number of neighbors to consider
        location_column*     the location column with the geometry
    """
    trainer = TweetsTrainer(model, filters, params, output_dir)
    result = trainer.train(input_data)
    print(result)


@train.command("products")
@click.pass_context
@click.option("-m", "--model", required=True, help="Model type to train against he products")
@click.option('-p', "--params", required=True, help="Model parameters")
@click.argument('input_data')
def train_products(ctx, model, params, input_data):
    r"""Train ML models for products.

    Trains the model requested with --model option and using
    the hyper parameters in the form:

    KWARGS: Depending on the model to train, the arguments my vary.
    Provide them as a comma separated key=value argument string, e.g.,
    key1=value1,key2=value2. Arguments with (*) are mandatory.
    """
    work_dir = ctx.obj["WORK_DIR"]
    trainer = ProductsTrainer(work_dir, model, params)
    try:
        result = trainer.train(input_data)
        print(result)
    except ModelTrainingException as e:
        print(e)


@wtsp.group()
@click.pass_context
def transform(ctx):
    """Transform module.

    Use this module to transform data using the trained models.

    Note: You need to first execute the train module first.
    """
    pass
