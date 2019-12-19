"""CLI implementation."""

import click
import logging
from wtsp import utils
from wtsp.__version__ import __version__
from wtsp.describe.describe import Describer
from wtsp.train.base import Trainer
from wtsp.core.base import DEFAULT_WORK_DIR


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
    return describer.describe(input_data)


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
    return describer.describe(input_data)


@wtsp.group()
@click.pass_context
def train(ctx):
    """Train module.

    Use this module to train the different models used for the project.
    """
    pass


@train.command("tweets")
@click.pass_context
@click.option('-m', '--model', required=True, help='Executes a model training in the tweets domain')
@click.argument('input_file')
@click.argument('kwargs', required=True, nargs=-1)
def train_tweets(ctx, model, input_file, kwargs):
    r"""Train ML models within the tweets domain.

    Provide the model to train via the --model option

    KWARGS: Depending on the model to train, the arguments my vary.
    Provide them as a comma separated key=value argument string, e.g.,
    key1=value1,key2=value2. Arguments with (*) are mandatory

    For model 'nearest-neighbor':

        n_neighbors*         The number of neighbors to consider\n
        location*            The base location to filter the data points (tweets.place_name)
    """
    work_dir = ctx['WORKDIR']
    debug = ctx['DEBUG']
    trainer = Trainer(work_dir, debug, "tweets", model)
    args = utils.parse_kwargs(kwargs)
    return trainer.run(input_file, **args)


@train.command("products")
@click.pass_context
@click.option("-m", "--model", required=True, help="Model type to train against he products")
@click.option("-d", "--data", required=True, help="Data set name to use in training")
@click.argument("kwargs", required=True, nargs=-1)
def train_products(ctx, model, data, kwargs):
    r"""Train ML models for products.

    Trains the model requested with --model option and using
    the hyper parameters in the form:

    KWARGS: Depending on the model to train, the arguments my vary.
    Provide them as a comma separated key=value argument string, e.g.,
    key1=value1,key2=value2. Arguments with (*) are mandatory.
    """
    pass


@wtsp.group()
@click.pass_context
def transform(ctx):
    """Transform module.

    Use this module to transform data using the trained models.

    Note: You need to first execute the train module first.
    """
    pass
