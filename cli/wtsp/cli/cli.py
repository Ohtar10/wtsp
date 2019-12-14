"""CLI implementation."""

import click
from wtsp import utils
from wtsp.__version__ import __version__
from wtsp.train.base import Trainer


def docstring_parameter(*sub):
    """Decorate the main click command to format the docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output.')
@click.option('-wd', '--work-dir', default='~/.wtsp',
              help='Which folder to use as working directory. Default to ~/.wtsp')
@click.pass_context
@docstring_parameter(__version__)
def wtsp(ctx, debug, work_dir):
    """Where To Sell Products (wtsp) {0}."""
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['WORKDIR'] = work_dir


@wtsp.group()
@click.pass_context
def data(ctx):
    """Data module.

    Use this module to load and manage the data
    that you want to use.
    """
    pass


@data.command("load")
@click.pass_context
@click.option("-d", "--domain", required=True, help="Data set domain to load. <tweets|products>")
@click.option("-n", "--name", required=True, help="Name to use to identify the data set")
@click.option("-o", "--overwrite", is_flag=True)
@click.argument("path")
def data_load(ctx, domain, name, overwrite, path):
    """Data load.

    Load data into the local tool metadata with a name.
    This command will apply minor transformations on the
    data to enable easier process in later stages.
    """
    pass


@data.command("list")
@click.pass_context
def data_list(ctx):
    """Data list.

    List the current loaded data including
    intermediate results of later stages.
    """
    pass


@data.command("head")
@click.pass_context
@click.option("-n", "--name", required=True, help="Name of the data registry to display")
@click.option("-t", "--top", default=10, help="Number of rows to display")
def data_head(ctx, name, top):
    """Data head.

    Displays the top n (default 10) rows
    of the data set corresponding to the given
    name.
    """
    pass


@data.command("delete")
@click.pass_context
@click.option("-n", "--name", required=True, help="Name of the data registry to delete")
def data_delete(ctx, name):
    """Data delete.

    Deletes a data registry and its content.
    """
    pass


@wtsp.group()
@click.pass_context
def describe(ctx):
    """Describe module.

    Use this module to generate descriptive data 
    that might help you to take decisions.
    """
    pass


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


@wtsp.group()
@click.pass_context
def export(ctx):
    """Export module.

    Use this module to export in different formants the results of
    the previous two modules.

    Note: You need to first execute one or both of the previous
    modules in order to invoke reports.
    """
    pass


@export.group("tweet")
@click.pass_context
def export_tweet(ctx):
    """Export tweet.

    Use this command to export tweet related reports, plots
    and data.
    """
    pass


@export_tweet.command("stats")
@click.pass_context
@click.option("-d", "--data", required=True, help="Data set or result to export")
@click.argument("output_path", required=True)
def export_tweet_stats(ctx, data, output_path):
    """Tweet stats.

    Exports a handful of basic stats and plots about
    the tweets data sets.
    """
    pass


@export_tweet.command("map")
@click.pass_context
@click.option("-d", "--data", required=True, help="Data set or result to export")
@click.argument("output_path", required=True)
def export_tweet_map(ctx, data, output_path):
    """Tweet map.

    Export different forms of tweet maps as HTML including
    raw tweets locations, locations with cluster colors and
    cluster polygons.
    """
    pass


@export.group("product")
@click.pass_context
def export_product(ctx):
    """Export product.

    Exports different forms of metrics results
    related to product models and data.
    """
    pass


@export_product.command("stats")
@click.pass_context
@click.option("-d", "--data", required=True, help="Data set or result to export")
@click.argument("output_path", required=True)
def export_product_stats(ctx, data, output_path):
    """
    Product stats.

    Exports basic statistics about the product
    data sets.
    """
    pass


@export_product.command("metrics")
@click.pass_context
@click.option("-d", "--data", required=True, help="Data set or result to export")
@click.argument("output_path", required=True)
def export_product_metrics(ctx, data, output_path):
    """
    Product metrics.

    Export metric results related to the product
    models.
    """
    pass


@export.command("wtsp")
@click.pass_context
@click.option("-d", "--data", required=True, help="Data set or result to export")
@click.argument("output_path", required=True)
def export_wtsp(ctx, data, output_path):
    """Export wtsp results.

    Exports the final result of all the process.
    Includes a cluster classification and metadata
    and an HTML map with the final classifications.
    """
    pass
