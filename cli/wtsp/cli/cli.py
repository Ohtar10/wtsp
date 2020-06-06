"""CLI implementation."""

import logging

import click
import sys
from wtsp.__version__ import __version__
from wtsp.core.base import DEFAULT_WORK_DIR
from wtsp.describe.describe import Describer
from wtsp.exceptions import WTSPBaseException
from wtsp.train.products import ProductsTrainer
from wtsp.train.tweets import TweetsTrainer
from wtsp.transform.transformers import WhereToSellProductsTransformer, EmbeddingsTransformer


def docstring_parameter(*sub):
    """Decorate the main click command to format the docstring."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug output.')
@click.option('-wd', '--work-dir', default=DEFAULT_WORK_DIR,
              help='Which folder to use as working directory. Default to ~/wtsp')
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
@click.option("-o", "--output-dir", help="Path where the describe results will be printed out.")
@click.option("-g", "--groupby", default="place_name", help="The group by column to use.")
@click.option("-c", "--count", default="tweet", help="The value to count by group.")
@click.option("-mc", "--min-count", default=1000, help="Only present counts above this minimum count")
@click.argument("input-data", required=True)
def describe_tweets(ctx, filters, output_dir, groupby, count, min_count, input_data):
    """Describe tweets.

    Use this command to print counts of tweets per place_name.
    """
    if not output_dir:
        output_dir = ctx.obj["WORK_DIR"]

    describer = Describer(output_dir, groupby, count, "tweets", filters, min_count)
    try:
        result = describer.describe(input_data)
        print(result)
    except WTSPBaseException as e:
        print(e)
        sys.exit(e)


@describe.command("products")
@click.pass_context
@click.option("-o", "--output-dir", help="Path where the describe results will be printed out.")
@click.option("-g", "--groupby", default="categories", help="The group by column to use.")
@click.option("-c", "--count", default="document", help="The value to count by group.")
@click.option("-mc", "--min-count", default=5000, help="Only present counts above this minimum count")
@click.option('--explode', default=False, is_flag=True, help='Enable exploding of the groupby column.')
@click.argument("input-data", required=True)
def describe_products(ctx, output_dir, groupby, count, min_count, explode, input_data):
    """Describe products.

    Use this command to print counts of products per category.
    """
    if not output_dir:
        output_dir = ctx.obj["WORK_DIR"]

    describer = Describer(output_dir, groupby, count, "documents", min_count=min_count, explode=explode)
    try:
        result = describer.describe(input_data)
        print(result)
    except WTSPBaseException as e:
        print(e)
        sys.exit(e)


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
@click.option("-o", "--output-dir", help="Path where the describe results will be printed out.")
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
    if not output_dir:
        output_dir = ctx.obj["WORK_DIR"]

    trainer = TweetsTrainer(model, filters, params, output_dir)
    try:
        result = trainer.train(input_data)
        print(result)
    except WTSPBaseException as e:
        print(e)
        sys.exit(e)


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

    For model 'embeddings':

        label_col           The column name that holds the label
        doc_col             The column name that holds the document
        lr                  The learning rate
        epochs              The epochs/iterations to train the Doc2Vec
        vec_size            The desired embedding vector size
        alpha               Alpha parameter to pass to Doc2Vec
        min_alpha           Minimum alpha parameter for the Doc2Vec model

    For model 'classifier':

        label_col           The column name that holds the label
        doc_col             The column name that holds the document
        classes             The amount of expected classes to find
        test_size           The size of the testing set for hold-out
        lr                  The learning rate
        epochs              The epochs/iterations to train the Doc2Vec
        vec_size            The desired embedding vector size
    """
    work_dir = ctx.obj["WORK_DIR"]
    trainer = ProductsTrainer(work_dir, model, params)
    try:
        result = trainer.train(input_data)
        print(result)
    except WTSPBaseException as e:
        print(e)
        sys.exit(e)


@wtsp.group()
@click.pass_context
def predict(ctx):
    """Predict module.

    Use this module to make predictions over tweets clusters or document
    embeddings.
    """
    pass


@predict.command("clusters")
@click.pass_context
@click.option("-f", "--filters", required=True,
              help="Filters to use over the data set columns to narrow down the load.")
@click.option('-p', "--params", required=True, help="Model parameters")
@click.argument("input_data")
def predict_clusters(ctx, filters, params, input_data):
    """Predict clusters.

    Use this command to transform twitter data using the trained models.

    Note: You need to train the embeddings and classifier first,
    ensure that these models are already trained in your working
    directory.

    Params:

        center          The geographic coordinates to center the map output in format: lat;long, e.g., 34.1;118.3
        eps             The epsilon value to train the DBSCAN cluster based on the nearest neighbors model.
        n_neighbors     The minimum number of neighbors per cluster
        location_column The name of the column containing the location to use
        min_score       The minimum classification score to show on clusters.
    """
    work_dir = ctx.obj["WORK_DIR"]
    transformer = WhereToSellProductsTransformer(work_dir, filters, params)
    try:
        result = transformer.transform(input_data)
        print(result)
    except WTSPBaseException as e:
        print(e)
        sys.exit(e)
    except Exception as e:
        print(e)


@predict.command("embeddings")
@click.pass_context
@click.option("-lc", "--label-column", default="categories", help="Column name for the labels")
@click.option("-dc", "--document-column", default="document", help="Column name for the documents")
@click.option("--train-test-split/--no-train-test-split", default=False,
              help="Sets if after transform you want to split into training and testing sets.")
@click.option("-ts", "--test-size", default=0.3, help="The fraction to use as test size, default to 0.3")
@click.argument("input_data")
def predict_embeddings(ctx, label_column,
                       document_column,
                       train_test_split,
                       test_size,
                       input_data):
    """Predict Embeddings.

    Use this command to transform raw product documents into embeddings.

    Note: You need to train the embeddings and classifier first,
    ensure that these models are already trained in your working
    directory.
    """
    work_dir = ctx.obj["WORK_DIR"]
    try:
        transformer = EmbeddingsTransformer(work_dir,
                              label_column=label_column,
                              document_column=document_column,
                              tt_split=train_test_split,
                              test_size=test_size,
                              )
        transformer.transform(input_data)

    except Exception as e:
        print(e)
        sys.exit(e)
