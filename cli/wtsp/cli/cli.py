import click
from wtsp import utils
from wtsp.__version__ import __version__
from wtsp.train.base import Trainer


def docstring_parameter(*sub):
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
    """Where To Sell Products (wtsp) {0}"""

    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['WORKDIR'] = work_dir


@wtsp.group()
@click.pass_context
def describe(ctx):
    """Describe module.

    Use this module to generate descriptive data that might help you to take decisions."""
    pass


@wtsp.group()
@click.pass_context
def train(ctx):
    """Train module.

    Use this module to train the different models used for the project."""
    pass


@train.command("tweets")
@click.pass_context
@click.option('-m', '--model', required=True, help='Executes a model training in the tweets domain')
@click.argument('input')
@click.argument('kwargs', required=True, nargs=-1)
def train_tweets(ctx, model, input_file, kwargs):
    """Trains ML models within the tweets domain.

    Provide the model to train via the --model option

    KWARGS: Depending on the model to train, the arguments my vary.
    Provide them as a comma separated key=value argument string, e.g.,
    key1=value1,key2=value2. Arguments with (*) are mandatory

    For model 'nearest-neighbor':

        n_neighbors*         The number of neighbors to consider\n
        location*            The base location to filter the data points (tweets.place_name)"""
    work_dir = ctx['WORKDIR']
    debug = ctx['DEBUG']
    trainer = Trainer(work_dir, debug, model)
    args = utils.parse_kwargs(kwargs)
    return trainer.run(input_file, **args)


@wtsp.group()
@click.pass_context
def transform(ctx):
    """Transform module.

    Use this module to transform data using the trained models.

    Note: You need to first execute the train module first."""
    pass


@wtsp.group()
@click.pass_context
def report(ctx):
    """Report module.

    Use this module to export in different formants the results of
    the previous two modules.

    Note: You need to first execute one or both of the previous
    modules in order to invoke reports."""
    pass

