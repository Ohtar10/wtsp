from click.testing import CliRunner
from wtsp.cli import cli

"""Functional and integration tests over all the CLI"""


def test_train_tweets_clusters():
    runner = CliRunner()
    result = runner.invoke(cli.train, ['--clusters', '--input'])
    assert result.exit_code == 0
    assert result.output == "Clusters trained and saved successfully."


def test_train_tweets_nearest_neighbors():
    runner = CliRunner()
    result = runner.invoke(cli.train, ['--nearest-neighbors', '--input'])
    assert result.exit_code == 0
    assert result.output == "Nearest Neighbors trained and saved successfully. Use the report option to see the results"
