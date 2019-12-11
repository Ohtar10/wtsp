import pytest
from click.testing import CliRunner
from wtsp.cli import cli

"""Functional and integration tests over all the CLI"""


def test_train_tweets_clusters():
    pytest.xfail("Not yet implemented")


def test_train_tweets_nearest_neighbors():
    runner = CliRunner()
    args = ['--model=nearest-neighbors', '<input-file>', 'n_neighbors=2,location="Los Angeles"']
    result = runner.invoke(cli.train_tweets, args)
    assert result.exit_code == 0
    assert result.output == "Nearest Neighbors trained and saved successfully. Use the report option to see the results"
