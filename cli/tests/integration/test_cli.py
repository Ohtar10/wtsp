"""CLI smoke tests."""

from click.testing import CliRunner
from wtsp.cli import cli


def test_wtsp_command():
    runner = CliRunner()
    result = runner.invoke(cli.wtsp, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.wtsp.__doc__.split())


def test_describe_command():
    runner = CliRunner()
    result = runner.invoke(cli.describe, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.describe.__doc__.split())


def test_train_command():
    runner = CliRunner()
    result = runner.invoke(cli.train, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.train.__doc__.split())


def test_train_tweets_command():
    runner = CliRunner()
    result = runner.invoke(cli.train_tweets, ['--help'])
    assert result.exit_code == 0


def test_train_products_command():
    runner = CliRunner()
    result = runner.invoke(cli.train_products, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.train_products.__doc__.split())


def test_transform_command():
    runner = CliRunner()
    result = runner.invoke(cli.transform, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.transform.__doc__.split())

