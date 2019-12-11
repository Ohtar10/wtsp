from click.testing import CliRunner
from wtsp.cli import cli

"""Functional and integration tests over all the CLI"""


def test_help_command():
    runner = CliRunner()
    result = runner.invoke(cli.wtsp, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.wtsp.__doc__.split())


def test_describe_help_command():
    runner = CliRunner()
    result = runner.invoke(cli.describe, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.describe.__doc__.split())


def test_train_help_command():
    runner = CliRunner()
    result = runner.invoke(cli.train, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.train.__doc__.split())


def test_transform_help_command():
    runner = CliRunner()
    result = runner.invoke(cli.transform, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.transform.__doc__.split())


def test_report_help_command():
    runner = CliRunner()
    result = runner.invoke(cli.report, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.report.__doc__.split())

