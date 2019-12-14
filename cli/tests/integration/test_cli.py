"""CLI smoke tests."""

from click.testing import CliRunner
from wtsp.cli import cli


def test_wtsp_command():
    runner = CliRunner()
    result = runner.invoke(cli.wtsp, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.wtsp.__doc__.split())


def test_data_command():
    runner = CliRunner()
    result = runner.invoke(cli.data, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.data.__doc__.split())


def test_data_load_command():
    runner = CliRunner()
    result = runner.invoke(cli.data_load, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.data_load.__doc__.split())


def test_data_list_command():
    runner = CliRunner()
    result = runner.invoke(cli.data_list, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.data_list.__doc__.split())


def test_data_head_command():
    runner = CliRunner()
    result = runner.invoke(cli.data_head, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.data_head.__doc__.split())


def test_data_delete_command():
    runner = CliRunner()
    result = runner.invoke(cli.data_delete, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.data_delete.__doc__.split())


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
    assert all(item in result.output.split() for item in cli.train_tweets.__doc__.split())


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


def test_export_command():
    runner = CliRunner()
    result = runner.invoke(cli.export, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export.__doc__.split())


def test_export_tweet_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_tweet, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_tweet.__doc__.split())


def test_export_tweet_stats_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_tweet_stats, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_tweet_stats.__doc__.split())


def test_export_tweet_map_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_tweet_map, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_tweet_map.__doc__.split())


def test_export_product_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_product, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_product.__doc__.split())


def test_export_product_stats_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_product_stats, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_product_stats.__doc__.split())


def test_export_product_metrics_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_product_metrics, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_product_metrics.__doc__.split())


def test_export_wtsp_command():
    runner = CliRunner()
    result = runner.invoke(cli.export_wtsp, ['--help'])
    assert result.exit_code == 0
    assert all(item in result.output.split() for item in cli.export_wtsp.__doc__.split())
