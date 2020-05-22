"""Tests related to the describe command."""
from click.testing import CliRunner
from wtsp.cli import cli
from tests import common, tests_path
import os


def test_describe_tweets():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['tweets',
                                          "--filters",
                                          "country_code=US",
                                          "--output-dir",
                                          output_path,
                                          "--min-count",
                                          10,
                                          input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    tweets_describe_result = f"{output_path}/tweets/country_code=US"
    assert os.path.exists(tweets_describe_result)
    # and the content
    assert os.path.exists(f"{tweets_describe_result}/counts.csv")
    assert os.path.exists(f"{tweets_describe_result}/bar_chart.png")
    common.delete_path(tweets_describe_result)


def test_describe_tweets_no_filters_should_fail():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['tweets',
                                          "--output-dir",
                                          output_path,
                                          "--min-count",
                                          10,
                                          input_data])
    assert result.exit_code != 0
    assert "Error: Missing option '-f' / '--filters'" in result.output


def test_describe_tweets_no_input_data_should_fail():
    runner = CliRunner()
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['tweets',
                                          "--filters",
                                          "country_code=US",
                                          "--output-dir",
                                          output_path,
                                          "--min-count",
                                          10])
    assert result.exit_code != 0
    assert "Error: Missing argument 'INPUT_DATA'" in result.output


def test_describe_products():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_PRODUCTS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['products',
                                          "--output-dir",
                                          output_path,
                                          "--groupby",
                                          "categories",
                                          "--min-count",
                                          10,
                                          input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    products_describe_result = f"{output_path}/documents"
    assert os.path.exists(products_describe_result)
    # and the content
    assert os.path.exists(f"{products_describe_result}/counts.csv")
    assert os.path.exists(f"{products_describe_result}/bar_chart.png")
    common.delete_path(products_describe_result)


def test_describe_products_with_explode():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_PRODUCTS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['products',
                                          "--output-dir",
                                          output_path,
                                          "--groupby",
                                          "categories",
                                          "--min-count",
                                          10,
                                          "--explode",
                                          input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    products_describe_result = f"{output_path}/documents"
    assert os.path.exists(products_describe_result)
    # and the content
    assert os.path.exists(f"{products_describe_result}/counts.csv")
    assert os.path.exists(f"{products_describe_result}/bar_chart.png")
    common.delete_path(products_describe_result)


def test_describe_products_no_input_data_should_fail():
    runner = CliRunner()
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.describe, ['products',
                                          "--output-dir",
                                          output_path,
                                          "--min-count",
                                          10])
    assert result.exit_code != 0
    assert "Error: Missing argument 'INPUT_DATA'" in result.output
