import os

from click.testing import CliRunner

from tests import common, tests_path
from tests.common import copy_folder_recursively
from wtsp.cli import cli


def test_transform_where_to_sell_products():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)

    models_path = f"{output_path}/products/models/"
    # we are going to assume the working directory already has a embeddings model trained
    model_assets_path = common.get_full_path(tests_path, common.ASSETS_PATH)
    copy_folder_recursively(f"{model_assets_path}/products", models_path)

    params = "center='34;-118',eps=0.04,n_neighbors=2,location_column=location_geometry,min_score=0.1"
    result = runner.invoke(cli.wtsp, ["--work-dir",
                                      output_path,
                                      "predict",
                                      "clusters",
                                      "--filters",
                                      "place_name='Los Angeles'",
                                      "--params",
                                      params,
                                      input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    result_dir = f"{output_path}/where_to_sell_in/place_name=Los Angeles"
    assert os.path.exists(result_dir)
    # and the content
    assert os.path.exists(f"{result_dir}/classified_clusters.csv")
    assert os.path.exists(f"{result_dir}/classified_clusters.html")
    common.delete_path(output_path)


def test_transform_no_filters_should_fail():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    params = "center='34;-118',eps=0.04,n_neighbors=2,location_column=location_geometry,min_score=0.1"
    result = runner.invoke(cli.wtsp, ["--work-dir",
                                      output_path,
                                      "predict",
                                      "clusters",
                                      "--params",
                                      params,
                                      input_data])
    assert result.exit_code != 0
    assert "Error: Missing option '-f' / '--filters'" in result.output


def test_transform_no_params_should_fail():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.wtsp, ["--work-dir",
                                      output_path,
                                      "predict",
                                      "clusters",
                                      "--filters",
                                      "place_name='Los Angeles'",
                                      input_data])
    assert result.exit_code != 0
    assert "Error: Missing option '-p' / '--params'" in result.output


def test_transform_no_input_data_should_fail():
    runner = CliRunner()
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    params = "center='34;-118',eps=0.04,n_neighbors=2,location_column=location_geometry,min_score=0.1"
    result = runner.invoke(cli.wtsp, ["--work-dir",
                                      output_path,
                                      "predict",
                                      "clusters",
                                      "--params",
                                      params,
                                      "--filters",
                                      "place_name='Los Angeles'"])
    assert result.exit_code != 0
    assert "Error: Missing argument 'INPUT_DATA'" in result.output


def test_transform_embeddings():
    runner = CliRunner()
    input_path = common.get_full_path(tests_path, common.RAW_PRODUCTS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    models_path = f"{output_path}/products/models/"

    # we are going to assume the working directory already has a embeddings model trained
    model_assets_path = common.get_full_path(tests_path, common.ASSETS_PATH)
    copy_folder_recursively(f"{model_assets_path}/products", models_path)

    result = runner.invoke(cli.wtsp, [
        "--work-dir",
        output_path,
        "predict",
        "embeddings",
        input_path
    ])

    assert result.exit_code == 0
    # validate the existence of the output files
    result_embeddings = f"{output_path}/embeddings/document_embeddings.npz"
    assert os.path.exists(result_embeddings)
    result_cat_encoder = f"{output_path}/embeddings/category_encoder.save"
    assert os.path.exists(result_cat_encoder)
    common.delete_path(output_path)
