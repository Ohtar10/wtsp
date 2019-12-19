import os
from click.testing import CliRunner
from tests import common, tests_path
from wtsp.cli import cli


def test_train_tweets_n_neighbors():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    result = runner.invoke(cli.train_tweets, ['--model',
                                              "nearest-neighbors",
                                              "--filters",
                                              "place_name='Los Angeles'",
                                              "--params",
                                              "n_neighbors=10,location_column=location_geometry",
                                              "--output-dir",
                                              output_path,
                                              input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    result_dir = f"{output_path}/tweets/place_name=Los Angeles"
    assert os.path.exists(result_dir)
    # and the content
    assert os.path.exists(f"{result_dir}/nearest_neighbors.png")
    assert os.path.exists(f"{result_dir}/scatter_plot.png")
    common.delete_path(result_dir)


def test_train_product_embeddings():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_PRODUCTS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)
    params = "label_col=categories,doc_col=document,lr=0.0002,epochs=10,vec_size=300,alpha=0.025,min_alpha=0.00025"
    result = runner.invoke(cli.wtsp, ['--work-dir',
                                      output_path,
                                      "train",
                                      "products",
                                      "--model",
                                      "embeddings",
                                      "--params",
                                      params,
                                      input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    result_dir = f"{output_path}/products/models/embeddings"
    assert os.path.exists(result_dir)
    # and the content
    assert os.path.exists(f"{result_dir}/d2v_model.model")
    common.delete_path(result_dir)