import os
from click.testing import CliRunner
from tests import common, tests_path
from tests.common import copy_folder_recursively
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
    common.delete_path(output_path)


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
    common.delete_path(output_path)


def test_train_product_classifier():
    runner = CliRunner()
    input_data = common.get_full_path(tests_path, common.RAW_PRODUCTS_PATH)
    output_path = common.get_full_path(tests_path, common.TEST_WORK_DIR_PATH)

    embeddings_path = common.get_full_path(tests_path, common.EMBEDDINGS_PATH)
    models_path = f"{output_path}/products/models/embeddings"
    # we are going to assume the working directory already has a embeddings model trained
    copy_folder_recursively(embeddings_path, models_path)

    params = "label_col=categories,doc_col=document,classes=10,test_size=0.3," \
             "lr=0.0002,epochs=10,vec_size=300,alpha=0.025,min_alpha=0.00025"
    result = runner.invoke(cli.wtsp, ['--work-dir',
                                      output_path,
                                      "train",
                                      "products",
                                      "--model",
                                      "classifier",
                                      "--params",
                                      params,
                                      input_data])
    assert result.exit_code == 0
    # validate the existence of the output directory
    result_dir = f"{output_path}/products/models/classifier"
    assert os.path.exists(result_dir)
    # and the content
    assert os.path.exists(f"{result_dir}/category_encoder.model")
    assert os.path.exists(f"{result_dir}/prod_classifier-def.yaml")
    assert os.path.exists(f"{result_dir}/prod_classifier-weights.h5")
    assert os.path.exists(f"{result_dir}/training_history.png")
    assert os.path.exists(f"{result_dir}/classification_report.png")
    common.delete_path(output_path)
