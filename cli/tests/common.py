import os.path
import shutil

ASSETS_PATH = "tests/assets"

# input data variables
RAW_TWEETS_PATH = f"{ASSETS_PATH}/tweets"
RAW_PRODUCTS_PATH = f"{ASSETS_PATH}/products"

# application variables
TEST_WORK_DIR_PATH = f"{ASSETS_PATH}/test_work_dir"


def get_tweets_work_dir(base_path=TEST_WORK_DIR_PATH):
    return f"{base_path}/tweets"


def get_tweets_nn_dir(base_path = get_tweets_work_dir()):
    return f"{base_path}/nnm"


def get_full_path(file, relative):
    main_script_dir = os.path.dirname(file)
    return os.path.join(main_script_dir, relative)


def delete_path(path):
    shutil.rmtree(path, ignore_errors=True)