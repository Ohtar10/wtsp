"""Tests related to tweet model training."""

import os.path

import pytest

from tests import common
from tests import tests_path
from wtsp.exceptions import InvalidArgumentException
from wtsp.train.tweets import GeoTweetsNearestNeighbors


def test_train_tweets_nearest_neighbors_ok():
    input_data = common.get_full_path(tests_path, common.RAW_TWEETS_PATH)
    trainer = GeoTweetsNearestNeighbors(common.TEST_WORK_DIR_PATH,
                                        debug=True,
                                        n_neighbors=2,
                                        country="United States",
                                        place="Los Angeles")
    result = trainer.run(input_data)

    assert result == GeoTweetsNearestNeighbors.SUCCESS_MESSAGE
    # check the output has data
    output_path = common.get_tweets_nn_dir()
    assert os.path.exists(output_path)
    # TODO check the output is the expected
    assert False


@pytest.mark.parametrize(
    "inputs,n_neighbors,country,place",
    [
        (None, None, None, None),
        ("", "", "", ""),
        ("blah", "blah", "blah", "blah"),
        (common.RAW_TWEETS_PATH, None, None, None),
        (common.RAW_TWEETS_PATH, "", "", ""),
        (common.RAW_TWEETS_PATH, "blah", "blah", "blah"),
        (common.RAW_TWEETS_PATH, 2, None, None),
        (common.RAW_TWEETS_PATH, 2, "", "")
    ]
)
def test_train_tweets_nearest_neighbors_fail_on_invalid_params(inputs, n_neighbors, country, place):
    trainer = GeoTweetsNearestNeighbors(common.TEST_WORK_DIR_PATH,
                                        debug=True,
                                        n_neighbors=n_neighbors,
                                        country=country,
                                        place=place)
    with pytest.raises(InvalidArgumentException):
        trainer.run(inputs)
